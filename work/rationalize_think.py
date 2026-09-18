#!/usr/bin/env python3
"""用底模替角色扮演資料集補上 <think> 推理。

背景：Aratako 的 roleplay 資料只有 role/content，沒有任何推理內容。用這種資料
訓練會把「不思考、快收尾」學進去。這個腳本把同樣的角色設定和對話歷史餵給
底模（thinking 模式），把它的推理存進 assistant message 的 reasoning_content；
Qwen3.8 的 chat template 會把它渲染成真正的 <think> 區塊，訓練時就能保住
思考習慣。

兩種模式：
  free         （預設）底模自己思考、自己回覆；該回合的台詞也換成底模的回覆。
               推理和台詞完全一致、沒有任何「被指定台詞」的痕跡。歷史回合仍是
               資料集原文，角色扮演風格照學。
  rationalize  把資料集的台詞先告訴底模，請它寫出導向這句台詞的推理。底模常在
               推理裡提到「指示」「已決定的返答」，只能靠逐句清洗，僅供實驗。

兩個子命令：
  generate  透過 Ollama /api/chat，或 OpenAI 相容的 /v1/chat/completions（llama-server、
            vLLM；--api openai）產生，逐筆追加到 generated.jsonl（可中斷續跑）
  build     套用過濾門檻，輸出訓練用的 train.jsonl（messages 格式，可直接餵 runner）

範例：
  python rationalize_think.py generate --samples 4000 --parallel 4
  python rationalize_think.py generate --api openai --ollama http://pod:11434 --parallel 12
  python rationalize_think.py build
"""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_DATASET = "Aratako/Synthetic-Japanese-Roleplay-NSFW-DeepSeek-V3-0324-20k-formatted"
DEFAULT_MODEL = "qwen3.8-27b-heretic-6fcab5-kl:latest"
DEFAULT_OUT = Path(__file__).resolve().parent / "datasets" / "aratako-rp-think"

RATIONALIZE_INSTRUCTION = (
    "\n\n【思考の指示】\n"
    "あなた（キャラクター）は、次にこう言うことをすでに心の中で決めています：\n"
    "---\n{reply}\n---\n"
    "返答の前に、キャラクター本人として思考してください：いまの状況、相手の言葉の意図、"
    "自分の感情や狙い、そしてなぜこの言い方・この行動になるのか。\n"
    "守ること：\n"
    "・思考の中で、この指示や「返答が決まっている／与えられている」ことに触れない。\n"
    "・別の返答案を下書きしたり比較したりしない。上の返答そのものに至る考えを書く。\n"
    "・思考が終わったら、上の返答を一字一句変えずにそのまま出力する。"
)

# rationalize 模式：推理裡提到「被指定的台詞」的句子不能拿來訓練。
META_PATTERN = re.compile(
    r"指示|与えられ|決まって|決められ|既定|指定|提示され|提供|台本|スクリプト|そのまま出力|一字一句"
    r"|指令|给定|給定|必须输出|必須輸出|原样|原樣|一字不差|逐字|系统提示|系統提示"
    r"|instruction|predetermined|verbatim|word for word|exactly as|must output|as requested"
    r"|given (?:reply|response|line|text)|the (?:reply|response|text) is already|system prompt",
    re.IGNORECASE,
)
THINK_TAG = re.compile(r"</?think>")
SENTENCE_SPLIT = re.compile(r"(?<=[。．！？!?])\s*|\n+")


def load_rows(dataset: str, split: str):
    from datasets import load_dataset

    return load_dataset(dataset, split=split)


def assistant_indices(messages) -> list[int]:
    return [
        i for i, m in enumerate(messages)
        if m["role"] == "assistant" and i > 0 and messages[i - 1]["role"] == "user"
    ]


def pick_turn(messages, mode: str, rng: random.Random) -> int | None:
    candidates = assistant_indices(messages)
    if not candidates:
        return None
    return candidates[-1] if mode == "last" else rng.choice(candidates)


def build_request(messages, turn: int, mode: str, model: str, num_ctx: int, num_predict: int) -> dict:
    history = [dict(role=m["role"], content=m["content"]) for m in messages[:turn]]
    if mode == "rationalize":
        instruction = RATIONALIZE_INSTRUCTION.format(reply=messages[turn]["content"].strip())
        if history and history[0]["role"] == "system":
            history[0]["content"] = history[0]["content"].rstrip() + instruction
        else:
            history.insert(0, {"role": "system", "content": instruction.strip()})
    return {
        "model": model,
        "messages": history,
        "think": True,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "num_ctx": num_ctx, "num_predict": num_predict,
        },
    }


API_CHOICES = ("ollama", "openai")


def to_openai_payload(payload: dict) -> dict:
    """把 Ollama /api/chat 的請求轉成 OpenAI 相容的 /v1/chat/completions（llama-server、vLLM）。

    Ollama 對 qwen35 架構強制單路；llama-server 的 -np 才能真的並行，而它的 Ollama
    相容層只有 /api/tags 之類的列表端點、沒有 /api/chat，所以聊天要走 /v1。
    num_ctx 在 llama-server 是啟動參數（-c），這裡不送。
    """
    options = payload.get("options") or {}
    return {
        "model": payload["model"],
        "messages": payload["messages"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": options.get("num_predict"),
        "temperature": options.get("temperature"),
        "top_p": options.get("top_p"),
        "top_k": options.get("top_k"),
        "chat_template_kwargs": {"enable_thinking": bool(payload.get("think", True))},
    }


LOOP_WINDOW = 160   # 用最後這麼多字當指紋
LOOP_REPEATS = 3    # 指紋在前文再出現這麼多次就判定為迴圈


def looks_looped(text: str, window: int = LOOP_WINDOW, repeats: int = LOOP_REPEATS) -> bool:
    """量化後的底模在長思考裡偶爾會卡進同一段草稿反覆改寫的迴圈；
    最後 window 字若已在前文出現 repeats 次以上，就當作迴圈。"""
    if len(text) < window * (repeats + 1):
        return False
    tail = text[-window:]
    return text.count(tail) > repeats


def merge_chat_stream(lines, abort_on_loop: bool = True) -> dict:
    """把 Ollama /api/chat 的串流（每行一個 JSON）合併成和非串流相同的回應。

    邊收邊看：偵測到思考迴圈就停止讀取，呼叫端關閉連線後 Ollama 會中止生成，
    不必等到 num_predict 用完。這種回應 done_reason 標成 "loop"。
    """
    thinking, content, final = [], [], {}
    think_text, count = "", 0
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.strip()
        if not line:
            continue
        chunk = json.loads(line)
        message = chunk.get("message") or {}
        piece = message.get("thinking") or ""
        thinking.append(piece)
        content.append(message.get("content") or "")
        if chunk.get("done"):
            final = chunk
            break
        count += 1
        if abort_on_loop and piece and count % 32 == 0:
            think_text = "".join(thinking)
            if looks_looped(think_text):
                final = {"done": True, "done_reason": "loop", "eval_count": count}
                break
    merged = dict(final)
    merged["message"] = {"role": "assistant", "thinking": "".join(thinking), "content": "".join(content)}
    merged.setdefault("done", bool(final))
    return merged


def merge_sse_stream(lines, abort_on_loop: bool = True) -> dict:
    """把 OpenAI 相容的 SSE 串流合併成和 merge_chat_stream 相同形狀的回應。

    llama-server 開 --jinja 且 --reasoning-format 不是 none 時（預設 auto），
    <think> 內容會放在 delta.reasoning_content；vLLM 的 --reasoning-parser 也用同一個欄位。
    """
    thinking, content = [], []
    finish_reason = None
    usage, timings = {}, {}
    count = 0
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.strip()
        if not line.startswith("data:"):
            continue  # 空行與 ": keep-alive" 之類的註解
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if chunk.get("usage"):
            usage = chunk["usage"]
        if chunk.get("timings"):
            timings = chunk["timings"]
        piece = ""
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            piece = delta.get("reasoning_content") or delta.get("reasoning") or ""
            thinking.append(piece)
            content.append(delta.get("content") or "")
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
        count += 1
        if abort_on_loop and piece and count % 32 == 0 and looks_looped("".join(thinking)):
            finish_reason = "loop"
            break
    eval_duration = int(timings["predicted_ms"] * 1e6) if timings.get("predicted_ms") else None
    return {
        "message": {"role": "assistant", "thinking": "".join(thinking), "content": "".join(content)},
        "done": finish_reason is not None,
        "done_reason": finish_reason,
        "eval_count": usage.get("completion_tokens") or timings.get("predicted_n") or count,
        "eval_duration": eval_duration,
    }


# RunPod 之類的 proxy 會擋 Python-urllib 預設的 User-Agent（403），一律自報名字。
HEADERS = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "LoRA-Forge/1.0"}


def request_headers(api_key: str | None = None, accept: str | None = None) -> dict:
    """vLLM 開 --api-key（或 VLLM_API_KEY）時要帶 Bearer token；Ollama、llama-server 會忽略。"""
    headers = dict(HEADERS)
    if accept:
        headers["Accept"] = accept
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def server_alive(base_url: str, timeout: int = 15, api: str = "ollama", api_key: str | None = None) -> bool:
    path = "/api/tags" if api == "ollama" else "/v1/models"
    try:
        request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", headers=request_headers(api_key))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def wait_for_server(base_url: str, max_wait: int, poll: int = 30, api: str = "ollama", api_key: str | None = None) -> bool:
    """伺服器（或 proxy）掛掉時等它回來，最多等 max_wait 秒。"""
    started = time.time()
    while time.time() - started < max_wait:
        if server_alive(base_url, api=api, api_key=api_key):
            return True
        time.sleep(poll)
    return server_alive(base_url, api=api, api_key=api_key)


def load_done_indices(path: Path) -> set[int]:
    """已完成的樣本 idx；連線錯誤的紀錄不算完成，續跑時會重做。"""
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "idx" in record and not record.get("error"):
                done.add(record["idx"])
    return done


def call_ollama(base_url: str, payload: dict, timeout: int, api: str = "ollama", api_key: str | None = None) -> dict:
    # 用串流接收：長思考一筆要跑好幾分鐘，非串流時中間沒有任何資料，
    # RunPod 之類的 HTTP proxy 會把閒置連線切掉；串流讓連線一直有資料。
    started = time.time()
    if api == "openai":
        data = json.dumps(to_openai_payload(payload)).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/chat/completions", data=data,
            headers=request_headers(api_key, accept="text/event-stream"), method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            merged = merge_sse_stream(response)
    else:
        data = json.dumps({**payload, "stream": True}).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/chat", data=data, headers=request_headers(api_key), method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            merged = merge_chat_stream(response)
    if not merged.get("done"):
        raise OSError("串流在收到結束訊號之前就斷了")
    if not merged.get("eval_duration"):
        merged["eval_duration"] = int((time.time() - started) * 1e9)  # vLLM 沒有 timings，用牆鐘估
    return merged


def similarity(a: str, b: str) -> float:
    a, b = a.strip()[:300], b.strip()[:300]
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def clean_thinking(thinking: str) -> tuple[str, list[str]]:
    """逐句移除提到指示的句子，回傳 (清洗後文字, 被移除的句子)。"""
    kept, removed = [], []
    for segment in SENTENCE_SPLIT.split(thinking):
        if not segment or not segment.strip():
            continue
        (removed if META_PATTERN.search(segment) else kept).append(segment.strip())
    return "\n".join(kept).strip(), removed


def evaluate(mode: str, thinking: str, content: str, reply: str) -> dict:
    raw = THINK_TAG.sub("", thinking or "").strip()
    record = {
        "thinking_raw": raw,
        "similarity": round(similarity(content or "", reply), 3),
    }
    if mode == "rationalize":
        cleaned, removed = clean_thinking(raw)
        record.update(
            thinking=cleaned,
            meta_removed=removed,
            removed_ratio=round(1 - len(cleaned) / len(raw), 3) if raw else 1.0,
        )
    else:
        record.update(thinking=raw, meta_removed=[], removed_ratio=0.0)
    record["thinking_chars"] = len(record["thinking"])
    record["content_chars"] = len((content or "").strip())
    return record


def accepted(record: dict, args) -> bool:
    if record.get("error") or record.get("turn") is None:
        return False
    if record.get("done_reason", "stop") != "stop":
        return False
    if (record.get("thinking_chars") or 0) < args.min_think_chars:
        return False
    if args.max_think_chars and (record.get("thinking_chars") or 0) > args.max_think_chars:
        return False
    if record.get("mode") == "rationalize":
        if (record.get("removed_ratio") or 0) > args.max_removed_ratio:
            return False
        if (record.get("similarity") or 0) < args.min_similarity:
            return False
        return True
    return (record.get("content_chars") or 0) >= args.min_content_chars


def generate(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_path = out_dir / "generated.jsonl"
    rows = load_rows(args.dataset, args.split)
    rng = random.Random(args.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    selected = order[: args.samples]

    done = load_done_indices(generated_path)
    todo = [i for i in selected if i not in done]
    print(
        f"資料集 {len(rows):,} 筆，抽樣 {len(selected):,}，已完成 {len(selected) - len(todo):,}，"
        f"待生成 {len(todo):,}（模式 {args.mode}，回合 {args.turn}，並行 {args.parallel}，後端 {args.api}）",
        flush=True,
    )
    if not todo:
        return

    lock = threading.Lock()
    stats = {"ok": 0, "rejected": 0, "error": 0, "think_chars": 0, "content_chars": 0,
             "eval_tokens": 0, "eval_ns": 0}
    started = time.time()

    def work(idx: int) -> dict:
        messages = rows[idx]["messages"]
        turn = pick_turn(messages, args.turn, random.Random(f"{args.seed}-{idx}"))
        if turn is None:
            return {"idx": idx, "turn": None, "mode": args.mode, "error": "沒有 user→assistant 的回合"}
        payload = build_request(messages, turn, args.mode, args.model, args.num_ctx, args.num_predict)
        response = None
        outages = 0
        while response is None:
            last_error = None
            for attempt in range(3):
                try:
                    response = call_ollama(args.ollama, payload, args.timeout, args.api, args.api_key)
                    break
                except urllib.error.HTTPError as exc:
                    if 400 <= exc.code < 500 and exc.code not in (408, 425, 429):
                        # 請求本身被拒（例如 vLLM 的 400：prompt 加 max_tokens 超過 max-model-len），
                        # 重送也不會過，直接記成這一筆的錯誤，不當成伺服器失聯。
                        try:
                            detail = exc.read(300).decode("utf-8", errors="replace")
                        except (OSError, ValueError):
                            detail = ""
                        return {"idx": idx, "turn": turn, "mode": args.mode, "error": f"HTTP {exc.code}: {detail or exc.reason}"}
                    last_error = f"HTTPError: {exc}"
                    time.sleep(5 * (attempt + 1))
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    time.sleep(5 * (attempt + 1))
            if response is not None:
                break
            # 連續三次都失敗：多半是伺服器或 proxy 出狀況（524 逾時、404 pod 重啟、
            # 連線被拒）。等它恢復再重做這一筆；只有等不到才記成錯誤。
            outages += 1
            if outages > args.max_outages:
                return {"idx": idx, "turn": turn, "mode": args.mode, "error": last_error}
            with lock:
                print(f"[idx {idx}] 連線失敗（{last_error}），等待伺服器恢復（最多 {args.outage_wait} 秒）…", flush=True)
            if not wait_for_server(args.ollama, args.outage_wait, api=args.api, api_key=args.api_key):
                return {"idx": idx, "turn": turn, "mode": args.mode, "error": f"伺服器 {args.outage_wait} 秒內沒有恢復；最後錯誤 {last_error}"}
        # 底模的思考長度是雙峰的：同一筆有時 800 token 收尾、有時跑滿上限。
        # 被截斷就重新抽樣幾次，比一律加大 num_predict 便宜得多。
        attempts = 1
        while response.get("done_reason") in ("length", "loop") and attempts <= args.retry_on_length:
            attempts += 1
            try:
                response = call_ollama(args.ollama, payload, args.timeout, args.api, args.api_key)
            except (urllib.error.URLError, TimeoutError, OSError):
                break
        message = response.get("message", {})
        return {
            "idx": idx, "turn": turn, "mode": args.mode, "model": args.model, "attempts": attempts,
            "reply": messages[turn]["content"],
            "content": (message.get("content") or "").strip(),
            **evaluate(args.mode, message.get("thinking", ""), message.get("content", ""), messages[turn]["content"]),
            "eval_count": response.get("eval_count"),
            "eval_duration": response.get("eval_duration"),
            "done_reason": response.get("done_reason"),
        }

    with generated_path.open("a", encoding="utf-8") as sink, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(work, idx) for idx in todo]
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = future.result()
            with lock:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                if record.get("error"):
                    stats["error"] += 1
                elif accepted(record, args):
                    stats["ok"] += 1
                else:
                    stats["rejected"] += 1
                stats["think_chars"] += record.get("thinking_chars") or 0
                stats["content_chars"] += record.get("content_chars") or 0
                stats["eval_tokens"] += record.get("eval_count") or 0
                stats["eval_ns"] += record.get("eval_duration") or 0
                if n % args.report_every == 0 or n == len(todo):
                    elapsed = time.time() - started
                    finished = max(stats["ok"] + stats["rejected"] + stats["error"], 1)
                    tps = stats["eval_tokens"] / (stats["eval_ns"] / 1e9) if stats["eval_ns"] else 0
                    print(
                        f"[{n}/{len(todo)}] 通過 {stats['ok']} 拒絕 {stats['rejected']} 錯誤 {stats['error']}"
                        f" | 平均推理 {stats['think_chars'] // finished} 字、回覆 {stats['content_chars'] // finished} 字"
                        f" | 單串流 {tps:.1f} tok/s | 整體 {n / elapsed * 3600:.0f} 筆/小時"
                        f"，剩餘約 {(len(todo) - n) / max(n / elapsed, 1e-9) / 3600:.1f} 小時",
                        flush=True,
                    )


def build(args) -> None:
    out_dir = Path(args.out)
    generated_path = out_dir / "generated.jsonl"
    train_path = out_dir / "train.jsonl"
    rows = load_rows(args.dataset, args.split)
    chosen: dict[int, dict] = {}
    seen = rejected = 0
    with generated_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # 生成仍在寫入時，最後一行可能只寫了一半
            seen += 1
            if accepted(record, args):
                chosen[record["idx"]] = record
            else:
                rejected += 1
    written = with_think = 0
    with train_path.open("w", encoding="utf-8") as sink:
        for idx in range(len(rows)):
            messages = [dict(role=m["role"], content=m["content"]) for m in rows[idx]["messages"]]
            record = chosen.get(idx)
            if record is None and args.only_with_think:
                continue
            if record is not None:
                turn = record["turn"]
                messages[turn]["reasoning_content"] = record["thinking"]
                if record.get("mode", "free") == "free":
                    # 底模自己的推理配底模自己的回覆；之後的回合是資料集接著原句寫的，
                    # 已經對不上，所以在這裡截斷。
                    messages[turn]["content"] = record["content"]
                    messages = messages[: turn + 1]
                with_think += 1
            sink.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            written += 1
    stats = {
        "seen": seen, "rejected": rejected, "accepted": len(chosen), "written": written,
        "with_think": with_think, "only_with_think": bool(args.only_with_think),
        "min_think_chars": args.min_think_chars, "max_think_chars": args.max_think_chars,
        "min_content_chars": args.min_content_chars,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "build_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"generated.jsonl {seen:,} 筆，拒絕 {rejected:,}，採用 {len(chosen):,}；"
        f"train.jsonl 寫出 {written:,} 筆（含推理 {with_think:,}）→ {train_path}"
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--min-think-chars", type=int, default=200, help="推理至少要有的字數")
    parser.add_argument("--max-think-chars", type=int, default=6000,
                        help="推理最多字數（0 = 不限制）；約 0.7 token/字，6000 字加上對話仍放得進 8192 context")
    parser.add_argument("--min-content-chars", type=int, default=10, help="free 模式：回覆至少要有的字數")
    parser.add_argument("--min-similarity", type=float, default=0.6, help="rationalize 模式：回覆和原台詞的相似度下限")
    parser.add_argument("--max-removed-ratio", type=float, default=0.3, help="rationalize 模式：清洗掉的比例上限")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="用 Ollama 或 OpenAI 相容伺服器產生推理（可續跑）")
    gen.add_argument("--mode", choices=("free", "rationalize"), default="free")
    gen.add_argument("--api", choices=API_CHOICES, default="ollama",
                     help="ollama：/api/chat；openai：/v1/chat/completions（llama-server -np、vLLM 才能真的多路並行）")
    gen.add_argument("--model", default=DEFAULT_MODEL, help="Ollama 的模型名稱；llama-server 只載一個模型，填任意名稱")
    gen.add_argument("--ollama", default="http://localhost:11434", help="推理伺服器的 URL")
    gen.add_argument("--api-key", default=os.environ.get("THINK_API_KEY", ""),
                     help="OpenAI 相容伺服器的 API key（vLLM --api-key）；預設讀環境變數 THINK_API_KEY")
    gen.add_argument("--samples", type=int, default=1000)
    gen.add_argument("--turn", choices=("last", "random"), default="last",
                     help="補哪一回合；free 模式選 random 時會把該回合之後的對話截掉")
    gen.add_argument("--seed", type=int, default=3407)
    gen.add_argument("--parallel", type=int, default=4)
    gen.add_argument("--num-ctx", type=int, default=16384)
    gen.add_argument("--num-predict", type=int, default=5120)
    gen.add_argument("--retry-on-length", type=int, default=2, help="被截斷或偵測到思考迴圈時重新抽樣的次數")
    gen.add_argument("--timeout", type=int, default=1800)
    gen.add_argument("--outage-wait", type=int, default=3600, help="伺服器失聯時最多等幾秒再放棄該筆")
    gen.add_argument("--max-outages", type=int, default=3, help="同一筆最多經歷幾次失聯等待")
    gen.add_argument("--report-every", type=int, default=10)
    gen.set_defaults(func=generate)

    bld = sub.add_parser("build", help="套用門檻，輸出 train.jsonl")
    bld.add_argument("--only-with-think", action="store_true", help="只輸出有推理的樣本")
    bld.set_defaults(func=build)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

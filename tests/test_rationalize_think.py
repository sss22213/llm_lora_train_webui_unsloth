import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rationalize_think", ROOT / "work" / "rationalize_think.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def test_merge_chat_stream_joins_thinking_and_content():
    lines = [
        json.dumps({"message": {"role": "assistant", "thinking": "考え", "content": ""}, "done": False}),
        json.dumps({"message": {"role": "assistant", "thinking": "る。", "content": ""}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": "「はい」"}, "done": False}).encode("utf-8"),
        "",
        json.dumps({"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop", "eval_count": 7, "eval_duration": 10}),
    ]
    merged = script.merge_chat_stream(lines)
    assert merged["message"] == {"role": "assistant", "thinking": "考える。", "content": "「はい」"}
    assert merged["done"] is True and merged["done_reason"] == "stop" and merged["eval_count"] == 7


def test_merge_chat_stream_without_done_is_flagged():
    merged = script.merge_chat_stream([json.dumps({"message": {"content": "x"}, "done": False})])
    assert merged["done"] is False


def test_free_mode_accepts_only_complete_records():
    args = type("Args", (), {"min_think_chars": 200, "max_think_chars": 6000, "min_content_chars": 10, "max_removed_ratio": 0.3, "min_similarity": 0.6})()
    good = {"mode": "free", "turn": 3, "done_reason": "stop", "thinking_chars": 500, "content_chars": 80}
    assert script.accepted(good, args)
    assert not script.accepted({**good, "thinking_chars": 9000}, args)
    assert not script.accepted({**good, "done_reason": "length"}, args)
    assert not script.accepted({**good, "thinking_chars": 50}, args)
    assert not script.accepted({**good, "error": "boom"}, args)


def test_free_request_keeps_system_prompt_untouched():
    messages = [{"role": "system", "content": "設定"}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    payload = script.build_request(messages, 2, "free", "m", 16384, 8192)
    assert payload["messages"] == messages[:2]
    assert payload["think"] is True and payload["options"]["num_predict"] == 8192
    rationalized = script.build_request(messages, 2, "rationalize", "m", 16384, 8192)
    assert "yo" in rationalized["messages"][0]["content"]


def test_loop_detection_stops_the_stream_early():
    block = "焔「見事なまでの執着だ。舌の動きが不規則になっている」（澪の顎を掴み、舌を強引に引き抜く）-> これだと曖昧。\n" * 3
    assert not script.looks_looped("普通の思考。" * 40)
    assert script.looks_looped(block * 4)
    looped = block * 6
    chunks = [json.dumps({"message": {"thinking": looped[i:i + 8]}, "done": False}) for i in range(0, len(looped), 8)]
    chunks.append(json.dumps({"message": {"content": "「はい」"}, "done": True, "done_reason": "stop"}))
    merged = script.merge_chat_stream(chunks)
    assert merged["done_reason"] == "loop"
    assert merged["message"]["content"] == ""      # 沒讀到最後那行，代表提早停了
    normal = script.merge_chat_stream(chunks, abort_on_loop=False)
    assert normal["done_reason"] == "stop"


def test_resume_redoes_records_that_failed_on_connection(tmp_path):
    path = tmp_path / "generated.jsonl"
    path.write_text(
        json.dumps({"idx": 1, "done_reason": "stop"}) + "\n"
        + json.dumps({"idx": 2, "error": "HTTPError: HTTP Error 524: <none>"}) + "\n"
        + json.dumps({"idx": 3, "done_reason": "length"}) + "\n"
        + "not json\n",
        encoding="utf-8",
    )
    assert script.load_done_indices(path) == {1, 3}
    assert script.load_done_indices(tmp_path / "missing.jsonl") == set()


def _sse(obj) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False)


def _messages():
    return [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]


def test_openai_payload_maps_ollama_options():
    payload = script.build_request(_messages(), 1, "free", "m", 16384, 5120)
    converted = script.to_openai_payload(payload)
    assert converted["messages"] == [{"role": "user", "content": "hi"}]
    assert converted["max_tokens"] == 5120 and converted["top_k"] == 20 and converted["stream"] is True
    assert converted["chat_template_kwargs"] == {"enable_thinking": True}
    assert "num_ctx" not in json.dumps(converted)  # llama-server 的 -c 是啟動參數


def test_merge_sse_stream_splits_reasoning_and_content():
    lines = [
        ": keep-alive",
        _sse({"choices": [{"delta": {"role": "assistant", "reasoning_content": "考え"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {"reasoning_content": "る。"}, "finish_reason": None}]}).encode("utf-8"),
        "",
        _sse({"choices": [{"delta": {"content": "「はい」"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": {"predicted_n": 7, "predicted_ms": 100.0}}),
        _sse({"choices": [], "usage": {"completion_tokens": 7, "prompt_tokens": 3}}),
        "data: [DONE]",
    ]
    merged = script.merge_sse_stream(lines)
    assert merged["message"] == {"role": "assistant", "thinking": "考える。", "content": "「はい」"}
    assert merged["done"] is True and merged["done_reason"] == "stop"
    assert merged["eval_count"] == 7 and merged["eval_duration"] == 100_000_000


def test_merge_sse_stream_reports_length_and_missing_end():
    cut = script.merge_sse_stream([_sse({"choices": [{"delta": {"content": "x"}, "finish_reason": "length"}]}), "data: [DONE]"])
    assert cut["done"] is True and cut["done_reason"] == "length"
    broken = script.merge_sse_stream([_sse({"choices": [{"delta": {"content": "x"}, "finish_reason": None}]})])
    assert broken["done"] is False


def test_merge_sse_stream_aborts_on_loop():
    chunk = "同じ下書きを繰り返す。" * 20
    lines = [_sse({"choices": [{"delta": {"reasoning_content": chunk}, "finish_reason": None}]}) for _ in range(64)]
    lines.append(_sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
    merged = script.merge_sse_stream(lines)
    assert merged["done"] is True and merged["done_reason"] == "loop"


def test_call_ollama_openai_posts_to_v1(monkeypatch):
    captured = {}

    class FakeResponse:
        def __init__(self, lines):
            self.lines = lines

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            return iter(self.lines)

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["accept"] = request.get_header("Accept")
        return FakeResponse([
            _sse({"choices": [{"delta": {"reasoning_content": "t"}, "finish_reason": None}]}),
            _sse({"choices": [{"delta": {"content": "c"}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ])

    monkeypatch.setattr(script.urllib.request, "urlopen", fake_urlopen)
    payload = script.build_request(_messages(), 1, "free", "m", 16384, 5120)
    merged = script.call_ollama("http://pod:11434/", payload, 30, api="openai")
    assert captured["url"] == "http://pod:11434/v1/chat/completions"
    assert captured["accept"] == "text/event-stream"
    assert captured["body"]["max_tokens"] == 5120 and captured["body"]["stream"] is True
    assert merged["message"] == {"role": "assistant", "thinking": "t", "content": "c"}
    assert merged["done_reason"] == "stop" and merged["eval_duration"] > 0  # 沒有 timings 就用牆鐘


def test_client_errors_are_recorded_without_outage_wait(tmp_path, monkeypatch):
    import io
    import types
    import urllib.error

    rows = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}]
    monkeypatch.setattr(script, "load_rows", lambda dataset, split: rows)
    monkeypatch.setattr(script.time, "sleep", lambda seconds: None)
    calls = []

    def rejecting(base_url, payload, timeout, api="ollama", api_key=None):
        calls.append(api)
        raise urllib.error.HTTPError(base_url, 400, "Bad Request", {}, io.BytesIO(b'{"error":"prompt too long"}'))

    monkeypatch.setattr(script, "call_ollama", rejecting)
    monkeypatch.setattr(script, "wait_for_server", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not wait")))
    args = types.SimpleNamespace(
        out=str(tmp_path), dataset="x", split="train", seed=1, samples=1, turn="last", mode="free", model="m",
        num_ctx=16384, num_predict=5120, ollama="http://pod:8000", api="openai", api_key="", timeout=30, max_outages=3,
        outage_wait=10, retry_on_length=2, parallel=1, report_every=1, min_think_chars=200, max_think_chars=6000,
        min_content_chars=10, max_removed_ratio=0.3, min_similarity=0.6,
    )
    script.generate(args)
    record = json.loads((tmp_path / "generated.jsonl").read_text(encoding="utf-8").strip())
    assert record["error"].startswith("HTTP 400") and "prompt too long" in record["error"]
    assert calls == ["openai"]  # 4xx 不重試
    assert script.load_done_indices(tmp_path / "generated.jsonl") == set()  # 續跑時仍會重做


def test_api_key_is_sent_as_bearer(monkeypatch):
    seen = {}

    class FakeResponse:
        def __init__(self, lines):
            self.lines = lines

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            return iter(self.lines)

        status = 200

    def fake_urlopen(request, timeout):
        seen[request.full_url] = request.get_header("Authorization")
        return FakeResponse([_sse({"choices": [{"delta": {"content": "c"}, "finish_reason": "stop"}]}), "data: [DONE]"])

    monkeypatch.setattr(script.urllib.request, "urlopen", fake_urlopen)
    payload = script.build_request(_messages(), 1, "free", "m", 16384, 5120)
    script.call_ollama("http://pod:8000", payload, 30, api="openai", api_key="secret")
    assert script.server_alive("http://pod:8000", api="openai", api_key="secret")
    assert seen == {"http://pod:8000/v1/chat/completions": "Bearer secret", "http://pod:8000/v1/models": "Bearer secret"}
    assert "Authorization" not in script.request_headers(None)

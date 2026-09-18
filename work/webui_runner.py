"""Configuration-driven Unsloth LoRA/QLoRA training runner used by the WebUI."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ("model", "dataset", "dataset_format", "output_directory")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"設定缺少必要欄位：{', '.join(missing)}")
    return config


def event(message: str) -> None:
    print(f"[LoRA Forge] {message}", flush=True)


def json_tools(value):
    if not value:
        return None
    return json.loads(value) if isinstance(value, str) else value


# Qwen3 系列的 chat template 對沒有 reasoning_content 的 assistant 回合會補上
# 「<think>\n\n</think>\n\n」；只訓練 assistant 時這段落在 loss 區間，等於反覆
# 教模型「不要思考」。有內容的 think 區塊（真正的推理）不會被動到。
EMPTY_THINK = re.compile(r"<think>\n\s*</think>\n\n")


def strip_empty_think_blocks(text: str) -> str:
    return EMPTY_THINK.sub("", text)


# 空 think 的三種處理：train（照常算 loss）、mask（留在上下文但標 -100，內容在
# 已關閉的 think 之後學，和推理時的位置一致）、strip（整段移除）。舊設定的
# strip_empty_think=True 視為 strip。
EMPTY_THINK_MODES = ("train", "mask", "strip")


def resolve_empty_think_mode(config: dict) -> str:
    mode = config.get("empty_think")
    if mode in EMPTY_THINK_MODES:
        return mode
    return "strip" if config.get("strip_empty_think") else "train"


def mask_empty_think_labels(input_ids, labels, marker, blank) -> list:
    """把緊接在 assistant marker 後面的空 think token 標成 -100，回傳新的 labels。"""
    seq = list(marker) + list(blank)
    n_marker, n_seq = len(marker), len(seq)
    input_ids = list(input_ids)
    labels = list(labels)
    j = 0
    while j <= len(input_ids) - n_seq:
        if input_ids[j] == seq[0] and input_ids[j : j + n_seq] == seq:
            for k in range(j + n_marker, j + n_seq):
                labels[k] = -100
            j += n_seq
        else:
            j += 1
    return labels


# Qwen3.5/3.8、Gemma-4 這類附視覺塔的底模，unsloth 回傳的是 Processor（內含
# .tokenizer）。TRL 原生的資料前處理拿 Processor 直接 tokenize 純文字時會多
# 一層 batch 維度（[[ids]]），train_on_responses_only 因而找不到 assistant
# marker、把整個資料集遮成 -100（unsloth 2026.9 + trl 0.24 實際踩到）。這裡
# 的訓練資料全是純文字，一律把內層 tokenizer 交給 SFTTrainer；儲存 adapter
# 仍用原本的物件，輸出內容不變。
def text_tokenizer_of(tokenizer):
    return getattr(tokenizer, "tokenizer", tokenizer)


def assert_flat_input_ids(dataset) -> None:
    """SFTTrainer 前處理後的 input_ids 必須是一維 token 序列，否則提早失敗。"""
    try:
        first = dataset[0]["input_ids"]
    except (KeyError, IndexError, TypeError):
        return
    if hasattr(first, "tolist"):
        first = first.tolist()
    if first and isinstance(first[0], (list, tuple)):
        raise RuntimeError(
            "SFTTrainer 產生的 input_ids 多了一層 batch 維度（[[ids]]），通常是把 "
            "Processor 而非 tokenizer 交給了 SFTTrainer；這會讓 "
            "train_on_responses_only 找不到 assistant marker"
        )


def main(config: dict) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # 減少 CUDA allocator 碎片：27B QLoRA + 8192 context 在 32GB 卡上很緊，
    # OOM 當下曾有 1.45 GiB「reserved but unallocated」。需在首次 CUDA 呼叫前設定。
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    # 官方映像檔把 UNSLOTH_VLLM_STANDBY 設成 1（GRPO/RL 讓 vLLM 常駐共用顯存用）。
    # unsloth_zoo 匯入時會因此把上面的 expandable_segments 從 PYTORCH_ALLOC_CONF
    # 移除（unsloth_zoo/__init__.py 的 UNSLOTH_VLLM_STANDBY == "1" 分支），結果就是
    # 樣本長度落差大時 allocator 逐步碎片化——曾在第 2153 步累積 5.68 GiB
    # 「reserved but unallocated」而 OOM。這裡是純 SFT，沒有用 fast_inference，
    # 且本映像檔的 vllm 已因 transformers 5.x 不可用，直接關掉這個模式。
    os.environ["UNSLOTH_VLLM_STANDBY"] = "0"
    # 固定 fused CE loss 每個 chunk 的記憶體上限。預設值是「首次呼叫時剩餘
    # VRAM 的一半（上限 4 GiB）」，而且被 functools.cache 快取住：等於用第一步
    # 記憶體還很寬裕時的快照，決定往後每一步的 chunk 大小。訓練後期顯存變緊時
    # 那個 chunk 就配置不出來（OOM 當下正是卡在 1.75 GiB 的 CE chunk）。
    os.environ.setdefault("UNSLOTH_CE_LOSS_TARGET_GB", "1.0")

    # 先修復編譯快取裡已知的壞產物（unsloth_zoo 對 force_accelerate_hooks
    # 的 getsource bug，詳見 patches/README.md），再 import unsloth。
    try:
        from patches.sanitize_unsloth_cache import sanitize_unsloth_cache

        for note in sanitize_unsloth_cache():
            event(f"Unsloth 快取修補：{note}")
    except Exception as exc:  # 修補只是防護，失敗不應擋下訓練
        event(f"Unsloth 快取修補略過：{exc}")

    # Unsloth 必須在 trl/transformers 之前 import：它會把兩者的類別換成
    # patched 版本。若先綁定原始 SFTConfig/SFTTrainer，SFTTrainer 會因
    # isinstance 檢查失敗改用 to_dict() 重建設定，而 to_dict 會把所有
    # *_token 欄位遮罩成 '<EOS_TOKEN>' 這類字串，訓練會在 eos 驗證時失敗。
    from unsloth import FastLanguageModel, FastModel
    from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only

    import torch
    from datasets import load_dataset
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(config["output_directory"])
    checkpoint_dir = output_dir / "checkpoints"
    adapter_dir = output_dir / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model_name = config["model"]
    model_family = config.get("model_family", "auto")
    if model_family == "auto":
        lowered = model_name.lower()
        model_family = "multimodal" if "gemma-4" in lowered or "-vl" in lowered else "language"
    model_class = FastModel if model_family == "multimodal" else FastLanguageModel

    def restore_decorated_forwards(modeling_module) -> None:
        # transformers 5.14 用 force_accelerate_hooks 包住部分 forward（如
        # Qwen3.5 linear_attn），且內層 wrapped 沒掛 functools.wraps。unsloth
        # 編譯器 getsource 會取到 wrapped 而非真正的 forward，生成引用不存在
        # 變數的 stub（NameError: name 'args' is not defined）。這裡從 closure
        # 取回原始 forward 換掉；編譯產物的類別原始碼仍保留裝飾器文字，
        # accelerate hook 行為不變。上游（unsloth-zoo compiler）修正後可移除。
        for symbol in vars(modeling_module).values():
            if not isinstance(symbol, type):
                continue
            forward = symbol.__dict__.get("forward")
            code = getattr(forward, "__code__", None)
            if code is None or forward.__name__ != "wrapped":
                continue
            if "forward_func" not in code.co_freevars or forward.__closure__ is None:
                continue
            cell = forward.__closure__[code.co_freevars.index("forward_func")]
            symbol.forward = cell.cell_contents

    max_seq_length = int(config["max_seq_length"])
    # 依架構（config.model_type）判斷，不能比對模型名稱：像
    # ThinkingCap-Qwen3.6-27B 名稱裡沒有 qwen3.5，底層卻是 qwen3_5 架構。
    try:
        from transformers import AutoConfig

        model_type = getattr(AutoConfig.from_pretrained(model_name), "model_type", "")
        restore_decorated_forwards(
            importlib.import_module(
                f"transformers.models.{model_type}.modeling_{model_type}"
            )
        )
    except Exception:  # trust_remote_code 等非內建架構沒有對應 modeling 模組
        pass
    event(f"載入模型 {model_name}（{model_family}, max_seq_length={max_seq_length}）")
    model, tokenizer = model_class.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=bool(config.get("load_in_4bit", True)),
    )
    text_tokenizer = text_tokenizer_of(tokenizer)
    if text_tokenizer is not tokenizer:
        event(
            f"底模附 {type(tokenizer).__name__}，訓練改用內層 "
            f"{type(text_tokenizer).__name__}（純文字資料）"
        )

    common_lora = {
        "r": int(config["lora_r"]),
        "lora_alpha": int(config["lora_alpha"]),
        "lora_dropout": float(config["lora_dropout"]),
        "bias": "none",
        "random_state": int(config["seed"]),
    }
    if model_family == "multimodal":
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            **common_lora,
        )
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            target_modules=config["target_modules"],
            use_gradient_checkpointing="unsloth",
            **common_lora,
        )

    dataset_args = [config["dataset"]]
    if config.get("dataset_config"):
        dataset_args.append(config["dataset_config"])
    event(f"載入資料集 {config['dataset']} / {config['dataset_split']}")
    try:
        dataset = load_dataset(*dataset_args, split=config["dataset_split"])
    except Exception as exc:
        dataset_config = config.get("dataset_config")
        if not dataset_config:
            # 多 config（子集）的資料集若不指定 config，datasets 會把所有
            # parquet 混在一起讀，不同 schema 的子集會造成 CastError。
            hint = "若此資料集有多個 config（子集），請在 WebUI 的「Config」欄位擇一填入後重試"
            try:
                from datasets import get_dataset_config_names

                names = get_dataset_config_names(config["dataset"])
                if len(names) > 1:
                    hint = (
                        "此資料集有多個 config（子集），"
                        "請在 WebUI 的「Config」欄位擇一填入後重試：\n"
                        + "、".join(names)
                    )
            except Exception:  # 列 config 只是輔助，失敗就用一般提示
                pass
            raise ValueError(f"載入資料集失敗：{exc}\n{hint}") from exc
        # 有些 repo 的 YAML configs 只列 config_name、沒寫 data_files（如
        # r0b0tlab 蒸餾集），此時任何 config 名稱都會退回「讀取全部檔案」，
        # 混到不同 schema 的子目錄就 CastError。改用 data_dir 直接鎖定
        # 子目錄重試。
        dataset = None
        for data_dir in (f"data/{dataset_config}", dataset_config):
            event(f"以 config 名稱載入失敗，改用 data_dir={data_dir} 重試")
            try:
                dataset = load_dataset(
                    config["dataset"], data_dir=data_dir, split=config["dataset_split"]
                )
                break
            except Exception:
                continue
        if dataset is None:
            raise

    source_filter = config.get("source_filter")
    if source_filter:
        if "first_source_dataset" not in dataset.column_names:
            raise ValueError("設定了來源過濾，但資料集沒有 first_source_dataset 欄位")
        event(f"過濾來源：{source_filter}")
        dataset = dataset.filter(
            lambda example: example["first_source_dataset"] == source_filter
        )

    max_samples = int(config.get("max_samples", 0))
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    if len(dataset) == 0:
        raise ValueError("資料集過濾後沒有任何樣本")
    event(f"使用 {len(dataset):,} 筆樣本")

    dataset_format = config["dataset_format"]
    prompt_completion = False
    response_markers: tuple[str, str] | None = None

    def strip_bos(text: str) -> str:
        if model_family == "multimodal":
            return text.removeprefix("<bos>")
        return text

    empty_think = resolve_empty_think_mode(config)
    if empty_think == "strip":
        event("移除空 think 區塊：assistant 回合直接接內容，不教模型跳過思考")
    elif empty_think == "mask":
        event("遮罩空 think 區塊：保留在上下文但不計 loss，內容在已關閉的 think 之後學")

    def render_postprocess(text: str) -> str:
        text = strip_bos(text)
        return strip_empty_think_blocks(text) if empty_think == "strip" else text

    if dataset_format == "sharegpt":
        dataset = standardize_sharegpt(dataset)

        # standardize_sharegpt 只轉換 from/value 型資料；已是 role/content 的
        # 資料集（欄位叫 messages）會原樣通過，不會產生 conversations 欄位。
        if "conversations" in dataset.column_names:
            conversation_field = "conversations"
        elif "messages" in dataset.column_names:
            event("資料集沒有 conversations 欄位，改用 messages 欄位（role/content 格式）")
            conversation_field = "messages"
        else:
            raise ValueError(
                "sharegpt 格式需要 conversations 或 messages 欄位；"
                f"資料集欄位為：{', '.join(dataset.column_names)}"
            )

        def render_sharegpt(example):
            return {
                "text": render_postprocess(
                    tokenizer.apply_chat_template(
                        example[conversation_field],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
            }

        dataset = dataset.map(render_sharegpt, remove_columns=dataset.column_names)
    elif dataset_format == "messages":
        if "messages" not in dataset.column_names:
            raise ValueError("messages 格式需要 messages 欄位")

        def render_messages(example):
            kwargs = {}
            if "tools" in example and example.get("tools"):
                kwargs["tools"] = json_tools(example["tools"])
            return {
                "text": render_postprocess(
                    tokenizer.apply_chat_template(
                        example["messages"],
                        tokenize=False,
                        add_generation_prompt=False,
                        **kwargs,
                    )
                )
            }

        dataset = dataset.map(render_messages, remove_columns=dataset.column_names)
    elif dataset_format == "messages_json":
        # r0b0tlab canonical trace：sft_* 子集用原生 messages/tools 欄位，
        # canonical/smoke 等子集把相同結構存成 JSON 字串（messages_json /
        # tools_json）；兩種實體格式都支援。
        if "messages" in dataset.column_names:
            messages_field, tools_field = "messages", "tools"
        elif "messages_json" in dataset.column_names:
            messages_field, tools_field = "messages_json", "tools_json"
        else:
            raise ValueError("messages_json 格式需要 messages 或 messages_json 欄位")
        has_tools = tools_field in dataset.column_names

        def normalize_canonical_tools(raw):
            tools = json_tools(raw)
            if not tools:
                return None
            normalized = []
            for tool in tools:
                function = dict(tool.get("function") or {})
                # 原生欄位把工具的 JSON schema 存成 parameters_json 字串，
                # chat template 預期的是展開後的 parameters dict。
                parameters_json = function.pop("parameters_json", None)
                if parameters_json:
                    function.setdefault("parameters", json.loads(parameters_json))
                normalized.append({**tool, "function": function})
            return normalized

        def render_canonical(example):
            raw = example[messages_field]
            # 缺值以 None／空字串／空列表表示（name、tool_call_id、
            # reasoning_content、tool_calls），留著會讓部分 chat template 印出
            # 空 think 區塊或把 None 印成 "None"。trainable 是資料集自用標記
            #（此資料集僅 assistant 為 True，語意與 assistant-only loss 相同）。
            messages = []
            for message in json.loads(raw) if isinstance(raw, str) else raw:
                cleaned = {
                    key: value
                    for key, value in message.items()
                    if value and key not in ("role", "content", "trainable")
                }
                cleaned["role"] = message["role"]
                cleaned["content"] = message.get("content") or ""
                # 資料裡 tool call 的 arguments 是 JSON 字串，Qwen3.6 template
                # 會對它 |items 展開，必須先解析成 dict。
                for tool_call in cleaned.get("tool_calls", ()):
                    function = tool_call.get("function") or {}
                    if isinstance(function.get("arguments"), str):
                        try:
                            function["arguments"] = json.loads(function["arguments"])
                        except json.JSONDecodeError:
                            pass
                messages.append(cleaned)
            kwargs = {}
            if has_tools:
                tools = normalize_canonical_tools(example[tools_field])
                if tools:
                    kwargs["tools"] = tools
            return {
                "text": render_postprocess(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        **kwargs,
                    )
                )
            }

        dataset = dataset.map(render_canonical, remove_columns=dataset.column_names)
    elif dataset_format == "text":
        text_field = config["text_field"]
        if text_field not in dataset.column_names:
            raise ValueError(f"找不到文字欄位：{text_field}")
        if text_field != "text":
            dataset = dataset.rename_column(text_field, "text")
        dataset = dataset.select_columns(["text"])
    elif dataset_format == "prompt_completion":
        if not {"prompt", "completion"}.issubset(dataset.column_names):
            raise ValueError("prompt_completion 格式需要 prompt 與 completion 欄位")
        keep = [name for name in ("prompt", "completion", "tools") if name in dataset.column_names]
        dataset = dataset.select_columns(keep)
        prompt_completion = True
    elif dataset_format == "fable_trace":
        if "row_json" not in dataset.column_names:
            raise ValueError("FABLE trace 格式需要 row_json 欄位")

        def render_fable(example):
            try:
                payload = json.loads(example["row_json"])
                messages = payload["messages"]
                if len(messages) < 2 or messages[-1].get("role") != "assistant":
                    return {"prompt": "", "completion": "", "valid": False}
                tools = json_tools(payload.get("tools"))
                kwargs = {"tools": tools} if tools else {}
                prompt = strip_bos(
                    tokenizer.apply_chat_template(
                        messages[:-1],
                        tokenize=False,
                        add_generation_prompt=True,
                        **kwargs,
                    )
                )
                full = strip_bos(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        **kwargs,
                    )
                )
                if not full.startswith(prompt):
                    return {"prompt": "", "completion": "", "valid": False}
                return {
                    "prompt": prompt,
                    "completion": full[len(prompt):],
                    "valid": True,
                }
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                return {"prompt": "", "completion": "", "valid": False}

        dataset = dataset.map(render_fable, remove_columns=dataset.column_names)
        dataset = dataset.filter(lambda example: example["valid"])
        dataset = dataset.remove_columns("valid")
        prompt_completion = True
        if len(dataset) == 0:
            raise ValueError("沒有可轉換成 prompt/completion 的 FABLE trace")
    else:
        raise ValueError(f"不支援的資料格式：{dataset_format}")

    if dataset_format == "fable_trace" and config.get("filter_overlength", True):
        before = len(dataset)

        def fits_context(example):
            token_ids = text_tokenizer(
                example["prompt"] + example["completion"],
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            return len(token_ids) <= max_seq_length

        event("移除超過 context window 的 trace，避免截斷最後訓練目標")
        dataset = dataset.filter(fits_context)
        event(f"長度過濾：保留 {len(dataset):,} / {before:,} 筆")
        if len(dataset) == 0:
            raise ValueError("所有 FABLE trace 都超過目前 max_seq_length")

    event(f"樣本預覽：\n{str(dataset[0])[:1600]}")

    training_kwargs = {
        "output_dir": str(checkpoint_dir),
        "max_length": max_seq_length,
        "per_device_train_batch_size": int(config["batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "warmup_steps": int(config["warmup_steps"]),
        "learning_rate": float(config["learning_rate"]),
        "logging_steps": int(config["logging_steps"]),
        "save_steps": int(config["save_steps"]),
        "save_total_limit": 2,
        "optim": config["optim"],
        "weight_decay": float(config["weight_decay"]),
        "lr_scheduler_type": config["lr_scheduler_type"],
        "seed": int(config["seed"]),
        "packing": bool(config.get("packing", False)),
        "report_to": "none",
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "completion_only_loss": True if prompt_completion else None,
    }
    if not prompt_completion:
        training_kwargs["dataset_text_field"] = "text"

    max_steps = int(config.get("max_steps", 0))
    if max_steps > 0:
        training_kwargs["max_steps"] = max_steps
    else:
        training_kwargs["num_train_epochs"] = float(config["num_train_epochs"])

    trainer = SFTTrainer(
        model=model,
        processing_class=text_tokenizer,
        train_dataset=dataset,
        args=SFTConfig(**training_kwargs),
    )
    assert_flat_input_ids(trainer.train_dataset)

    if config.get("assistant_only_loss", True) and not prompt_completion and dataset_format != "text":
        if model_family == "multimodal":
            response_markers = ("<start_of_turn>user\n", "<start_of_turn>model\n")
        else:
            response_markers = ("<|im_start|>user\n", "<|im_start|>assistant\n")
        event("只對 assistant 回應計算 loss")
        trainer = train_on_responses_only(
            trainer,
            instruction_part=response_markers[0],
            response_part=response_markers[1],
        )

    chat_formats = ("sharegpt", "messages", "messages_json")
    if empty_think == "mask" and dataset_format in chat_formats and model_family != "multimodal":
        marker = text_tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        blank = text_tokenizer("<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"]
        has_labels = "labels" in trainer.train_dataset.column_names

        def apply_mask(batch):
            ids_col = batch["input_ids"]
            labels_col = batch["labels"] if has_labels else ids_col
            return {
                "labels": [
                    mask_empty_think_labels(ids, labels, marker, blank)
                    for ids, labels in zip(ids_col, labels_col)
                ]
            }

        before = trainer.train_dataset[0]["labels"] if has_labels else trainer.train_dataset[0]["input_ids"]
        trainer.train_dataset = trainer.train_dataset.map(apply_mask, batched=True)
        after = trainer.train_dataset[0]["labels"]
        newly_masked = sum(1 for a, b in zip(before, after) if b == -100 and a != -100)
        event(f"樣本 0 遮罩了 {newly_masked // max(len(blank), 1)} 個空 think 區塊（不計 loss）")

    last_checkpoint = get_last_checkpoint(str(checkpoint_dir))
    if last_checkpoint:
        event(f"從 checkpoint 繼續：{last_checkpoint}")
    event("開始訓練")
    stats = trainer.train(resume_from_checkpoint=last_checkpoint)

    event("儲存 LoRA adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    # safetensors 經由 tempfile 寫出（權限 600）且容器內是 root，host 端
    # 一般使用者會讀不到；統一放開成全域可讀，方便其他專案取用 adapter。
    for path in adapter_dir.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    metrics = {
        **stats.metrics,
        "peak_vram_gb": (
            torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        ),
        "samples": len(dataset),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    event(
        f"完成：{adapter_dir}｜耗時 {metrics.get('train_runtime', 0):.0f} 秒｜"
        f"峰值 VRAM {metrics['peak_vram_gb']:.1f} GB"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    try:
        loaded = load_config(arguments.config)
        if arguments.validate_only:
            print(json.dumps(loaded, ensure_ascii=False, indent=2))
        else:
            main(loaded)
    except Exception as exc:
        event(f"失敗：{exc}")
        raise

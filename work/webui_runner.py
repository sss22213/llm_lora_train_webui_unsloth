"""Configuration-driven Unsloth LoRA/QLoRA training runner used by the WebUI."""

from __future__ import annotations

import argparse
import importlib
import json
import os
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


def main(config: dict) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # 減少 CUDA allocator 碎片：27B QLoRA + 8192 context 在 32GB 卡上很緊，
    # OOM 當下曾有 1.45 GiB「reserved but unallocated」。需在首次 CUDA 呼叫前設定。
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

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
    dataset = load_dataset(*dataset_args, split=config["dataset_split"])

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

    if dataset_format == "sharegpt":
        dataset = standardize_sharegpt(dataset)

        def render_sharegpt(example):
            return {
                "text": strip_bos(
                    tokenizer.apply_chat_template(
                        example["conversations"],
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
                "text": strip_bos(
                    tokenizer.apply_chat_template(
                        example["messages"],
                        tokenize=False,
                        add_generation_prompt=False,
                        **kwargs,
                    )
                )
            }

        dataset = dataset.map(render_messages, remove_columns=dataset.column_names)
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
        # 多模態模型的 tokenizer 是 Processor，直接呼叫時第一個位置參數是
        # images，會把文字當圖片解碼；計算長度一律用內部的純文字 tokenizer。
        text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

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
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(**training_kwargs),
    )

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

    last_checkpoint = get_last_checkpoint(str(checkpoint_dir))
    if last_checkpoint:
        event(f"從 checkpoint 繼續：{last_checkpoint}")
    event("開始訓練")
    stats = trainer.train(resume_from_checkpoint=last_checkpoint)

    event("儲存 LoRA adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
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

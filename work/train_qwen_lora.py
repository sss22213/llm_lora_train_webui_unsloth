"""Qwen3.5-9B QLoRA 冒煙測試(smoke test)。

用公開資料集 FineTome-100k 的前 3000 筆跑 100 步,
驗證整條訓練管線(下載 → 量化載入 → LoRA → 訓練 → 存檔)。
之後把 load_dataset 換成你自己的資料集即可。

執行:python /workspace/work/train_qwen_lora.py
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL = "unsloth/Qwen3.5-9B"
MAX_SEQ_LEN = 4096
OUTPUT_DIR = "/workspace/work/outputs/qwen3.5-9b-lora-smoke"

# ── 1. 載入模型(4-bit 量化)──
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit = True,
)

# ── 2. 掛上 LoRA adapter ──
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ── 3. 資料集:轉成 Qwen 的 chat template ──
dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:3000]")
dataset = standardize_sharegpt(dataset)


def to_text(examples):
    texts = tokenizer.apply_chat_template(
        examples["conversations"], tokenize = False, add_generation_prompt = False,
    )
    return {"text": texts}


dataset = dataset.map(to_text, batched = True)
print("樣本預覽:\n", dataset[0]["text"][:500])

# ── 4. 訓練 ──
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,   # 等效 batch size = 8
        warmup_steps = 10,
        max_steps = 100,                   # 冒煙測試;正式訓練改 num_train_epochs
        learning_rate = 2e-4,
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = OUTPUT_DIR,
        report_to = "none",
    ),
)

gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu.name}, 保留 VRAM: {torch.cuda.max_memory_reserved() / 1024**3:.1f} GB")

stats = trainer.train()
print(f"訓練完成,耗時 {stats.metrics['train_runtime']:.0f} 秒, "
      f"峰值 VRAM {torch.cuda.max_memory_reserved() / 1024**3:.1f} GB")

# ── 5. 存 LoRA adapter ──
model.save_pretrained(OUTPUT_DIR + "/adapter")
tokenizer.save_pretrained(OUTPUT_DIR + "/adapter")
print(f"Adapter 已存到 {OUTPUT_DIR}/adapter")

# 之後要匯出給 Ollama / llama.cpp 用的話:
# model.save_pretrained_gguf(OUTPUT_DIR + "/gguf", tokenizer, quantization_method="q4_k_m")

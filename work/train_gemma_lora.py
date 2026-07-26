"""Gemma 4 12B QLoRA 冒煙測試(smoke test)。

用公開資料集 FineTome-100k 的前 3000 筆跑 100 步,
驗證整條訓練管線(下載 → 4-bit 載入 → LoRA → 訓練 → 存檔)。
之後把 load_dataset 換成你自己的資料集即可。

執行:python /workspace/work/train_gemma_lora.py
"""
from unsloth import FastModel
from unsloth.chat_templates import standardize_sharegpt
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL = "unsloth/gemma-4-12b-it"
MAX_SEQ_LEN = 2048
OUTPUT_DIR = "/workspace/work/outputs/gemma4-12b-lora-smoke"

# ── 1. 載入模型(4-bit 量化)──
# Gemma 4 是多模態模型,用 FastModel(不是 FastLanguageModel)
model, tokenizer = FastModel.from_pretrained(
    model_name = MODEL,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit = True,
)

# ── 2. 掛上 LoRA adapter(只訓練語言層,不動視覺層)──
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers = False,
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

# ── 3. 資料集:轉成 Gemma 的 chat template ──
dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:3000]")
dataset = standardize_sharegpt(dataset)


def to_text(examples):
    texts = [
        # Gemma 的 template 會自帶 <bos>,SFTTrainer 分詞時又會加一次,
        # 所以這裡先拿掉,避免雙重 BOS 影響訓練品質
        tokenizer.apply_chat_template(
            convo, tokenize = False, add_generation_prompt = False,
        ).removeprefix("<bos>")
        for convo in examples["conversations"]
    ]
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

stats = trainer.train()
print(f"訓練完成,耗時 {stats.metrics['train_runtime']:.0f} 秒, "
      f"峰值 VRAM {torch.cuda.max_memory_reserved() / 1024**3:.1f} GB")

# ── 5. 存 LoRA adapter ──
model.save_pretrained(OUTPUT_DIR + "/adapter")
tokenizer.save_pretrained(OUTPUT_DIR + "/adapter")
print(f"Adapter 已存到 {OUTPUT_DIR}/adapter")

# 之後要匯出 GGUF 給 Ollama / llama.cpp 用的話:
# model.save_pretrained_gguf(OUTPUT_DIR + "/gguf", tokenizer, quantization_method="q4_k_m")

"""Qwen3.5-9B QLoRA — 工具呼叫(tool calling)資料集微調練習。

從 FineTome 冒煙測試升級的下一步:改用 agent/tool-call 型資料集
NousResearch/hermes-function-calling-v1(Apache-2.0)。比起前一個腳本,
多了兩個新概念:

1. 手動格式轉換:這類資料的 conversations 多了 tool 角色(工具執行結果),
   standardize_sharegpt 不處理,要自己映射成 chat template 訊息。
2. train_on_responses_only:只對 assistant 段落計算 loss。
   user 提問與工具回傳只當上下文,不讓模型學著「生成」它們——
   agent 資料裡工具輸出佔大量篇幅,這一步對訓練品質影響很大。

執行:python /workspace/work/train_qwen_toolcall_lora.py
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL = "unsloth/Qwen3.5-9B"
MAX_SEQ_LEN = 8192                 # 工具呼叫對話比一般聊天長,序列長度拉高
OUTPUT_DIR = "/workspace/work/outputs/qwen3.5-9b-lora-toolcall"

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

# ── 3. 資料集:hermes-function-calling-v1 的 func_calling 子集(多輪 agentic)──
# 欄位是 sharegpt 風格的 conversations([{from, value}, ...]),
# 角色除了 system/human/gpt,多了 tool(工具執行結果)。
# system 訊息裡已內含可用工具的 JSON 簽名,對話是自成一體的。
dataset = load_dataset("NousResearch/hermes-function-calling-v1",
                       "func_calling", split = "train")

# tool 以 user 角色餵回模型(Hermes 慣例:內容包在 <tool_response> 標籤裡,
# 資料大多已自帶標籤,沒有的補上),避免 template 不支援 tool 角色的問題
ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant", "tool": "user"}


def to_text(example):
    messages = []
    for turn in example["conversations"]:
        text = turn["value"]
        if turn["from"] == "tool" and "<tool_response>" not in text:
            text = f"<tool_response>\n{text}\n</tool_response>"
        messages.append({"role": ROLE_MAP[turn["from"]], "content": text})
    return {"text": tokenizer.apply_chat_template(
        messages, tokenize = False, add_generation_prompt = False)}


dataset = dataset.map(to_text, remove_columns = dataset.column_names)
print("樣本預覽:\n", dataset[0]["text"][:800])

# ── 4. 訓練 ──
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,   # 序列變長,batch 降為 1
        gradient_accumulation_steps = 8,   # 等效 batch size = 8
        warmup_steps = 10,
        max_steps = 100,                   # 冒煙測試;正式訓練改 num_train_epochs = 2
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

# 只對 assistant 段落計 loss。這兩個標記字串必須跟上面「樣本預覽」
# 印出的實際 template 一致(Qwen 是 ChatML),不一致會 mask 錯位置。
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

# 驗證 masking:labels 中 -100 = 不計 loss。只解碼保留下來的 tokens,
# 印出來應該全是 assistant 的回應(含 <tool_call> JSON),沒有 user/工具內容
kept = [t for t in trainer.train_dataset[0]["labels"] if t != -100]
print("計入 loss 的部分(前 500 字):\n", tokenizer.decode(kept)[:500])

stats = trainer.train()
print(f"訓練完成,耗時 {stats.metrics['train_runtime']:.0f} 秒, "
      f"峰值 VRAM {torch.cuda.max_memory_reserved() / 1024**3:.1f} GB")

# ── 5. 存 LoRA adapter ──
model.save_pretrained(OUTPUT_DIR + "/adapter")
tokenizer.save_pretrained(OUTPUT_DIR + "/adapter")
print(f"Adapter 已存到 {OUTPUT_DIR}/adapter")

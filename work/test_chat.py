"""載入訓練好的 LoRA adapter,在終端機裡跟模型對話,測試訓練效果。

執行:sudo docker compose exec unsloth python /workspace/work/test_chat.py
離開:輸入 exit
"""
from unsloth import FastModel
from transformers import TextStreamer

ADAPTER = "/workspace/work/outputs/gemma4-12b-lora-smoke/adapter"

model, tokenizer = FastModel.from_pretrained(
    model_name = ADAPTER,       # 直接指向 adapter 資料夾,會自動載入底模 + LoRA
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastModel.for_inference(model)  # 切換到推理模式(比較快)

print("模型載入完成,開始對話(輸入 exit 離開)\n")
while True:
    try:
        q = input("你: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if q.lower() in ("exit", "quit", ""):
        break
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")
    print("Gemma: ", end = "")
    model.generate(
        input_ids = inputs,
        max_new_tokens = 512,
        temperature = 0.7,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )
    print()

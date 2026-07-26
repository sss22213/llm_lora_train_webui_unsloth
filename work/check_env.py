"""容器環境健檢:確認 GPU、CUDA、Unsloth 都正常。

用法:docker compose exec unsloth python /workspace/work/check_env.py
"""
import torch

print(f"PyTorch        : {torch.__version__}")
print(f"CUDA available : {torch.cuda.is_available()}")
print(f"GPU            : {torch.cuda.get_device_name(0)}")
cap = torch.cuda.get_device_capability(0)
print(f"Compute cap    : sm_{cap[0]}{cap[1]}  (RTX 5090 應為 sm_120)")
print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"BF16 support   : {torch.cuda.is_bf16_supported()}")

import unsloth
print(f"Unsloth        : {unsloth.__version__}")
print(f"Unsloth path   : {unsloth.__file__}  (應指向 /opt/unsloth-src)")

import bitsandbytes
import triton
print(f"bitsandbytes   : {bitsandbytes.__version__}")
print(f"triton         : {triton.__version__}")

print("\n環境正常,可以開始訓練。")

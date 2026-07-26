# Unsloth 編譯快取修補

## 問題

unsloth_zoo（<= 2026.7.2）的原始碼抽取器看不懂 transformers 5.14 新增的
`@force_accelerate_hooks` 裝飾器（內層 `wrapped` 沒掛 `functools.wraps`）。
對有掛這個裝飾器的 forward（目前是 Qwen3.5 架構的線性注意力層
`Qwen3_5GatedDeltaNet.forward`），`inspect.getsource` 會抓到裝飾器的 wrapper
而不是真正的 forward，生成的快取模組因此含有壞掉的 stub：

```python
def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    return wrapped(self, *args, **kwargs)   # NameError: name 'args' is not defined
```

訓練會在第一個 step 直接掛掉（2026-07-25 用 `bottlecapai/ThinkingCap-Qwen3.6-27B`
訓練時踩到——模型名稱沒有 qwen3.5 字樣，但底層是 `qwen3_5` 架構）。

## 防護機制（自動，不需手動介入）

`webui_runner.py` 每次啟動訓練時有兩層防護：

1. **`sanitize_unsloth_cache.py`**（本資料夾）：在 import unsloth 之前掃描
   `unsloth_compiled_cache/unsloth_compiled_module_*.py`，找到壞 stub 就地修復
   （改成委派呼叫 transformers 原版 forward）；無法文字修復時刪檔讓 unsloth
   重新生成。冪等，健康檔案不動。
2. **`restore_decorated_forwards()`**（`webui_runner.py`）：模型載入前依
   `AutoConfig.model_type` 找到對應的 modeling 模組，把
   `force_accelerate_hooks` 從 closure 拆回原始 forward。這使得 unsloth
   **重新生成**快取（例如升級 unsloth / unsloth_zoo / transformers 之後）
   時 getsource 能抓到正確原始碼，生成的程式碼直接就是好的。

兩層合起來涵蓋：既有壞快取（第 1 層修）、升級後重新生成（第 2 層防）。

## 檔案

- `sanitize_unsloth_cache.py`：自動修補腳本，可獨立執行：
  `python sanitize_unsloth_cache.py [快取目錄]`（預設吃 `UNSLOTH_COMPILE_LOCATION`
  環境變數，否則用目前目錄下的 `unsloth_compiled_cache`）。
- `unsloth_compiled_module_qwen3_5_forward.patch`：2026-07-25 手動修補的參考
  diff（unsloth 2026.7.3 / unsloth_zoo 2026.7.2 / transformers 5.14.1 生成的
  檔案）。只是文件紀錄；實際修補一律走 `sanitize_unsloth_cache.py`，因為
  重新生成的檔案行號會變，這份 diff 不一定套得上。

## 何時可以移除

上游 unsloth_zoo 修正 `force_accelerate_hooks` 的處理後，這兩層防護都會
自然變成 no-op，可以整個移除（`webui_runner.py` 裡的 sanitize 呼叫、
`restore_decorated_forwards` 與本資料夾）。升級後看訓練 log：若再也沒有
「Unsloth 快取修補」訊息即表示不再需要。

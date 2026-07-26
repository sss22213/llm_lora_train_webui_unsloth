"""偵測並修復 Unsloth 編譯快取中壞掉的 forward stub。

unsloth_zoo <= 2026.7.2 的原始碼抽取器看不懂 transformers 5.14 新增的
@force_accelerate_hooks 裝飾器（內層 wrapped 沒掛 functools.wraps），
getsource 會抓到裝飾器的 wrapper 而不是真正的 forward，生成這種 stub：

    @force_accelerate_hooks("conv1d")
    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        return wrapped(self, *args, **kwargs)   # NameError: name 'args' is not defined

修復方式：在生成的類別定義前捕捉 transformers 原版類別的 forward（模組
開頭已 import、尚未被類別定義遮蔽），把 stub 改成委派呼叫。原版 forward
本身已帶正確的 force_accelerate_hooks 包裝，因此 stub 上的裝飾器一併移除。

無法文字修復時退而求其次：刪掉整個快取檔讓 unsloth 重新生成——前提是
webui_runner.py 已在編譯前先 restore_decorated_forwards()，重生成的
程式碼就會是正確的。

冪等：已修復或沒有問題的檔案不會被動到。上游修正後此腳本自然變成 no-op。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# 壞 stub 的本體：簽名沒宣告 *args 卻引用它
_RETURN_WRAPPED = re.compile(r"^([ \t]+)return wrapped\(self, \*args, \*\*kwargs\)\s*$")
_DEF_FORWARD = re.compile(r"^[ \t]+def forward\(")
_DECORATOR = re.compile(r"^[ \t]+@")
_CLASS_DEF = re.compile(r"^class (\w+)\s*[(:]")
# 簽名整體：def forward(self, <params>):（可能跨多行）
_SIGNATURE = re.compile(r"^([ \t]+)def forward\(\s*self\s*,?(.*)\)\s*:\s*$", re.DOTALL)


def _split_params(params: str) -> list[str]:
    """依頂層逗號切開參數（annotation 裡的逗號、括號不算）。"""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in params:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def _build_call_args(params: list[str]) -> list[str] | None:
    """把簽名參數轉成委派呼叫的引數；回傳 None 表示簽名太特殊，放棄修復。"""
    call: list[str] = []
    for param in params:
        if param == "*":
            continue
        if param == "/":  # positional-only 無法用 name=name 轉呼叫
            return None
        stars = len(param) - len(param.lstrip("*"))
        name_match = re.match(r"\w+", param.lstrip("*"))
        if name_match is None:
            return None
        name = name_match.group(0)
        if stars == 2:
            call.append(f"**{name}")
        else:
            call.append(f"{name}={name}")
    return call


def _repair_source(source: str) -> tuple[str, str | None]:
    """嘗試修復。回傳 (狀態, 新原始碼)：

    - ("repaired", new_source)：找到壞 stub 且已修復
    - ("clean", None)：return wrapped 的簽名其實有 *args，程式碼是健康的
    - ("unfixable", None)：找到壞 stub 但無法安全修復，呼叫端應刪檔重生成
    """
    lines = source.splitlines(keepends=True)
    # (line_start, line_end_exclusive, replacement_text)，行區間替換
    edits: list[tuple[int, int, str]] = []
    captured_classes: set[str] = set()
    found_broken = False

    for ri, line in enumerate(lines):
        return_match = _RETURN_WRAPPED.match(line)
        if return_match is None:
            continue

        # 往上找最近的 def forward( 行，中間的行都是簽名的一部分
        di = ri - 1
        while di >= 0 and not _DEF_FORWARD.match(lines[di]):
            di -= 1
        if di < 0:
            continue  # 模組層級的 wrapped 定義本身，不是 stub
        signature = "".join(lines[di:ri])
        sig_match = _SIGNATURE.match(signature)
        if sig_match is None:
            return ("unfixable", None)
        indent, params_text = sig_match.group(1), sig_match.group(2)

        params = _split_params(params_text)
        has_star_args = any(
            p.startswith("*") and not p.startswith("**") and p != "*" for p in params
        )
        if has_star_args:
            continue  # 簽名真的有 *args，這段程式碼沒壞
        found_broken = True

        call_args = _build_call_args(params)
        if call_args is None:
            return ("unfixable", None)

        # 往上收集緊貼的裝飾器行，並找出包住這個 forward 的 class
        deco_start = di
        while deco_start > 0 and _DECORATOR.match(lines[deco_start - 1]):
            deco_start -= 1
        class_line = None
        class_name = None
        for ci in range(deco_start - 1, -1, -1):
            class_match = _CLASS_DEF.match(lines[ci])
            if class_match:
                class_line, class_name = ci, class_match.group(1)
                break
        if class_name is None:
            return ("unfixable", None)

        # 原版類別必須在 class 定義之前就已 import，捕捉行才拿得到它。
        # 生成檔裡 class 定義是該名稱第一次「定義」，更早的出現必然是 import。
        preamble = "".join(lines[:class_line])
        if not re.search(rf"\b{class_name}\b", preamble):
            return ("unfixable", None)

        original_ref = f"_unsloth_patch_original_{class_name}_forward"
        if class_name not in captured_classes:
            captured_classes.add(class_name)
            capture = (
                f"# {class_name} 在此仍是 transformers 原版類別（下方類別定義才會遮蔽它），\n"
                f"# 其 forward 已帶正確的 force_accelerate_hooks 包裝，直接委派即可。\n"
                f"{original_ref} = {class_name}.forward\n\n"
            )
            edits.append((class_line, class_line, capture))

        # 拿掉壞 stub 上的 force_accelerate_hooks 裝飾器（原版 forward 已包過）
        kept_decorators = "".join(
            deco for deco in lines[deco_start:di] if "force_accelerate_hooks" not in deco
        )
        body_indent = return_match.group(1)
        body = (
            f"{body_indent}return {original_ref}(\n"
            f"{body_indent}    self,\n"
            + "".join(f"{body_indent}    {arg},\n" for arg in call_args)
            + f"{body_indent})\n"
        )
        edits.append((deco_start, ri + 1, kept_decorators + signature + body))

    if not found_broken:
        return ("clean", None)

    for start, end, text in sorted(edits, key=lambda edit: edit[0], reverse=True):
        lines[start:end] = [text]
    return ("repaired", "".join(lines))


def _drop_pycache(py_file: Path) -> None:
    pycache = py_file.parent / "__pycache__"
    if pycache.is_dir():
        for stale in pycache.glob(f"{py_file.stem}.*.pyc"):
            stale.unlink(missing_ok=True)


def sanitize_unsloth_cache(cache_dir: Path | None = None) -> list[str]:
    """修復（或刪除）壞掉的快取檔，回傳處理紀錄；沒事就回傳空 list。"""
    if cache_dir is None:
        cache_dir = Path(os.environ.get("UNSLOTH_COMPILE_LOCATION", "unsloth_compiled_cache"))
    actions: list[str] = []
    if not cache_dir.is_dir():
        return actions

    for py_file in sorted(cache_dir.glob("unsloth_compiled_module_*.py")):
        source = py_file.read_text(encoding="utf-8")
        if "return wrapped(self, *args" not in source:
            continue
        status, repaired = _repair_source(source)
        if status == "repaired":
            py_file.write_text(repaired, encoding="utf-8")
            _drop_pycache(py_file)
            actions.append(f"{py_file.name}：已就地修復壞掉的 forward stub")
        elif status == "unfixable":
            py_file.unlink()
            _drop_pycache(py_file)
            actions.append(f"{py_file.name}：無法文字修復，已刪除待重新生成")
    return actions


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    results = sanitize_unsloth_cache(target)
    for line in results:
        print(line)
    if not results:
        print("快取沒有需要修復的檔案")

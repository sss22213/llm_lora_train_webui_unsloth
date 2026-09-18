import importlib.util
from pathlib import Path

from webui.main import TrainingRequest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("webui_runner", ROOT / "work" / "webui_runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

EMPTY = "<|im_start|>assistant\n<think>\n\n</think>\n\n「こんにちは」<|im_end|>\n"
REAL = "<|im_start|>assistant\n<think>\n考える。\n</think>\n\n答え<|im_end|>\n"


def test_empty_think_blocks_are_removed_but_real_reasoning_is_kept():
    text = "<|im_start|>user\nhi<|im_end|>\n" + EMPTY + EMPTY + REAL
    out = runner.strip_empty_think_blocks(text)
    assert out.count("<think>") == 1
    assert "<|im_start|>assistant\n「こんにちは」<|im_end|>" in out
    assert REAL in out


def test_request_defaults_to_training_empty_think_as_is():
    assert TrainingRequest(model="m", output_name="o").empty_think == "train"
    assert TrainingRequest(model="m", output_name="o", empty_think="mask").empty_think == "mask"


def test_mode_resolution_keeps_old_strip_flag_working():
    assert runner.resolve_empty_think_mode({}) == "train"
    assert runner.resolve_empty_think_mode({"strip_empty_think": True}) == "strip"
    assert runner.resolve_empty_think_mode({"empty_think": "mask", "strip_empty_think": True}) == "mask"
    assert runner.resolve_empty_think_mode({"empty_think": "bogus"}) == "train"


def test_mask_only_touches_blank_think_after_assistant_marker():
    marker, blank = [1, 2, 3], [4, 5, 6, 5]
    #        user.. | marker  | blank      | content | marker  | real think.. | content
    ids =    [9, 9,   1, 2, 3,  4, 5, 6, 5,  7, 8,     1, 2, 3,  4, 5, 10, 6, 5, 7]
    labels = [-100, -100, -100, -100, -100, 4, 5, 6, 5, 7, 8, -100, -100, -100, 4, 5, 10, 6, 5, 7]
    out = runner.mask_empty_think_labels(ids, labels, marker, blank)
    assert out[5:9] == [-100] * 4          # 空 think 不計 loss
    assert out[9:11] == [7, 8]             # 內容照舊
    assert out[14:20] == [4, 5, 10, 6, 5, 7]  # 真正的推理不受影響
    assert labels[5:9] == [4, 5, 6, 5]     # 不改動輸入


def test_mask_handles_blank_at_the_very_end():
    ids = [1, 2, 3, 4, 5, 6, 5]
    out = runner.mask_empty_think_labels(ids, ids, [1, 2, 3], [4, 5, 6, 5])
    assert out == [1, 2, 3, -100, -100, -100, -100]

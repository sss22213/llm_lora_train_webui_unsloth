import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("webui_runner", ROOT / "work" / "webui_runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FakeTokenizer:
    pass


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()


def test_processor_is_unwrapped_to_inner_tokenizer():
    processor = FakeProcessor()
    assert runner.text_tokenizer_of(processor) is processor.tokenizer


def test_plain_tokenizer_is_returned_unchanged():
    tokenizer = FakeTokenizer()
    assert runner.text_tokenizer_of(tokenizer) is tokenizer


def test_flat_input_ids_pass():
    runner.assert_flat_input_ids([{"input_ids": [1, 2, 3]}])


def test_nested_input_ids_fail_fast():
    with pytest.raises(RuntimeError, match="batch"):
        runner.assert_flat_input_ids([{"input_ids": [[1, 2, 3]]}])


def test_datasets_without_input_ids_are_ignored():
    runner.assert_flat_input_ids([{"text": "x"}])
    runner.assert_flat_input_ids([])

import json
import time
import types

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import webui.main as webui_main
from webui.main import (
    ThinkBuildRequest,
    ThinkGenerateRequest,
    ThinkJobManager,
    app,
    parse_progress,
    scan_think_datasets,
    think_build_command,
    think_generate_command,
    is_local_ollama,
)


client = TestClient(app)


def test_generate_request_defaults_and_command(tmp_path):
    request = ThinkGenerateRequest(ollama_url="http://ollama:11434/")
    assert request.mode == "free" and request.turn == "last"
    assert request.ollama_url == "http://ollama:11434"
    command = think_generate_command(request, tmp_path)
    assert command.index("--out") < command.index("generate") < command.index("--mode")
    assert command[command.index("--mode") + 1] == "free"
    assert command[command.index("--ollama") + 1] == "http://ollama:11434"


def test_generate_request_rejects_non_http_url():
    with pytest.raises(ValidationError):
        ThinkGenerateRequest(ollama_url="ollama:11434")


def test_build_command_flags(tmp_path):
    request = ThinkBuildRequest(output_name="demo", only_with_think=True, min_think_chars=50)
    command = think_build_command(request, tmp_path, "org/data")
    assert command[-2:] == ["build", "--only-with-think"]
    assert command[command.index("--min-think-chars") + 1] == "50"
    assert command[command.index("--dataset") + 1] == "org/data"


def test_parse_progress_takes_last_line():
    text = "資料集 100 筆\n[10/200] 通過 9 拒絕 1\n[20/200] 通過 18 拒絕 2 錯誤 0\n"
    assert parse_progress(text) == {"done": 20, "total": 200, "text": "[20/200] 通過 18 拒絕 2 錯誤 0"}
    assert parse_progress("nothing here") is None


def test_think_endpoints_are_available():
    assert client.get("/api/think/jobs").status_code == 200
    assert isinstance(client.get("/api/think/datasets").json(), list)
    missing = client.post("/api/think/build", json={"output_name": "no-such-dataset-xyz"})
    assert missing.status_code == 404
    bad_url = client.get("/api/think/ollama", params={"url": "ollama:11434"})
    assert bad_url.status_code == 400


def test_scan_think_datasets_reads_meta_and_counts(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "datasets" / "demo"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "generated.jsonl").write_text('{"idx": 1}\n{"idx": 2}\n', encoding="utf-8")
    (dataset_dir / "meta.json").write_text(json.dumps({"dataset": "org/data", "mode": "free"}), encoding="utf-8")
    monkeypatch.setattr(webui_main, "DATASETS_DIR", tmp_path / "datasets")

    items = scan_think_datasets()
    assert len(items) == 1
    assert items[0]["name"] == "demo"
    assert items[0]["generated_rows"] == 2
    assert items[0]["train_rows"] is None
    assert items[0]["meta"]["dataset"] == "org/data"


def test_think_manager_runs_script_and_reports_progress(tmp_path, monkeypatch):
    script = tmp_path / "fake_think.py"
    script.write_text(
        """\
import json
import sys
from pathlib import Path

out = Path(sys.argv[sys.argv.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
with (out / "generated.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"idx": 1}) + "\\n")
    handle.write(json.dumps({"idx": 2}) + "\\n")
print("[1/2] 通過 1 拒絕 0 錯誤 0", flush=True)
print("[2/2] 通過 2 拒絕 0 錯誤 0", flush=True)
""",
        encoding="utf-8",
    )
    think_jobs = tmp_path / "think_jobs"
    think_jobs.mkdir()
    monkeypatch.setattr(webui_main, "THINK_SCRIPT", script)
    monkeypatch.setattr(webui_main, "THINK_JOBS_DIR", think_jobs)
    monkeypatch.setattr(webui_main, "WORK_DIR", tmp_path)
    monkeypatch.setattr(webui_main, "manager", types.SimpleNamespace(list=lambda: []))

    local = ThinkJobManager()
    request = ThinkGenerateRequest(output_name="demo", samples=2)
    out_dir = tmp_path / "datasets" / "demo"
    out_dir.mkdir(parents=True)
    job = local.create("generate", request.model_dump(), think_generate_command(request, out_dir), out_dir)

    deadline = time.monotonic() + 5
    while local.get(job.id).status in ("queued", "running") and time.monotonic() < deadline:
        time.sleep(0.02)

    finished = local.serialize(local.get(job.id))
    assert finished["status"] == "completed"
    assert finished["progress"] == {"done": 2, "total": 2, "text": "[2/2] 通過 2 拒絕 0 錯誤 0"}
    assert (out_dir / "generated.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert local.active() is None


def test_local_ollama_detection():
    assert is_local_ollama("http://host.docker.internal:11434")
    assert is_local_ollama("http://localhost:11434")
    assert is_local_ollama("http://192.168.50.192:11434")
    assert not is_local_ollama("https://abc123-11434.proxy.runpod.net")
    assert not is_local_ollama("http://1.1.1.1:11434")


def test_build_can_run_while_remote_generate_is_active(tmp_path, monkeypatch):
    script = tmp_path / "slow_think.py"
    script.write_text(
        """\
import sys, time
from pathlib import Path

out = Path(sys.argv[sys.argv.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
if "build" in sys.argv:
    (out / "train.jsonl").write_text("{}\\n", encoding="utf-8")
else:
    (out / "generated.jsonl").write_text('{"idx": 1}\\n', encoding="utf-8")
    time.sleep(1.5)
""",
        encoding="utf-8",
    )
    think_jobs = tmp_path / "think_jobs"
    think_jobs.mkdir()
    monkeypatch.setattr(webui_main, "THINK_SCRIPT", script)
    monkeypatch.setattr(webui_main, "THINK_JOBS_DIR", think_jobs)
    monkeypatch.setattr(webui_main, "WORK_DIR", tmp_path)
    monkeypatch.setattr(webui_main, "manager", types.SimpleNamespace(list=lambda: []))

    local = ThinkJobManager()
    out_dir = tmp_path / "datasets" / "demo"
    out_dir.mkdir(parents=True)
    gen_request = ThinkGenerateRequest(
        output_name="demo", samples=1, ollama_url="https://abc123-11434.proxy.runpod.net"
    )
    gen = local.create("generate", gen_request.model_dump(), think_generate_command(gen_request, out_dir), out_dir)
    assert local.active().id == gen.id
    assert not local.generation_active()  # 遠端 Ollama，本機 GPU 沒被占用

    # 生成還在跑時可以組資料集
    build_request = ThinkBuildRequest(output_name="demo")
    build = local.create("build", build_request.model_dump(), think_build_command(build_request, out_dir, "x/y"), out_dir)
    assert build.kind == "build"
    assert local.active().id == gen.id  # 仍以生成任務為主

    # 但同類任務一次只能一個
    with pytest.raises(RuntimeError, match="生成任務在執行中"):
        local.create("generate", gen_request.model_dump(), think_generate_command(gen_request, out_dir), out_dir)

    deadline = time.monotonic() + 6
    while local.running() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert local.get(gen.id).status == "completed"
    assert local.get(build.id).status == "completed"
    assert local.active() is None


def test_generate_command_passes_backend(tmp_path):
    assert ThinkGenerateRequest().api == "ollama"
    request = ThinkGenerateRequest(api="openai", ollama_url="https://pod-11434.proxy.runpod.net")
    command = think_generate_command(request, tmp_path)
    assert command[command.index("--api") + 1] == "openai"


def test_probe_lists_models_from_openai_compatible_server(monkeypatch):
    import json

    class FakeResponse:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(self.body).encode("utf-8")

    seen = []

    def fake_urlopen(request, timeout):
        seen.append(request.full_url)
        if request.full_url.endswith("/v1/models"):
            return FakeResponse({"object": "list", "data": [{"id": "qwen3.8-27b-heretic", "object": "model"}]})
        return FakeResponse({"models": [{"name": "qwen3.8:latest", "size": 1}]})

    monkeypatch.setattr(webui_main.urllib.request, "urlopen", fake_urlopen)
    assert webui_main.ollama_models("http://pod:11434/", "openai") == [{"name": "qwen3.8-27b-heretic", "size": None}]
    assert webui_main.ollama_models("http://pod:11434", "ollama") == [{"name": "qwen3.8:latest", "size": 1}]
    assert seen == ["http://pod:11434/v1/models", "http://pod:11434/api/tags"]
    probe = client.get("/api/think/ollama", params={"url": "http://pod:11434", "api": "openai"})
    assert probe.status_code == 200
    assert probe.json()["api"] == "openai" and probe.json()["models"][0]["name"] == "qwen3.8-27b-heretic"


def test_api_key_goes_through_env_not_argv(tmp_path, monkeypatch):
    request = ThinkGenerateRequest(api="openai", api_key=" secret ")
    assert request.api_key == "secret"
    assert "secret" not in " ".join(think_generate_command(request, tmp_path))

    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"data": [{"id": "m"}]}'

    def fake_urlopen(req, timeout):
        seen["auth"] = req.get_header("Authorization")
        return FakeResponse()

    monkeypatch.setattr(webui_main.urllib.request, "urlopen", fake_urlopen)
    assert webui_main.ollama_models("http://pod:8000", "openai", "secret") == [{"name": "m", "size": None}]
    assert seen["auth"] == "Bearer secret"
    probe = client.get("/api/think/ollama", params={"url": "http://pod:8000", "api": "openai", "api_key": "secret"})
    assert probe.status_code == 200 and probe.json()["models"] == [{"name": "m", "size": None}]

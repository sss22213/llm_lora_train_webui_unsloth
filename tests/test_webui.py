import time
from pathlib import Path

from fastapi.testclient import TestClient

import webui.main as webui_main
from webui.main import JobManager, TrainingRequest, app, safe_slug


client = TestClient(app)


def test_index_and_health_are_available():
    index = client.get("/")
    assert index.status_code == 200
    assert "LoRA Forge" in index.text

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_fable_request_gets_safe_defaults():
    request = TrainingRequest(
        dataset="Crownelius/Complete-FABLE.5-traces-2M",
        dataset_format="fable_trace",
    )
    assert request.source_filter == "greghavens/fable-5-coding-and-debugging-traces"
    assert request.assistant_only_loss is True


def test_text_dataset_disables_assistant_only_loss():
    request = TrainingRequest(dataset_format="text", assistant_only_loss=True)
    assert request.assistant_only_loss is False


def test_safe_slug_rejects_path_characters():
    assert safe_slug("../Qwen model / test") == "Qwen-model-test"
    assert "/" not in safe_slug("model/name")


def test_adapter_endpoint_returns_a_list():
    response = client.get("/api/adapters")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_static_assets_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "webui" / "static" / "index.html").is_file()
    assert (root / "webui" / "static" / "styles.css").is_file()
    assert (root / "webui" / "static" / "app.js").is_file()


def test_job_manager_runs_subprocess_and_detects_adapter(tmp_path, monkeypatch):
    work_dir = tmp_path / "work"
    jobs_dir = tmp_path / "jobs"
    output_dir = tmp_path / "outputs"
    work_dir.mkdir()
    jobs_dir.mkdir()
    output_dir.mkdir()
    runner = work_dir / "fake_runner.py"
    runner.write_text(
        """\
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
adapter = Path(config["output_directory"]) / "adapter"
adapter.mkdir(parents=True)
(adapter / "adapter_config.json").write_text('{"r": 16}')
(adapter / "adapter_model.safetensors").write_bytes(b"test")
print("fake training complete", flush=True)
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(webui_main, "WORK_DIR", work_dir)
    monkeypatch.setattr(webui_main, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(webui_main, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(webui_main, "RUNNER", runner)

    local_manager = JobManager()
    job = local_manager.create(TrainingRequest(output_name="test-run"))
    deadline = time.monotonic() + 5
    while local_manager.get(job.id).status in ("queued", "running") and time.monotonic() < deadline:
        time.sleep(0.02)

    completed = local_manager.get(job.id)
    assert completed.status == "completed"
    assert "fake training complete" in local_manager.log(job.id, 0)[0]
    assert (Path(completed.output_directory) / "adapter" / "adapter_config.json").is_file()

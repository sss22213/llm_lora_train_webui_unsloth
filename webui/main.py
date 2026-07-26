from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from webui.unsloth_version import UnslothVersionManager


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "webui" / "static"
DATA_DIR = Path(os.getenv("APP_DATA_DIR", ROOT / "webui-data")).resolve()
OUTPUT_DIR = Path(os.getenv("APP_OUTPUT_DIR", ROOT / "work" / "outputs")).resolve()
WORK_DIR = Path(os.getenv("APP_WORK_DIR", ROOT / "work")).resolve()
RUNNER = Path(os.getenv("APP_TRAIN_RUNNER", WORK_DIR / "webui_runner.py")).resolve()
JOBS_DIR = DATA_DIR / "jobs"
HF_TOKEN_FILE = DATA_DIR / "hf_token"


def default_unsloth_dir() -> Path:
    configured = os.getenv("APP_UNSLOTH_DIR")
    if configured:
        return Path(configured)
    container_path = Path("/opt/unsloth-src")
    return container_path if container_path.exists() else ROOT / "unsloth"


UNSLOTH_DIR = default_unsloth_dir()

for directory in (DATA_DIR, JOBS_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    return slug[:72] or "lora-model"


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


class TrainingRequest(BaseModel):
    model: str = Field(default="unsloth/Qwen3.5-9B", min_length=1, max_length=300)
    model_family: Literal["auto", "language", "multimodal"] = "auto"
    hf_token: str | None = Field(default=None, max_length=2048, exclude=True, repr=False)
    output_name: str = Field(
        default="qwen3.5-finetune",
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9._-]+$",
    )

    dataset: str = Field(default="mlabonne/FineTome-100k", min_length=1, max_length=300)
    dataset_config: str | None = Field(default=None, max_length=200)
    dataset_split: str = Field(default="train", min_length=1, max_length=120)
    dataset_format: Literal[
        "sharegpt", "messages", "text", "prompt_completion", "fable_trace"
    ] = "sharegpt"
    source_filter: str | None = Field(default=None, max_length=300)
    text_field: str = Field(default="text", min_length=1, max_length=120)
    max_samples: int = Field(default=3000, ge=0, le=2_000_000)
    filter_overlength: bool = True

    max_seq_length: int = Field(default=4096, ge=256, le=131072)
    load_in_4bit: bool = True
    lora_r: int = Field(default=16, ge=1, le=512)
    lora_alpha: int = Field(default=16, ge=1, le=1024)
    lora_dropout: float = Field(default=0.0, ge=0.0, le=0.5)
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        min_length=1,
        max_length=32,
    )

    batch_size: int = Field(default=1, ge=1, le=128)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=1024)
    learning_rate: float = Field(default=2e-4, gt=0, le=1.0)
    warmup_steps: int = Field(default=10, ge=0, le=100000)
    max_steps: int = Field(default=100, ge=0, le=10_000_000)
    num_train_epochs: float = Field(default=1.0, gt=0, le=1000)
    logging_steps: int = Field(default=5, ge=1, le=100000)
    save_steps: int = Field(default=50, ge=1, le=1_000_000)
    optim: Literal["adamw_8bit", "adamw_torch"] = "adamw_8bit"
    weight_decay: float = Field(default=0.01, ge=0, le=1)
    lr_scheduler_type: Literal["linear", "cosine", "constant"] = "linear"
    seed: int = Field(default=3407, ge=0, le=2**31 - 1)
    packing: bool = False
    assistant_only_loss: bool = True

    @field_validator(
        "model", "dataset", "dataset_config", "dataset_split", "source_filter", "text_field",
        "hf_token",
    )
    @classmethod
    def normalize_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if any(ord(char) < 32 for char in value):
            raise ValueError("不可包含控制字元")
        if not value and info.field_name in {"model", "dataset", "dataset_split", "text_field"}:
            raise ValueError("不可為空白")
        return value or None

    @field_validator("target_modules")
    @classmethod
    def validate_target_modules(cls, values: list[str]) -> list[str]:
        cleaned = []
        for value in values:
            value = value.strip()
            if not re.fullmatch(r"[a-zA-Z0-9_.-]+", value):
                raise ValueError(f"無效的 target module：{value}")
            if value not in cleaned:
                cleaned.append(value)
        if not cleaned:
            raise ValueError("至少需要一個 target module")
        return cleaned

    @model_validator(mode="after")
    def validate_dataset_options(self):
        if self.dataset_format == "fable_trace" and not self.source_filter:
            self.source_filter = "greghavens/fable-5-coding-and-debugging-traces"
        if self.dataset_format == "text" and self.assistant_only_loss:
            self.assistant_only_loss = False
        return self


@dataclass
class Job:
    id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    request: dict
    output_directory: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None
    pid: int | None = None
    log_size: int = 0
    command: list[str] = field(default_factory=list)


class HFTokenStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.RLock()

    def get(self) -> str | None:
        with self.lock:
            try:
                return self.path.read_text(encoding="utf-8").strip() or None
            except OSError:
                return None

    def save(self, token: str) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(token + "\n")
                temporary.replace(self.path)
                self.path.chmod(0o600)
            finally:
                temporary.unlink(missing_ok=True)


def adapter_complete(output_directory: Path) -> bool:
    adapter = output_directory / "adapter"
    if not (adapter / "adapter_config.json").is_file():
        return False
    return any(
        (adapter / filename).is_file()
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    )


hf_token_store = HFTokenStore(HF_TOKEN_FILE)


class JobManager:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.jobs: dict[str, Job] = {}
        self.processes: dict[str, subprocess.Popen] = {}
        self.job_tokens: dict[str, str] = {}
        self._load_jobs()

    def _job_dir(self, job_id: str) -> Path:
        return JOBS_DIR / job_id

    def _persist(self, job: Job) -> None:
        directory = self._job_dir(job.id)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "job.json.tmp"
        temporary.write_text(
            json.dumps(asdict(job), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(directory / "job.json")

    def _load_jobs(self) -> None:
        for metadata in JOBS_DIR.glob("*/job.json"):
            try:
                job = Job(**json.loads(metadata.read_text(encoding="utf-8")))
                if job.status in ("queued", "running"):
                    job.status = "failed"
                    job.finished_at = utc_now()
                    job.error = "WebUI 服務重啟，原訓練程序已失聯；可從 checkpoint 重試。"
                    self._persist(job)
                elif job.status == "completed" and not adapter_complete(Path(job.output_directory)):
                    job.status = "failed"
                    job.error = "訓練程序已結束，但找不到完整 LoRA adapter。"
                    self._persist(job)
                self.jobs[job.id] = job
            except (OSError, TypeError, ValueError):
                continue

    def list(self) -> list[Job]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda job: job.created_at, reverse=True)

    def get(self, job_id: str) -> Job:
        with self.lock:
            try:
                return self.jobs[job_id]
            except KeyError as exc:
                raise KeyError(job_id) from exc

    def create(self, request: TrainingRequest) -> Job:
        with self.lock:
            if any(job.status in ("queued", "running") for job in self.jobs.values()):
                raise RuntimeError("GPU 正由另一個訓練任務使用中")
            if not RUNNER.is_file():
                raise RuntimeError(f"找不到訓練 runner：{RUNNER}")

            token = request.hf_token
            if token:
                hf_token_store.save(token)

            job_id = uuid.uuid4().hex[:12]
            output_name = safe_slug(request.output_name)
            output_directory = OUTPUT_DIR / f"{output_name}-{job_id[:6]}"
            public_request = request.model_dump(exclude={"hf_token"})
            config = {
                **public_request,
                "job_id": job_id,
                "output_directory": str(output_directory),
            }
            command = [sys.executable, str(RUNNER), str(self._job_dir(job_id) / "config.json")]
            job = Job(
                id=job_id,
                status="queued",
                request=public_request,
                output_directory=str(output_directory),
                created_at=utc_now(),
                command=command,
            )

            job_dir = self._job_dir(job_id)
            job_dir.mkdir(parents=True, exist_ok=False)
            (job_dir / "config.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.jobs[job_id] = job
            if token:
                self.job_tokens[job_id] = token
            self._persist(job)
            threading.Thread(target=self._run, args=(job_id,), daemon=True).start()
            return job

    def _run(self, job_id: str) -> None:
        job_dir = self._job_dir(job_id)
        log_path = job_dir / "run.log"
        with self.lock:
            job = self.jobs[job_id]
            if job.status == "cancelled":
                job.finished_at = utc_now()
                self._persist(job)
                return
            token = self.job_tokens.pop(job_id, None) or hf_token_store.get()
            job.status = "running"
            job.started_at = utc_now()
            self._persist(job)

        environment = os.environ.copy()
        environment.update({"PYTHONUNBUFFERED": "1", "TERM": "dumb", "NO_COLOR": "1"})
        if token:
            environment.update({"HF_TOKEN": token, "HUGGING_FACE_HUB_TOKEN": token})

        try:
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    job.command,
                    cwd=WORK_DIR,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=environment,
                    start_new_session=True,
                )
                with self.lock:
                    self.processes[job_id] = process
                    was_cancelled = job.status == "cancelled"
                    job.pid = process.pid if not was_cancelled else None
                    self._persist(job)
                if was_cancelled:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                exit_code = process.wait()

            with self.lock:
                if job.status != "cancelled":
                    complete = adapter_complete(Path(job.output_directory))
                    job.status = "completed" if exit_code == 0 and complete else "failed"
                    if exit_code == 0 and not complete:
                        job.error = "程序正常結束，但沒有產生完整 LoRA adapter。"
                    elif exit_code != 0:
                        job.error = f"訓練程序結束碼：{exit_code}"
                job.exit_code = exit_code
        except Exception as exc:
            with self.lock:
                job.status = "failed"
                job.error = str(exc)
        finally:
            with self.lock:
                self.processes.pop(job_id, None)
                job.pid = None
                job.finished_at = utc_now()
                job.log_size = log_path.stat().st_size if log_path.exists() else 0
                self._persist(job)

    def cancel(self, job_id: str) -> Job:
        with self.lock:
            job = self.get(job_id)
            if job.status not in ("queued", "running"):
                raise RuntimeError("此任務目前無法取消")
            job.status = "cancelled"
            job.error = "使用者取消訓練"
            process = self.processes.get(job_id)
            self._persist(job)

        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return job

    def retry(self, job_id: str) -> Job:
        with self.lock:
            job = self.get(job_id)
            if job.status not in ("failed", "cancelled"):
                raise RuntimeError("只有失敗或取消的任務可以重試")
            if any(
                other.status in ("queued", "running")
                for other in self.jobs.values()
                if other.id != job_id
            ):
                raise RuntimeError("GPU 正由另一個訓練任務使用中")
            job.status = "queued"
            job.started_at = None
            job.finished_at = None
            job.exit_code = None
            job.error = None
            job.pid = None
            self._persist(job)

            log_path = self._job_dir(job_id) / "run.log"
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n--- {utc_now()}：重試任務，若有 checkpoint 將自動續訓 ---\n")
            threading.Thread(target=self._run, args=(job_id,), daemon=True).start()
            return job

    def log(self, job_id: str, offset: int) -> tuple[str, int]:
        self.get(job_id)
        path = self._job_dir(job_id) / "run.log"
        if not path.exists():
            return "", 0
        size = path.stat().st_size
        offset = min(max(offset, 0), size)
        with path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read(256 * 1024)
        return chunk.decode("utf-8", errors="replace"), offset + len(chunk)


manager = JobManager()


def scan_adapters() -> list[dict]:
    adapters: list[dict] = []
    for output in OUTPUT_DIR.iterdir() if OUTPUT_DIR.is_dir() else []:
        adapter = output / "adapter"
        config_path = adapter / "adapter_config.json"
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
        try:
            size = sum(path.stat().st_size for path in adapter.rglob("*") if path.is_file())
            modified = datetime.fromtimestamp(adapter.stat().st_mtime, UTC).isoformat()
        except OSError:
            size = 0
            modified = None
        adapters.append(
            {
                "name": output.name,
                "path": str(adapter),
                "base_model": config.get("base_model_name_or_path"),
                "lora_r": config.get("r"),
                "size": size,
                "modified_at": modified,
            }
        )
    return sorted(adapters, key=lambda item: item["modified_at"] or "", reverse=True)


app = FastAPI(title="LoRA Forge WebUI", version="1.0.0", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def prevent_stale_assets(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    active = [job for job in manager.list() if job.status in ("queued", "running")]
    return {
        "status": "ok",
        "runner_available": RUNNER.is_file(),
        "active_job": asdict(active[0]) if active else None,
    }


@app.get("/api/system")
def system_info():
    gpus: list[dict[str, str | int]] = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            for index, line in enumerate(result.stdout.splitlines()):
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 6:
                    gpus.append(
                        {
                            "index": index,
                            "name": parts[0],
                            "memory_total_mb": int(parts[1]),
                            "memory_used_mb": int(parts[2]),
                            "memory_free_mb": int(parts[3]),
                            "utilization": int(parts[4]),
                            "temperature": int(parts[5]),
                        }
                    )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass

    disk = shutil.disk_usage(OUTPUT_DIR)
    return {
        "gpus": gpus,
        "output_directory": str(OUTPUT_DIR),
        "disk_free": disk.free,
        "disk_total": disk.total,
        "hf_token_saved": hf_token_store.get() is not None,
        "versions": {
            name: package_version(name)
            for name in ("unsloth", "torch", "transformers", "trl", "datasets")
        },
    }


@app.get("/api/jobs")
def list_jobs():
    return [asdict(job) for job in manager.list()]


@app.post("/api/jobs", status_code=202)
def create_job(request: TrainingRequest):
    try:
        return asdict(manager.create(request))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return asdict(manager.get(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="找不到任務") from exc


@app.get("/api/jobs/{job_id}/log")
def get_job_log(job_id: str, offset: int = Query(default=0, ge=0)):
    try:
        content, next_offset = manager.log(job_id, offset)
        return {"content": content, "offset": next_offset}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="找不到任務") from exc


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        return asdict(manager.cancel(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="找不到任務") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/retry", status_code=202)
def retry_job(job_id: str):
    try:
        return asdict(manager.retry(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="找不到任務") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/adapters")
def list_adapters():
    return scan_adapters()


unsloth_version_manager = UnslothVersionManager(
    repo_dir=UNSLOTH_DIR,
    state_file=DATA_DIR / "unsloth_version.json",
)


class UnslothVersionActionRequest(BaseModel):
    confirmation: str = Field(min_length=1, max_length=20)
    ref: str | None = Field(default=None, min_length=1, max_length=120)


def ensure_unsloth_version_idle() -> None:
    if any(job.status in ("queued", "running") for job in manager.list()):
        raise HTTPException(status_code=409, detail="訓練任務執行中，無法切換 Unsloth 版本")


def with_package_version(result: dict) -> dict:
    result["package_version"] = package_version("unsloth")
    return result


@app.get("/api/unsloth/version")
def get_unsloth_version(check_remote: bool = False):
    try:
        return with_package_version(unsloth_version_manager.status(check_remote=check_remote))
    except RuntimeError as exc:
        raise HTTPException(status_code=502 if check_remote else 409, detail=str(exc)) from exc


@app.get("/api/unsloth/version/tags")
def get_unsloth_tags(limit: int = Query(default=40, ge=1, le=100)):
    try:
        return {"tags": unsloth_version_manager.remote_tags(limit=limit)}
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/unsloth/version/update")
def update_unsloth_version(request: UnslothVersionActionRequest):
    if request.confirmation != "UPDATE":
        raise HTTPException(status_code=400, detail="更新確認值不正確")
    ensure_unsloth_version_idle()
    try:
        return with_package_version(unsloth_version_manager.update())
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/unsloth/version/checkout")
def checkout_unsloth_version(request: UnslothVersionActionRequest):
    if request.confirmation != "CHECKOUT":
        raise HTTPException(status_code=400, detail="切換確認值不正確")
    if not request.ref:
        raise HTTPException(status_code=400, detail="缺少要切換的版本名稱")
    ensure_unsloth_version_idle()
    try:
        return with_package_version(unsloth_version_manager.checkout(request.ref))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/unsloth/version/rollback")
def rollback_unsloth_version(request: UnslothVersionActionRequest):
    if request.confirmation != "ROLLBACK":
        raise HTTPException(status_code=400, detail="退版確認值不正確")
    ensure_unsloth_version_idle()
    try:
        return with_package_version(unsloth_version_manager.rollback())
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

from __future__ import annotations

import ipaddress
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
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
THINK_SCRIPT = Path(os.getenv("APP_THINK_SCRIPT", WORK_DIR / "rationalize_think.py")).resolve()
DATASETS_DIR = WORK_DIR / "datasets"
THINK_JOBS_DIR = DATA_DIR / "think_jobs"
OLLAMA_URL = os.getenv("APP_OLLAMA_URL", "http://host.docker.internal:11434")


def default_unsloth_dir() -> Path:
    configured = os.getenv("APP_UNSLOTH_DIR")
    if configured:
        return Path(configured)
    container_path = Path("/opt/unsloth-src")
    return container_path if container_path.exists() else ROOT / "unsloth"


UNSLOTH_DIR = default_unsloth_dir()

for directory in (DATA_DIR, JOBS_DIR, OUTPUT_DIR, THINK_JOBS_DIR):
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
        "sharegpt", "messages", "messages_json", "text", "prompt_completion", "fable_trace"
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
    empty_think: Literal["train", "mask", "strip"] = "train"

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
            ensure_no_think_generation()
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
            ensure_no_think_generation()
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


# ---------------------------------------------------------------------------
# 推理補完：用底模（經 Ollama）替沒有推理內容的對話資料集補上 <think>，
# 由 work/rationalize_think.py 執行；這裡只負責排程、log 與資料集掃描。
# ---------------------------------------------------------------------------

DEFAULT_THINK_DATASET = "Aratako/Synthetic-Japanese-Roleplay-NSFW-DeepSeek-V3-0324-20k-formatted"
DEFAULT_THINK_MODEL = "qwen3.8-27b-heretic-6fcab5-kl:latest"
PROGRESS_PATTERN = re.compile(r"\[(\d+)/(\d+)\]")


class ThinkGenerateRequest(BaseModel):
    dataset: str = Field(default=DEFAULT_THINK_DATASET, min_length=1, max_length=300)
    split: str = Field(default="train", min_length=1, max_length=120)
    output_name: str = Field(
        default="aratako-rp-think", min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9._-]+$"
    )
    mode: Literal["free", "rationalize"] = "free"
    # ollama：/api/chat。openai：/v1/chat/completions（llama-server、vLLM）；
    # Ollama 對 qwen35 架構強制單路，要多路並行得在伺服器端跑 llama-server -np N 並選 openai。
    api: Literal["ollama", "openai"] = "ollama"
    ollama_url: str = Field(default=OLLAMA_URL, min_length=1, max_length=300)
    ollama_model: str = Field(default=DEFAULT_THINK_MODEL, min_length=1, max_length=200)
    api_key: str = Field(default="", max_length=300)  # vLLM --api-key；經環境變數交給腳本，不進命令列
    samples: int = Field(default=1000, ge=1, le=500_000)
    turn: Literal["last", "random"] = "last"
    parallel: int = Field(default=4, ge=1, le=32)
    seed: int = Field(default=3407, ge=0, le=2**31 - 1)
    num_ctx: int = Field(default=16384, ge=1024, le=131072)
    num_predict: int = Field(default=5120, ge=64, le=32768)
    retry_on_length: int = Field(default=2, ge=0, le=10)
    min_think_chars: int = Field(default=200, ge=0, le=100_000)
    max_think_chars: int = Field(default=6000, ge=0, le=1_000_000)
    min_content_chars: int = Field(default=10, ge=0, le=100_000)

    @field_validator("dataset", "split", "ollama_model", "api_key")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("ollama_url")
    @classmethod
    def normalize_ollama_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("Ollama URL 必須以 http:// 或 https:// 開頭")
        return value


class ThinkBuildRequest(BaseModel):
    output_name: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9._-]+$")
    dataset: str | None = Field(default=None, max_length=300)
    split: str = Field(default="train", min_length=1, max_length=120)
    min_think_chars: int = Field(default=200, ge=0, le=100_000)
    max_think_chars: int = Field(default=6000, ge=0, le=1_000_000)
    min_content_chars: int = Field(default=10, ge=0, le=100_000)
    only_with_think: bool = False


def think_dataset_dir(output_name: str) -> Path:
    return DATASETS_DIR / safe_slug(output_name)


def is_local_ollama(url: str) -> bool:
    """Ollama 跑在這台機器（或同一個區網）時才會和訓練搶 GPU；RunPod 之類的遠端不用互斥。"""
    host = (urllib.parse.urlsplit(url.strip()).hostname or "").lower()
    if host in ("localhost", "host.docker.internal") or host.endswith(".local"):
        return True
    try:
        return ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def think_generate_command(request: ThinkGenerateRequest, out_dir: Path) -> list[str]:
    return [
        sys.executable, str(THINK_SCRIPT),
        "--dataset", request.dataset, "--split", request.split, "--out", str(out_dir),
        "--min-think-chars", str(request.min_think_chars),
        "--max-think-chars", str(request.max_think_chars),
        "--min-content-chars", str(request.min_content_chars),
        "generate",
        "--mode", request.mode, "--api", request.api,
        "--model", request.ollama_model, "--ollama", request.ollama_url,
        "--samples", str(request.samples), "--turn", request.turn, "--seed", str(request.seed),
        "--parallel", str(request.parallel), "--num-ctx", str(request.num_ctx),
        "--num-predict", str(request.num_predict), "--retry-on-length", str(request.retry_on_length),
        "--report-every", "10",
    ]


def think_build_command(request: ThinkBuildRequest, out_dir: Path, dataset: str) -> list[str]:
    command = [
        sys.executable, str(THINK_SCRIPT),
        "--dataset", dataset, "--split", request.split, "--out", str(out_dir),
        "--min-think-chars", str(request.min_think_chars),
        "--max-think-chars", str(request.max_think_chars),
        "--min-content-chars", str(request.min_content_chars),
        "build",
    ]
    if request.only_with_think:
        command.append("--only-with-think")
    return command


def parse_progress(text: str) -> dict | None:
    """從 generate 的 log 取最後一行進度（[n/total] ...）。"""
    last = None
    for line in text.splitlines():
        match = PROGRESS_PATTERN.search(line)
        if match:
            last = {"done": int(match.group(1)), "total": int(match.group(2)), "text": line.strip()}
    return last


def read_log_tail(path: Path, size: int = 4096) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(max(path.stat().st_size - size, 0))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


@dataclass
class ThinkJob:
    id: str
    kind: Literal["generate", "build"]
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


class ThinkJobManager:
    """和 JobManager 相同的流程，但跑的是推理補完腳本，且 generate 與訓練互斥。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.jobs: dict[str, ThinkJob] = {}
        self.processes: dict[str, subprocess.Popen] = {}
        self._load_jobs()

    def _job_dir(self, job_id: str) -> Path:
        return THINK_JOBS_DIR / job_id

    def _persist(self, job: ThinkJob) -> None:
        directory = self._job_dir(job.id)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "job.json.tmp"
        temporary.write_text(json.dumps(asdict(job), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(directory / "job.json")

    def _load_jobs(self) -> None:
        for metadata in THINK_JOBS_DIR.glob("*/job.json"):
            try:
                job = ThinkJob(**json.loads(metadata.read_text(encoding="utf-8")))
                if job.status in ("queued", "running"):
                    job.status = "failed"
                    job.finished_at = utc_now()
                    job.error = "WebUI 服務重啟，原程序已失聯；再送一次同名任務會從已完成的樣本續跑。"
                    self._persist(job)
                self.jobs[job.id] = job
            except (OSError, TypeError, ValueError):
                continue

    def list(self) -> list[ThinkJob]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda job: job.created_at, reverse=True)

    def get(self, job_id: str) -> ThinkJob:
        with self.lock:
            try:
                return self.jobs[job_id]
            except KeyError as exc:
                raise KeyError(job_id) from exc

    def running(self) -> list[ThinkJob]:
        with self.lock:
            return [job for job in self.jobs.values() if job.status in ("queued", "running")]

    def active(self) -> ThinkJob | None:
        """優先回傳生成任務（跑得久、佔資源），其次才是組資料集。"""
        jobs = self.running()
        for job in jobs:
            if job.kind == "generate":
                return job
        return jobs[0] if jobs else None

    def generation_active(self) -> bool:
        """有正在跑、而且用的是本機 GPU 的生成任務。"""
        return any(
            job.kind == "generate" and is_local_ollama(str(job.request.get("ollama_url", "")))
            for job in self.running()
        )

    def create(self, kind: str, request: dict, command: list[str], output_directory: Path) -> ThinkJob:
        with self.lock:
            # 生成與組資料集可以並行（build 只讀 generated.jsonl，會略過寫到一半的行），
            # 但同類任務一次只能一個。
            if any(job.kind == kind for job in self.running()):
                label = "生成" if kind == "generate" else "組資料集"
                raise RuntimeError(f"已有推理補完{label}任務在執行中")
            if (
                kind == "generate"
                and is_local_ollama(str(request.get("ollama_url", "")))
                and any(job.status in ("queued", "running") for job in manager.list())
            ):
                raise RuntimeError("訓練任務執行中，本機 Ollama 生成會和訓練搶 GPU；改用遠端 Ollama 或等訓練結束")
            if not THINK_SCRIPT.is_file():
                raise RuntimeError(f"找不到推理補完腳本：{THINK_SCRIPT}")
            job_id = uuid.uuid4().hex[:12]
            job = ThinkJob(
                id=job_id, kind=kind, status="queued", request=request,
                output_directory=str(output_directory), created_at=utc_now(), command=command,
            )
            self._job_dir(job_id).mkdir(parents=True, exist_ok=False)
            self.jobs[job_id] = job
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
            job.status = "running"
            job.started_at = utc_now()
            self._persist(job)

        environment = os.environ.copy()
        environment.update({"PYTHONUNBUFFERED": "1", "TERM": "dumb", "NO_COLOR": "1"})
        token = hf_token_store.get()
        if token:
            environment.update({"HF_TOKEN": token, "HUGGING_FACE_HUB_TOKEN": token})
        if job.request.get("api_key"):
            environment["THINK_API_KEY"] = str(job.request["api_key"])

        try:
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    job.command, cwd=WORK_DIR, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, env=environment, start_new_session=True,
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
                    artifact = "generated.jsonl" if job.kind == "generate" else "train.jsonl"
                    produced = (Path(job.output_directory) / artifact).is_file()
                    job.status = "completed" if exit_code == 0 and produced else "failed"
                    if exit_code != 0:
                        job.error = f"程序結束碼：{exit_code}"
                    elif not produced:
                        job.error = f"程序正常結束，但沒有產生 {artifact}。"
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

    def cancel(self, job_id: str) -> ThinkJob:
        with self.lock:
            job = self.get(job_id)
            if job.status not in ("queued", "running"):
                raise RuntimeError("此任務目前無法取消")
            job.status = "cancelled"
            job.error = "使用者取消；已完成的樣本保留，同名任務可續跑"
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

    def serialize(self, job: ThinkJob) -> dict:
        data = asdict(job)
        if job.kind == "generate":
            data["progress"] = parse_progress(read_log_tail(self._job_dir(job.id) / "run.log"))
        return data


def ensure_no_think_generation() -> None:
    if think_manager.generation_active():
        raise RuntimeError("推理補完（Ollama 生成）執行中，GPU 忙碌；請先等它完成或取消")


_line_count_cache: dict[str, tuple[int, float, int]] = {}


def count_lines(path: Path) -> int:
    try:
        stat = path.stat()
    except OSError:
        return 0
    cached = _line_count_cache.get(str(path))
    if cached and cached[0] == stat.st_size and cached[1] == stat.st_mtime:
        return cached[2]
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            total += chunk.count(b"\n")
    _line_count_cache[str(path)] = (stat.st_size, stat.st_mtime, total)
    return total


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def file_time(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
    except OSError:
        return None


def scan_think_datasets() -> list[dict]:
    items: list[dict] = []
    for directory in DATASETS_DIR.iterdir() if DATASETS_DIR.is_dir() else []:
        generated = directory / "generated.jsonl"
        if not directory.is_dir() or not generated.is_file():
            continue
        train = directory / "train.jsonl"
        items.append(
            {
                "name": directory.name,
                "path": str(directory),
                "meta": read_json(directory / "meta.json"),
                "generated_rows": count_lines(generated),
                "generated_at": file_time(generated),
                "train_rows": count_lines(train) if train.is_file() else None,
                "train_at": file_time(train) if train.is_file() else None,
                "build_stats": read_json(directory / "build_stats.json"),
            }
        )
    return sorted(items, key=lambda item: item["generated_at"] or "", reverse=True)


def ollama_models(url: str, api: str = "ollama", api_key: str = "") -> list[dict]:
    """列出推理伺服器上的模型：Ollama 走 /api/tags，OpenAI 相容（llama-server、vLLM）走 /v1/models。"""
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise ValueError("Ollama URL 必須以 http:// 或 https:// 開頭")
    path = "/api/tags" if api == "ollama" else "/v1/models"
    # RunPod 等 proxy 會用 403 擋 Python-urllib 預設的 User-Agent，自報名字才過得去。
    headers = {"Accept": "application/json", "User-Agent": "LoRA-Forge/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{url}{path}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        hint = "；伺服器要求 API key（vLLM 的 --api-key / VLLM_API_KEY），請填在表單的 API key 欄" if exc.code in (401, 403) else ""
        raise RuntimeError(f"連不到推理伺服器（{url}{path}）：{exc}{hint}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(f"連不到推理伺服器（{url}{path}）：{exc}") from exc
    if api == "openai":
        return [{"name": model.get("id"), "size": None} for model in payload.get("data", []) if model.get("id")]
    return [
        {"name": model.get("name"), "size": model.get("size")}
        for model in payload.get("models", [])
        if model.get("name")
    ]


think_manager = ThinkJobManager()


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
        "active_think_job": think_manager.serialize(think_manager.active()) if think_manager.active() else None,
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


@app.get("/api/think/jobs")
def list_think_jobs():
    return [think_manager.serialize(job) for job in think_manager.list()]


@app.post("/api/think/generate", status_code=202)
def create_think_generate(request: ThinkGenerateRequest):
    out_dir = think_dataset_dir(request.output_name)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = read_json(out_dir / "meta.json")
        meta.update(
            dataset=request.dataset, split=request.split, mode=request.mode, api=request.api,
            ollama_model=request.ollama_model, turn=request.turn, updated_at=utc_now(),
        )
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        job = think_manager.create(
            "generate", request.model_dump(), think_generate_command(request, out_dir), out_dir
        )
        return think_manager.serialize(job)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"無法建立資料集目錄：{exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/think/build", status_code=202)
def create_think_build(request: ThinkBuildRequest):
    out_dir = think_dataset_dir(request.output_name)
    if not (out_dir / "generated.jsonl").is_file():
        raise HTTPException(status_code=404, detail="找不到 generated.jsonl，請先產生推理")
    meta = read_json(out_dir / "meta.json")
    dataset = (request.dataset or "").strip() or meta.get("dataset")
    if not dataset:
        raise HTTPException(status_code=400, detail="缺少原始資料集名稱（meta.json 沒有紀錄，請手動填寫）")
    try:
        job = think_manager.create(
            "build", request.model_dump(), think_build_command(request, out_dir, dataset), out_dir
        )
        return think_manager.serialize(job)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/think/jobs/{job_id}")
def get_think_job(job_id: str):
    try:
        return think_manager.serialize(think_manager.get(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="找不到任務") from exc


@app.get("/api/think/jobs/{job_id}/log")
def get_think_job_log(job_id: str, offset: int = Query(default=0, ge=0)):
    try:
        content, next_offset = think_manager.log(job_id, offset)
        return {"content": content, "offset": next_offset}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="找不到任務") from exc


@app.post("/api/think/jobs/{job_id}/cancel")
def cancel_think_job(job_id: str):
    try:
        return think_manager.serialize(think_manager.cancel(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="找不到任務") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/think/datasets")
def list_think_datasets():
    return scan_think_datasets()


@app.get("/api/think/ollama")
def probe_ollama(
    url: str = Query(default=OLLAMA_URL, min_length=1, max_length=300),
    api: Literal["ollama", "openai"] = Query(default="ollama"),
    api_key: str = Query(default="", max_length=300),
):
    try:
        return {"url": url.strip().rstrip("/"), "api": api, "models": ollama_models(url, api, api_key.strip())}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


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

# LoRA Forge — Unsloth LoRA/QLoRA Training Environment

A LoRA/QLoRA fine-tuning environment built on the official
[Unsloth](https://github.com/unslothai/unsloth) Docker image, targeting the
RTX 5090 (Blackwell, sm_120). Target models: Gemma 4 12B, Qwen3.5, and the
Qwen3.6 family. Ships with the **LoRA Forge WebUI** for creating, cancelling,
resuming, and monitoring training jobs from the browser, including one-click
reuse of past training configurations.

## Directory layout

```
llm_lora_train/
├── Dockerfile             # Official unsloth/unsloth image (pinned by digest) + LoRA Forge deps
├── docker-compose.yml     # GPU (Docker native CDI), ports, volumes
├── .env.example           # Environment variable template (copy to .env)
├── DESIGN.md              # WebUI design system (Linear-style dark theme)
├── unsloth/               # Unsloth source code (vendored, see below)
├── webui/                 # LoRA Forge FastAPI backend + vanilla JS frontend
├── tests/                 # pytest suite (mounted read-only in the container)
├── webui-data/            # WebUI jobs, logs, private HF token (created at runtime, not in VCS)
└── work/                  # Mounted at /workspace/work
    ├── webui_runner.py    # Generic Unsloth training runner used by the WebUI
    ├── patches/           # Unsloth compiled-cache auto-repair (see below)
    └── check_env.py       # Environment health-check script
```

## Prerequisites

- NVIDIA driver and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) ≥ 1.19,
  with the CDI spec `/etc/cdi/nvidia.yaml` generated (usually kept up to date by
  `nvidia-cdi-refresh.service`; otherwise run
  `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`)
- Docker ≥ 28 (compose mounts the GPU via Docker native CDI)
- Add yourself to the docker group (one-time):

```bash
sudo usermod -aG docker $USER
newgrp docker   # or log out and back in
```

## Usage

```bash
cp .env.example .env       # fill in HF_TOKEN (required for gated models such as Gemma)

docker compose build       # first build (pulls the official image; large, takes a while)
docker compose up -d       # start
docker compose exec unsloth python /workspace/work/check_env.py   # health check

docker compose exec unsloth bash   # shell into the container
```

Service endpoints:

| Service | Location |
|---|---|
| LoRA Forge WebUI | http://localhost:6003 |
| JupyterLab | http://localhost:6001 (password = `JUPYTER_PASSWORD` in `.env`) |
| Unsloth Studio (optional) | http://localhost:6002 |
| SSH | port 6022 |

> **Security**: the WebUI has no authentication, and the compose port bindings
> listen on all interfaces. If the machine is not on a trusted network, change
> `"6003:8080"` in `docker-compose.yml` to `"127.0.0.1:6003:8080"` and access
> it through an SSH port forward. Do not expose it directly to the internet
> without authentication and TLS.

## LoRA Forge WebUI

Open <http://localhost:6003>. Main features:

- Three ready-made presets: FineTome, Complete FABLE, and Gemma 4.
- **Load past configurations**: pick a historical job from the list and fill
  the form with its full training parameters in one click (HF token excluded).
- Configure model, data format, LoRA rank/alpha, context length, batch size,
  and training schedule.
- Only one GPU training job runs at a time; background training, live logs,
  cancellation, and checkpoint resume.
- Scans `work/outputs` for completed PEFT adapters.
- The HF token is written only to `webui-data/hf_token`, never into job JSON
  or logs.

Tests:

```bash
docker compose run --rm --no-deps --entrypoint /opt/venv/bin/pytest webui -q /workspace/tests
node --check webui/static/app.js
```

## GPU access: Docker native CDI

The compose file uses Docker native CDI (`driver: cdi` + `nvidia.com/gpu=all`)
instead of `gpus: all`. Rationale: the legacy nvidia runtime hook edits device
permissions outside of cgroups, so any `systemctl daemon-reload` on the host
can revoke GPU access from long-running containers (symptom: training suddenly
fails with `Can't initialize NVML`). CDI writes device permissions into the OCI
spec where systemd can see them, fixing this for good. The setting is scoped to
this project only — it does not touch the global mode in
`/etc/nvidia-container-runtime/config.toml` and does not affect other
containers on the machine that use `gpus: all`.

## work/patches — Unsloth compiled-cache auto-repair

unsloth_zoo decompiles transformers modules with `inspect.getsource` to
generate `work/unsloth_compiled_cache/`. The `@force_accelerate_hooks`
decorator in transformers 5.14 lacks `functools.wraps`, so the generated code
for Qwen3.5/3.6 (the GatedDeltaNet layer) crashes with
`NameError: name 'args' is not defined`.

`webui_runner.py` runs a two-layer guard at every training start:

1. `patches/sanitize_unsloth_cache.py`: scans the existing cache and repairs
   broken forward stubs (unfixable files are deleted so unsloth regenerates
   them).
2. Restores the decorated forwards of the transformers module actually in use
   (looked up via `AutoConfig.model_type`), preventing broken caches from
   being regenerated after upgrades.

No manual steps are needed after upgrading unsloth / unsloth_zoo /
transformers. If the training log stops showing the「Unsloth 快取修補」line
after an upgrade, upstream has fixed the bug and this patch can be removed.
See [work/patches/README.md](work/patches/README.md) for details.

## Base image and unsloth/ source

The training stack (unsloth, unsloth_zoo, transformers, trl, torch) is the one
shipped inside the official `unsloth/unsloth` image, pinned **by digest** in the
`Dockerfile` (`ARG UNSLOTH_IMAGE`). `docker compose build` therefore never
changes the environment silently; to move to a newer image, pick a digest
(`docker buildx imagetools inspect unsloth/unsloth:<tag>` or the tag list on
Docker Hub), update the ARG, rebuild, and re-run a 100-step smoke job. The
build prints the resolved `python | torch | transformers | trl | unsloth | …`
versions right after the base image is loaded.

Nightly images since 2026-09 keep the venv at `/opt/unsloth-venv`; the
Dockerfile symlinks it to `/opt/venv`, so `docker-compose.yml`, this README and
old job records can keep using `/opt/venv/bin/python`.

`unsloth/` is a plain copy of upstream
[unslothai/unsloth](https://github.com/unslothai/unsloth) (commit `bb80602`,
2026-07-14, Apache-2.0, license file inside). It is **no longer installed** into
the container; it is only mounted at `/opt/unsloth-src` for the WebUI's
"Unsloth version" page, which is git-based and shows as unavailable for a plain
copy. To override the image's unsloth with your own checkout again, replace
`unsloth/` with a real git clone and add
`pip install --no-deps -e /opt/unsloth-src` back to the `Dockerfile`.

## Reasoning distillation (推理補完)

Chat datasets that ship only `role`/`content` (no reasoning) teach a thinking
model to skip its `<think>` phase. The **推理補完** view fixes that: it runs
`work/rationalize_think.py` against an Ollama server on the host, letting the
base model think and reply itself under the dataset's own system prompt and
history, and stores the reasoning as `reasoning_content` on the last assistant
turn. Qwen's chat template renders it as a real `<think>` block, so the LoRA
learns to reason in-domain instead of learning to stop.

- Generation is resumable: submitting the same output name again skips finished
  samples. Output lives in `work/datasets/<name>/` (`generated.jsonl`,
  `train.jsonl`, `meta.json`, `build_stats.json`).
- Generation through a *local* Ollama holds the GPU, so it is mutually
  exclusive with training jobs; a remote server (RunPod etc.) is not.
  Building `train.jsonl` can run while generation is still going.
- "組出 train.jsonl" applies the length thresholds and writes the training set;
  "用它建立訓練" pre-fills the training form (messages format, empty `<think>`
  blocks masked from the loss).
- The container reaches the host's Ollama via `host.docker.internal`
  (`extra_hosts` + `APP_OLLAMA_URL` in `docker-compose.yml`); override the URL in
  the form if Ollama runs elsewhere.

### Parallel generation with llama-server

Ollama serves the `qwen35` / `qwen35moe` architectures (Qwen3.5, 3.6, 3.8) one
request at a time: `server/sched.go` blocklists them and ignores
`OLLAMA_NUM_PARALLEL` (see ollama/ollama#14510, fix pending in #17144). For
batched generation run llama.cpp's `llama-server` on the same GGUF instead and
pick **llama-server / vLLM (OpenAI-compatible /v1)** as the 推理伺服器 in the
form (`--api openai` on the CLI). The script then streams
`/v1/chat/completions` and reads `delta.reasoning_content`; llama-server needs
`--jinja` (reasoning is split by `--reasoning-format auto`, the default).

Ollama keeps the model as a plain GGUF under `$OLLAMA_MODELS/blobs/`, so no
re-download is needed. On a pod that already has the model pulled:

```bash
GGUF=/workspace/ollama/blobs/$(ls -S /workspace/ollama/blobs | head -1)   # largest blob = the Q6_K model
llama-server -m "$GGUF" --host 0.0.0.0 --port 11434 -a qwen3.8-27b-heretic \
  -ngl 99 -np 12 -c 196608 -fa on -ctk q8_0 -ctv q8_0 --jinja --reasoning-format auto
```

`-np` is the number of slots (set the form's 並行請求 to the same value) and
`-c` is the total context shared by all slots (12 × 16384 here; check the
`n_ctx_seq` line in the log). The official CUDA image
`ghcr.io/ggml-org/llama.cpp:server-cuda` runs the same command with
`/app/llama-server` as its entrypoint. Q6_K weights (≈22 GB) plus a q8_0 KV
cache for 12 × 16k tokens fit a 48 GB card; a 96 GB card takes `-np 20`.

## Notes

- Model caches live in a docker volume named `hf-cache`; rebuilding containers
  does not re-download models.
- `ipc: host` is required — without it, multi-worker DataLoaders fail due to
  insufficient shared memory.
- `work/unsloth_compiled_cache/`, `work/outputs/`, `webui-data/`, and `.env`
  are excluded from version control (see `.gitignore`).

## License

This project is licensed under the [Apache License 2.0](LICENSE). The vendored
`unsloth/` directory is likewise Apache-2.0, with its original license file
shipped inside the directory.

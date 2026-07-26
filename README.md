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
├── Dockerfile             # Official unsloth/unsloth image + local source editable install
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

## unsloth/ — vendored source

`unsloth/` is the full source of
[unslothai/unsloth](https://github.com/unslothai/unsloth) (vendored from
upstream commit `bb80602`, 2026-07-14, Apache-2.0; the original LICENSE ships
inside the directory). Its `.git` has been removed and the directory is
managed directly by this project's git. The container points an editable
install at this directory, so source changes take effect on the **next
training job** (each job is a separate python process) — no image rebuild
required. JupyterLab kernels that already imported unsloth need a restart.

> Note: the WebUI's "Unsloth version" page is git-based (check for updates,
> pin tags, rollback) and shows as unavailable in vendored mode. To update
> unsloth, sync upstream manually, for example:
>
> ```bash
> git clone --depth 1 https://github.com/unslothai/unsloth.git /tmp/unsloth-new
> rsync -a --delete --exclude '.git' /tmp/unsloth-new/ unsloth/
> git add unsloth && git commit -m "unsloth: sync upstream <commit>"
> ```

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

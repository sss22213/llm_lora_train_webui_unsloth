# LoRA Forge — Unsloth LoRA/QLoRA 訓練環境

以 [Unsloth](https://github.com/unslothai/unsloth) 官方映像檔為基底的 LoRA/QLoRA
微調環境,針對 RTX 5090(Blackwell, sm_120)。目標模型:Gemma 4 12B、Qwen3.5、
Qwen3.6 系列。內附 **LoRA Forge WebUI**,可從瀏覽器建立、取消、續跑與監看訓練任務,
並載入過去的訓練設定重複使用。

## 目錄結構

```
llm_lora_train/
├── Dockerfile             # 官方 unsloth/unsloth 映像檔 + 本地原始碼 editable install
├── docker-compose.yml     # GPU(Docker 原生 CDI)、ports、volumes 設定
├── .env.example           # 環境變數範本(複製成 .env 使用)
├── DESIGN.md              # WebUI 設計系統(Linear 風格 dark theme)
├── unsloth/               # Unsloth 原始碼(vendored,見下方說明)
├── webui/                 # LoRA Forge FastAPI 後端 + 原生 JS 前端
├── tests/                 # pytest 測試(容器內唯讀掛載)
├── webui-data/            # WebUI 任務、log、私有 HF token(執行後產生,不進版控)
└── work/                  # 掛載到 /workspace/work
    ├── webui_runner.py    # WebUI 的通用 Unsloth 訓練 runner
    ├── patches/           # Unsloth compiled cache 自動修補(見下方說明)
    └── check_env.py       # 環境健檢腳本
```

## 前置需求

- NVIDIA driver 與 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) ≥ 1.19,
  且已產生 CDI 規格 `/etc/cdi/nvidia.yaml`(通常由 `nvidia-cdi-refresh.service` 自動維護;
  沒有的話執行 `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`)
- Docker ≥ 28(compose 使用 Docker 原生 CDI 掛載 GPU)
- 把自己加入 docker 群組(一次性):

```bash
sudo usermod -aG docker $USER
newgrp docker   # 或登出再登入
```

## 使用方式

```bash
cp .env.example .env       # 填入 HF_TOKEN(Gemma 是 gated model,必填)

docker compose build       # 首次建置(會拉官方映像檔,體積大、需要一段時間)
docker compose up -d       # 啟動
docker compose exec unsloth python /workspace/work/check_env.py   # 健檢

docker compose exec unsloth bash   # 進容器 shell
```

服務入口:

| 服務 | 位置 |
|---|---|
| LoRA Forge WebUI | http://localhost:6003 |
| JupyterLab | http://localhost:6001(密碼 = `.env` 的 `JUPYTER_PASSWORD`) |
| Unsloth Studio(選用) | http://localhost:6002 |
| SSH | port 6022 |

> **安全性**:WebUI 沒有登入驗證,而 compose 的 port 綁定是所有網路介面。
> 若機器不在信任的網路,請把 `docker-compose.yml` 的 `"6003:8080"` 改成
> `"127.0.0.1:6003:8080"`,改用 SSH port forward 存取;不要在沒有驗證與 TLS
> 的情況下直接公開到網際網路。

## LoRA Forge WebUI

開啟 <http://localhost:6003>。主要功能:

- FineTome、Complete FABLE 與 Gemma 4 三種開箱 preset。
- **載入過去設定**:從歷史任務清單一鍵帶入先前的完整訓練參數(不含 HF token)。
- 設定模型、資料格式、LoRA rank/alpha、context、batch 與訓練排程。
- 同一時間只執行一個 GPU 訓練任務;背景訓練、即時 log、取消與 checkpoint 續跑。
- 掃描 `work/outputs` 中已完成的 PEFT adapter。
- HF token 只寫入 `webui-data/hf_token`,不放進任務 JSON 或 log。

測試:

```bash
docker compose run --rm --no-deps --entrypoint /opt/venv/bin/pytest webui -q /workspace/tests
node --check webui/static/app.js
```

## GPU 掛載:Docker 原生 CDI

compose 以 Docker 原生 CDI(`driver: cdi` + `nvidia.com/gpu=all`)取代 `gpus: all`。
原因:傳統 nvidia runtime hook 是在 cgroup 之外直接改裝置權限,宿主機每次
`systemctl daemon-reload` 都可能收回長駐容器的 GPU 存取權(症狀:訓練突然報
`Can't initialize NVML`)。CDI 把裝置權限寫進 OCI 規格、systemd 可見,一勞永逸。
此設定只作用在本專案,不需要動 `/etc/nvidia-container-runtime/config.toml`
的全域模式,不影響機器上其他使用 `gpus: all` 的容器。

## work/patches — Unsloth 快取自動修補

unsloth_zoo 以 `inspect.getsource` 反編譯 transformers 模組產生
`work/unsloth_compiled_cache/`。transformers 5.14 的 `@force_accelerate_hooks`
裝飾器缺 `functools.wraps`,導致 Qwen3.5/3.6(GatedDeltaNet 層)的產生碼出現
`NameError: name 'args' is not defined`。

`webui_runner.py` 在每次訓練啟動時自動執行兩層防護:

1. `patches/sanitize_unsloth_cache.py`:掃描既有快取,修復壞掉的 forward stub
   (無法修復就刪掉讓 unsloth 重新產生)。
2. 依 `AutoConfig.model_type` 對正在使用的 transformers 模組還原被裝飾的
   forward,避免升級後重新產生出壞的快取。

升級 unsloth / unsloth_zoo / transformers 後不需要手動處理。若某次升級後訓練
log 不再出現「Unsloth 快取修補」字樣,代表上游已修復,可移除此 patch。
詳見 [work/patches/README.md](work/patches/README.md)。

## unsloth/ — vendored 原始碼

`unsloth/` 是 [unslothai/unsloth](https://github.com/unslothai/unsloth) 的完整原始碼
(vendored 自 upstream commit `bb80602`,2026-07-14,Apache-2.0 授權,LICENSE 隨附於
目錄內),已移除其 `.git`,由本專案的 git 直接管理。容器以 editable install 指向
這個目錄,修改原始碼後**下一個訓練任務**即生效(每個任務都是獨立 python 程序),
不需要重建映像檔;JupyterLab 中已載入 unsloth 的 kernel 需重啟才會生效。

> 注意:WebUI 的「Unsloth 版本」頁是 git-based(檢查更新、切 tag、rollback),
> 在 vendored 模式下會顯示不可用。要更新 unsloth 請手動同步 upstream,例如:
>
> ```bash
> git clone --depth 1 https://github.com/unslothai/unsloth.git /tmp/unsloth-new
> rsync -a --delete --exclude '.git' /tmp/unsloth-new/ unsloth/
> git add unsloth && git commit -m "unsloth: sync upstream <commit>"
> ```

## 注意事項

- 模型快取存在名為 `hf-cache` 的 docker volume,重建容器不會重新下載模型。
- `ipc: host` 是必要的,否則 DataLoader 多 worker 會因共享記憶體不足而失敗。
- `work/unsloth_compiled_cache/`、`work/outputs/`、`webui-data/`、`.env`
  皆不進版控(見 `.gitignore`)。

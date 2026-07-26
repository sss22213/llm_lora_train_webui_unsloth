# 基底:Unsloth 官方映像檔
# 已內含 CUDA 12.8+、PyTorch、bitsandbytes、xformers、triton,
# 並支援 Blackwell / RTX 5090(sm_120),另附 JupyterLab 與 Unsloth Studio
FROM unsloth/unsloth:latest

# 用本地 clone 的原始碼覆蓋映像檔內建的 unsloth(editable install)。
# docker-compose 會在執行期把 ./unsloth 掛載到同一路徑,
# 之後在宿主機 git pull 更新原始碼,重啟容器即生效,不需重建映像檔。
#
# 映像檔預設以非 root 使用者 unsloth:runtimeusers 執行,而 pip 的
# editable 安裝需要在原始碼目錄寫入 egg-info,所以先切到 root 安裝,
# 把所有權交還給 unsloth 後再切回去。
# 注意:pip 會把映像檔原本安裝的 unsloth 套件(含 studio)解除安裝,
# 但 entrypoint 寫死要用 site-packages/studio/frontend 建置 Studio 前端,
# 因此補一個 symlink 指回 clone 原始碼裡的 studio/。
# 另外 main 分支的 CLI 要求 $UNSLOTH_STUDIO_HOME/unsloth_studio 有獨立
# venv 才肯啟動 Studio;後端依賴其實都在 /opt/venv,symlink 過去即可。
#
# 整個容器直接以 root 執行(不切回映像檔預設的 unsloth 使用者),
# 一次解決原始碼目錄與掛載目錄的所有權限問題。
USER root
COPY unsloth /opt/unsloth-src
RUN pip install --no-deps -e /opt/unsloth-src \
 # main 分支需要比映像檔內建更新的整套訓練生態系。
 # transformers 必須分開安裝:unsloth_zoo 2026.7.3 宣告上限 <=5.5.0,
 # 但 Gemma 4 需要 >=5.6 才有 gemma4 模型類別;兩者一起裝會被 pip
 # 判定 ResolutionImpossible,分開裝 pip 只警告不會失敗,
 # 且此組合(zoo 2026.7.3 + transformers 5.14.1)已在容器內實測可正常
 # 載入並 patch Gemma 4。等官方放寬 zoo 上限後可改回合併安裝。
 # 已知副作用:映像檔內的 vllm 要求 transformers<5,升級後 vllm 不可用,
 # 但不影響 Unsloth 訓練路徑。
 && pip install --no-cache-dir "unsloth_zoo==2026.7.3" "trl==0.24.0" \
 && pip install --no-cache-dir "transformers==5.14.1" \
 && SITE=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
 && rm -rf "$SITE/studio" \
 && ln -sfn /opt/unsloth-src/studio "$SITE/studio" \
 && mkdir -p /workspace/.cache/huggingface /workspace/work /workspace/studio \
 && ln -sfn /opt/venv /workspace/studio/unsloth_studio

# 獨立的 LoRA Forge WebUI。與 Unsloth Studio 使用同一個映像與 Python
# 環境，但由 docker-compose 以不同 entrypoint 啟動。
COPY requirements-web.txt /tmp/requirements-web.txt
RUN pip install --no-cache-dir -r /tmp/requirements-web.txt

WORKDIR /workspace

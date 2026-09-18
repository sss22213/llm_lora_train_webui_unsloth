# 基底:Unsloth 官方映像檔
# 已內含 CUDA 12.8+、PyTorch、bitsandbytes、xformers、triton,
# 並支援 Blackwell / RTX 5090(sm_120),另附 JupyterLab 與 Unsloth Studio
# 釘住 digest：`docker compose build` 才不會因為上游 :latest 移動而悄悄換掉整套
# torch / transformers（2026-09-13 就是這樣把驗證過的環境換成 torch 2.11 nightly）。
# 要換版本時改這裡並重新驗證；`sudo docker images --digests unsloth/unsloth` 可查本機還有哪些。
#   2026-09-12 nightly（torch 2.11.0+cu128, /opt/unsloth-venv）：
ARG UNSLOTH_IMAGE=unsloth/unsloth@sha256:f2aeafc71364cce28cec99781ba15572ef2f46f6f8107604151c91d465a52642
FROM ${UNSLOTH_IMAGE}

# 訓練生態系（unsloth、unsloth_zoo、transformers、trl、torch 2.11）一律採用
# 映像檔內建、上游一起測過的組合，不再用 ./unsloth 的 editable install 覆蓋，
# 也不另外釘版本：nightly 基底已含 Qwen3.8 / Gemma 4 支援。要換整套版本就改上面的
# digest 重建。docker-compose 仍把 ./unsloth 掛到 /opt/unsloth-src 供「Unsloth
# 版本」頁面讀取；若之後要重新啟用原始碼覆蓋，把 ./unsloth 換成上游 git clone
# 並在此加回 `pip install --no-deps -e /opt/unsloth-src`。
#
# 整個容器直接以 root 執行（映像檔 2026-09 起預設即 root），訓練輸出與掛載目錄
# 不再有所有權問題。
USER root
# 官方映像檔的 venv 路徑曾從 /opt/venv 改為 /opt/unsloth-venv（2026-09 nightly）。
# 這裡統一以 /opt/venv 為準：若映像檔沒有 /opt/venv 就建 symlink 指向實際的
# venv，之後所有 pip/python 一律走 /opt/venv，docker-compose 的 entrypoint、
# README 與舊任務紀錄裡的 /opt/venv/bin/python 都不必跟著改。
RUN if [ ! -e /opt/venv ]; then \
      for candidate in /opt/unsloth-venv /opt/unsloth-studio/.venv; do \
        if [ -x "$candidate/bin/python" ]; then ln -s "$candidate" /opt/venv && break; fi; \
      done; \
    fi \
 && test -x /opt/venv/bin/python \
 && /opt/venv/bin/python -c "import sys; from importlib.metadata import version as v; \
      print('python', sys.version.split()[0], '|', ' | '.join(f'{n} {v(n)}' for n in \
      ('torch','transformers','trl','unsloth','unsloth_zoo','peft','bitsandbytes','xformers','triton')))"
ENV PATH=/opt/venv/bin:$PATH
RUN mkdir -p /workspace/.cache/huggingface /workspace/work

# 獨立的 LoRA Forge WebUI。與 Unsloth Studio 使用同一個映像與 Python
# 環境，但由 docker-compose 以不同 entrypoint 啟動。
COPY requirements-web.txt /tmp/requirements-web.txt
RUN pip install --no-cache-dir -r /tmp/requirements-web.txt

WORKDIR /workspace

"use strict";

const state = {
  thinkJobs: [], thinkDatasets: [], selectedThinkJobId: null, thinkLogOffset: 0, thinkLogTimer: null,
  jobs: [],
  adapters: [],
  system: null,
  health: null,
  selectedJobId: null,
  logOffset: 0,
  logTimer: null,
  unslothVersion: null,
  unslothTags: null,
};

const presets = {
  "qwen-finetome": {
    model: "unsloth/Qwen3.5-9B", model_family: "language", output_name: "qwen3.5-finetome",
    dataset: "mlabonne/FineTome-100k", dataset_config: "", dataset_split: "train",
    dataset_format: "sharegpt", source_filter: "", text_field: "text", max_samples: 3000,
    max_seq_length: 4096, load_in_4bit: true, lora_r: 16, lora_alpha: 16,
    lora_dropout: 0, batch_size: 2, gradient_accumulation_steps: 4,
    assistant_only_loss: true, packing: false, empty_think: "train", filter_overlength: true,
  },
  "qwen-fable": {
    model: "unsloth/Qwen3.5-9B", model_family: "language", output_name: "qwen3.5-fable-agent",
    dataset: "Crownelius/Complete-FABLE.5-traces-2M", dataset_config: "", dataset_split: "train",
    dataset_format: "fable_trace", source_filter: "greghavens/fable-5-coding-and-debugging-traces",
    text_field: "text", max_samples: 3000, max_seq_length: 8192, load_in_4bit: true,
    lora_r: 16, lora_alpha: 16, lora_dropout: 0, batch_size: 1,
    gradient_accumulation_steps: 8, assistant_only_loss: true, packing: false, empty_think: "train",
    filter_overlength: true,
  },
  "gemma-finetome": {
    model: "unsloth/gemma-4-12b-it", model_family: "multimodal", output_name: "gemma4-12b-finetome",
    dataset: "mlabonne/FineTome-100k", dataset_config: "", dataset_split: "train",
    dataset_format: "sharegpt", source_filter: "", text_field: "text", max_samples: 3000,
    max_seq_length: 2048, load_in_4bit: true, lora_r: 16, lora_alpha: 16,
    lora_dropout: 0, batch_size: 1, gradient_accumulation_steps: 8,
    assistant_only_loss: false, packing: false, empty_think: "train", filter_overlength: true,
  },
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (Array.isArray(body.detail)) message = body.detail.map((item) => item.msg).join("；");
      else if (body.detail) message = body.detail;
    } catch (_) { /* response is not JSON */ }
    throw new Error(message);
  }
  return response.json();
}

let toastTimer;
function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message;
  element.className = `toast show${error ? " error" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { element.className = "toast"; }, 3600);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
}

function formatDate(value) {
  if (!value) return "—";
  return new Intl.DateTimeFormat("zh-TW", { dateStyle: "short", timeStyle: "short" }).format(new Date(value));
}

function duration(job) {
  if (!job.started_at) return "尚未開始";
  const end = job.finished_at ? new Date(job.finished_at) : new Date();
  const seconds = Math.max(0, Math.floor((end - new Date(job.started_at)) / 1000));
  if (seconds < 60) return `${seconds} 秒`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒`;
  return `${Math.floor(seconds / 3600)} 小時 ${Math.floor((seconds % 3600) / 60)} 分`;
}

const statusNames = { queued: "排隊中", running: "訓練中", completed: "已完成", failed: "失敗", cancelled: "已取消" };

function showView(name) {
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === `${name}View`));
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === name));
  $("#sidebar").classList.remove("open");
  if (name === "jobs") loadJobs();
  if (name === "adapters") loadAdapters();
  if (name === "think") { loadThinkJobs(); loadThinkDatasets(); }
  if (name === "versions") loadUnslothVersion();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function setFormValue(name, value) {
  const input = document.querySelector(`[name="${name}"]`);
  if (!input) return;
  if (input.type === "checkbox") {
    input.checked = Boolean(value);
    return;
  }
  if (input.tagName === "SELECT" && ![...input.options].some((option) => option.value === String(value))) {
    const option = document.createElement("option");
    option.value = String(value);
    option.textContent = Number.isFinite(Number(value)) ? Number(value).toLocaleString("zh-TW") : String(value);
    input.append(option);
  }
  input.value = value;
}

function applyPreset(name) {
  const preset = presets[name];
  if (!preset) return;
  Object.entries(preset).forEach(([key, value]) => setFormValue(key, value));
  updateDatasetFields();
  updateSummary();
  showView("create");
  toast("已套用訓練 preset，可繼續調整參數");
}

function applyJobConfig(jobId) {
  const job = state.jobs.find((item) => item.id === jobId);
  if (!job?.request) {
    toast("找不到該任務的設定", true);
    return;
  }
  Object.entries(job.request).forEach(([key, value]) => {
    if (key === "hf_token") return; // token 不會被保存，也不應回填
    if (Array.isArray(value)) value = value.join(",");
    setFormValue(key, value ?? "");
  });
  updateDatasetFields();
  updateSummary();
  showView("create");
  const select = $("#historySelect");
  if (select) select.value = jobId;
  toast(`已載入「${job.request.output_name}」的訓練設定，可調整後重新送出`);
}

function renderHistoryOptions() {
  const loader = $("#historyLoader");
  const select = $("#historySelect");
  if (!state.jobs.length) {
    loader.hidden = true;
    return;
  }
  const previous = select.value;
  select.innerHTML = state.jobs.map((job) => {
    const model = (job.request.model || "").split("/").pop();
    const label = `${job.request.output_name} · ${model} · ${formatDate(job.created_at)} · ${statusNames[job.status]}`;
    return `<option value="${job.id}">${escapeHtml(label)}</option>`;
  }).join("");
  if (state.jobs.some((job) => job.id === previous)) select.value = previous;
  loader.hidden = false;
}

function updateDatasetFields() {
  const format = $("#datasetFormat").value;
  $("#sourceFilterField").style.display = format === "fable_trace" ? "grid" : "none";
  $("#textFieldWrap").style.display = format === "text" ? "grid" : "none";
  const messages = {
    sharegpt: "ShareGPT 會先標準化角色，再套用目前模型的 chat template。",
    messages: "保留 messages 與 tools 語意，再使用模型原生 chat template。",
    messages_json: "r0b0tlab trace：支援原生 messages/tools 欄位或 messages_json/tools_json 字串，清理後套用模型原生 chat template。多 config 資料集記得填「Config」欄位（如 sft_balanced）。",
    prompt_completion: "Prompt 只當上下文，TRL 僅對 completion tokens 計算 loss。",
    text: "純文字模式會對整段文字計算 language-modeling loss。",
    fable_trace: "解析 row_json、過濾指定來源，並只訓練每筆 trace 最後的 assistant 目標。",
  };
  $("#datasetNotice").textContent = messages[format];
}

function updateSummary() {
  const form = $("#trainingForm");
  const model = form.elements.model.value.split("/").pop() || "—";
  const dataset = form.elements.dataset.value.split("/").pop() || "—";
  const batch = Number(form.elements.batch_size.value || 0) * Number(form.elements.gradient_accumulation_steps.value || 0);
  const context = Number(form.elements.max_seq_length.value || 0);
  const maxSteps = Number(form.elements.max_steps.value || 0);
  $("#summaryModel").textContent = model;
  $("#summaryDataset").textContent = dataset;
  $("#summaryBatch").textContent = String(batch || "—");
  $("#summaryContext").textContent = context.toLocaleString("zh-TW");
  $("#summaryLength").textContent = maxSteps > 0 ? `${maxSteps.toLocaleString("zh-TW")} steps` : `${form.elements.num_train_epochs.value} epochs`;
  $("#summaryPrecision").textContent = form.elements.load_in_4bit.checked ? "4-bit QLoRA" : "LoRA";
}

function serializeForm(form) {
  const data = Object.fromEntries(new FormData(form).entries());
  const integers = ["max_samples", "max_seq_length", "lora_r", "lora_alpha", "batch_size", "gradient_accumulation_steps", "warmup_steps", "max_steps", "logging_steps", "save_steps", "seed"];
  const floats = ["lora_dropout", "learning_rate", "num_train_epochs", "weight_decay"];
  integers.forEach((key) => { data[key] = Number.parseInt(data[key], 10); });
  floats.forEach((key) => { data[key] = Number.parseFloat(data[key]); });
  ["load_in_4bit", "filter_overlength", "packing", "assistant_only_loss"].forEach((key) => { data[key] = form.elements[key].checked; });
  data.target_modules = data.target_modules.split(",").map((item) => item.trim()).filter(Boolean);
  ["hf_token", "dataset_config", "source_filter"].forEach((key) => { if (!data[key]) data[key] = null; });
  return data;
}

async function loadHealth() {
  try {
    state.health = await api("/api/health");
    const thinkActive = state.health.active_think_job;
    $("#healthDot").className = `status-dot ${state.health.active_job || thinkActive ? "busy" : "ok"}`;
    $("#healthText").textContent = state.health.active_job ? "GPU 任務執行中" : thinkActive ? "推理補完執行中" : "服務正常";
    $("#statService").textContent = "Online";
    $("#statRunner").textContent = state.health.runner_available ? "Runner ready" : "Runner missing";
    renderActiveJob();
  } catch (error) {
    $("#healthDot").className = "status-dot";
    $("#healthText").textContent = "服務異常";
    $("#statService").textContent = "Offline";
  }
}

async function loadSystem() {
  try {
    state.system = await api("/api/system");
    renderSystem();
  } catch (error) {
    $("#gpuCards").innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

function renderSystem() {
  const system = state.system;
  const gpu = system.gpus[0];
  $("#sidebarGpu").textContent = gpu ? gpu.name : "未偵測到 NVIDIA GPU";
  const versions = system.versions;
  $("#sidebarVersions").textContent = `Unsloth ${versions.unsloth || "—"} · TRL ${versions.trl || "—"}`;
  $("#statDisk").textContent = `可用 ${formatBytes(system.disk_free)} / ${formatBytes(system.disk_total)}`;
  if (!gpu) {
    $("#statGpuUtil").textContent = "—";
    $("#statGpuMemory").textContent = "nvidia-smi 無回應";
    $("#gpuCards").innerHTML = '<div class="empty-state">WebUI 容器內未偵測到 NVIDIA GPU。</div>';
    return;
  }
  $("#statGpuUtil").textContent = `${gpu.utilization}%`;
  $("#statGpuMemory").textContent = `${(gpu.memory_used_mb / 1024).toFixed(1)} / ${(gpu.memory_total_mb / 1024).toFixed(1)} GB VRAM`;
  $("#gpuCards").innerHTML = system.gpus.map((item) => {
    const memoryPercent = Math.round(item.memory_used_mb / item.memory_total_mb * 100);
    return `<article class="gpu-card">
      <div class="gpu-card-head"><strong>GPU ${item.index} · ${escapeHtml(item.name)}</strong><span>${item.temperature}°C</span></div>
      <div class="meter"><span style="width:${memoryPercent}%"></span></div>
      <div class="meter-label"><span>VRAM ${memoryPercent}%</span><span>${(item.memory_used_mb / 1024).toFixed(1)} / ${(item.memory_total_mb / 1024).toFixed(1)} GB</span></div>
      <div class="gpu-meta"><span>UTIL ${item.utilization}%</span><span>FREE ${(item.memory_free_mb / 1024).toFixed(1)} GB</span></div>
    </article>`;
  }).join("");
}

function renderActiveJob() {
  const job = state.health?.active_job;
  const thinkJob = state.health?.active_think_job;
  if (!job && thinkJob) {
    const progress = thinkJob.progress ? `${thinkJob.progress.done.toLocaleString("zh-TW")} / ${thinkJob.progress.total.toLocaleString("zh-TW")}` : "啟動中";
    $("#statActiveJob").textContent = `推理補完 · ${thinkJob.request.output_name}`;
    $("#activeJobCard").innerHTML = `<div class="active-job">
      <span class="status-badge ${thinkJob.status}">${thinkKindNames[thinkJob.kind] || thinkJob.kind}</span>
      <h3>${escapeHtml(thinkJob.request.output_name)}</h3>
      <p>${escapeHtml(thinkJob.request.ollama_model || "")}<br>${escapeHtml(progress)}</p>
      <button class="primary" id="openActiveThink">查看即時日誌</button>
    </div>`;
    $("#openActiveThink").addEventListener("click", () => { state.selectedThinkJobId = thinkJob.id; showView("think"); });
    return;
  }
  if (!job) {
    $("#activeJobCard").innerHTML = '<div class="empty-state">GPU 目前閒置，可以建立新任務。</div>';
    $("#statActiveJob").textContent = "目前閒置";
    return;
  }
  $("#statActiveJob").textContent = `${statusNames[job.status]} · ${job.request.output_name}`;
  $("#activeJobCard").innerHTML = `<div class="active-job">
    <span class="status-badge ${job.status}">${statusNames[job.status]}</span>
    <h3>${escapeHtml(job.request.output_name)}</h3>
    <p>${escapeHtml(job.request.model)}<br>${escapeHtml(job.request.dataset)}</p>
    <button class="primary" id="openActiveJob">查看即時日誌</button>
  </div>`;
  $("#openActiveJob").addEventListener("click", () => { state.selectedJobId = job.id; showView("jobs"); });
}

async function loadJobs(pollSelected = true) {
  try {
    state.jobs = await api("/api/jobs");
    $("#jobCount").textContent = state.jobs.length;
    $("#statJobs").textContent = state.jobs.length;
    renderJobs();
    renderHistoryOptions();
    if (state.selectedJobId) selectJob(state.selectedJobId, false, pollSelected);
  } catch (error) { toast(error.message, true); }
}

function renderJobs() {
  const list = $("#jobList");
  if (!state.jobs.length) {
    list.innerHTML = '<div class="empty-state">尚無任務。從「建立訓練」開始。</div>';
    return;
  }
  list.innerHTML = state.jobs.map((job) => `<button class="job-card ${job.id === state.selectedJobId ? "selected" : ""}" data-job-id="${job.id}">
    <div class="job-card-top"><strong>${escapeHtml(job.request.output_name)}</strong><span class="status-badge ${job.status}">${statusNames[job.status]}</span></div>
    <small>${escapeHtml(job.request.model)}</small>
    <div class="job-card-meta"><span>${formatDate(job.created_at)}</span><span>${duration(job)}</span></div>
  </button>`).join("");
  $$(".job-card").forEach((card) => card.addEventListener("click", () => selectJob(card.dataset.jobId)));
}

function selectJob(jobId, resetLog = true, startPolling = true) {
  const job = state.jobs.find((item) => item.id === jobId);
  if (!job) return;
  state.selectedJobId = jobId;
  renderJobs();
  $("#selectedStatus").className = `status-badge ${job.status}`;
  $("#selectedStatus").textContent = statusNames[job.status];
  $("#selectedTitle").textContent = job.request.output_name;
  $("#jobMeta").textContent = `${job.id} · ${duration(job)}${job.error ? ` · ${job.error}` : ""}`;
  $("#cancelButton").hidden = !["queued", "running"].includes(job.status);
  $("#retryButton").hidden = !["failed", "cancelled"].includes(job.status);
  $("#loadConfigButton").hidden = false;
  if (resetLog) {
    state.logOffset = 0;
    $("#console").textContent = "";
  }
  if (startPolling) pollLog();
}

async function pollLog() {
  clearTimeout(state.logTimer);
  if (!state.selectedJobId) return;
  try {
    const stickToBottom = $("#console").scrollTop + $("#console").clientHeight >= $("#console").scrollHeight - 32;
    const result = await api(`/api/jobs/${state.selectedJobId}/log?offset=${state.logOffset}`);
    if (result.content) $("#console").textContent += result.content;
    state.logOffset = result.offset;
    $("#logPosition").textContent = formatBytes(state.logOffset);
    if (stickToBottom) $("#console").scrollTop = $("#console").scrollHeight;
  } catch (error) { /* selected job may have been refreshed */ }
  const job = state.jobs.find((item) => item.id === state.selectedJobId);
  if (job && ["queued", "running"].includes(job.status)) state.logTimer = setTimeout(async () => { await loadJobs(false); pollLog(); }, 1800);
}

async function loadAdapters() {
  try {
    state.adapters = await api("/api/adapters");
    $("#adapterCount").textContent = state.adapters.length;
    $("#statAdapters").textContent = state.adapters.length;
    renderAdapters();
  } catch (error) { toast(error.message, true); }
}

function renderAdapters() {
  const library = $("#adapterLibrary");
  if (!state.adapters.length) {
    library.innerHTML = '<div class="empty-state">尚無已完成的 adapter。</div>';
    return;
  }
  library.innerHTML = state.adapters.map((adapter) => `<article class="adapter-card">
    <div class="adapter-card-head"><h2>${escapeHtml(adapter.name)}</h2><span class="status-badge completed">READY</span></div>
    <dl>
      <div><dt>Base model</dt><dd>${escapeHtml(adapter.base_model || "—")}</dd></div>
      <div><dt>LoRA rank</dt><dd>${escapeHtml(adapter.lora_r ?? "—")}</dd></div>
      <div><dt>大小</dt><dd>${formatBytes(adapter.size)}</dd></div>
      <div><dt>完成時間</dt><dd>${formatDate(adapter.modified_at)}</dd></div>
    </dl>
    <div class="path-box"><code title="${escapeHtml(adapter.path)}">${escapeHtml(adapter.path)}</code><button class="copy-button" data-copy="${escapeHtml(adapter.path)}">複製</button></div>
  </article>`).join("");
  $$(".copy-button").forEach((button) => button.addEventListener("click", async () => {
    await navigator.clipboard.writeText(button.dataset.copy);
    toast("已複製 adapter 路徑");
  }));
}

function gpuBusy() {
  return Boolean(state.health?.active_job || state.health?.active_think_job?.kind === "generate");
}

async function loadUnslothVersion(checkRemote = false) {
  try {
    state.unslothVersion = await api(`/api/unsloth/version${checkRemote ? "?check_remote=true" : ""}`);
    renderUnslothVersion();
    if (checkRemote) {
      toast(state.unslothVersion.update_available ? "偵測到新版本，可以更新" : "已是遠端最新版本");
    }
  } catch (error) {
    $("#unslothVersionNotice").textContent = `無法讀取版本資訊：${error.message}`;
    toast(error.message, true);
  }
}

function renderUnslothVersion() {
  const version = state.unslothVersion;
  if (!version) return;
  if (!version.available) {
    $("#unslothCommit").textContent = "—";
    $("#unslothSubject").textContent = "找不到 git 原始碼";
    $("#unslothVersionNotice").textContent = version.error || "Unsloth 原始碼不可用。";
    $("#updateVersionButton").disabled = true;
    $("#rollbackVersionButton").disabled = true;
    return;
  }
  const label = version.pinned_tag || version.branch || "detached";
  $("#unslothCommit").textContent = `${version.short_commit} · ${label}`;
  $("#unslothSubject").textContent = version.subject || "—";
  $("#unslothTracking").textContent = `${label} · ${version.tracking}`;
  const pkg = version.package_version ? `套件 ${version.package_version} · ` : "";
  $("#unslothCommitTime").textContent = `${pkg}${formatDate(version.committed_at)}`;

  if ("update_available" in version) {
    $("#unslothLatest").textContent = version.latest_short_commit;
    $("#unslothUpdateState").textContent = version.update_available ? "有新版本可更新" : "已是最新版本";
  }
  $("#updateVersionButton").disabled = !version.update_available || gpuBusy();
  $("#rollbackVersionButton").disabled = !version.rollback_available || gpuBusy();

  const dirtyBox = $("#unslothDirtyFiles");
  dirtyBox.hidden = !version.dirty;
  if (version.dirty) dirtyBox.textContent = `未提交的修改：\n${version.dirty_files.join("\n")}`;

  let notice;
  if (gpuBusy()) notice = "訓練任務執行中；為避免影響進行中的任務，暫時無法切換版本。";
  else if (version.dirty) notice = "原始碼有未提交的修改，切換版本前請先處理（可在容器或宿主機操作 git）。";
  else if (version.rollback_available) notice = `可退回上一個版本 ${version.previous_short_commit}（${version.previous_subject || "—"}）。`;
  else notice = "切換版本會立即改變掛載的原始碼；已完成訓練不受影響，下一個任務使用新版本。";
  $("#unslothVersionNotice").textContent = notice;
}

async function checkUnslothUpdate() {
  const button = $("#checkVersionButton");
  button.disabled = true;
  button.textContent = "檢查中...";
  try { await loadUnslothVersion(true); }
  finally { button.disabled = false; button.textContent = "檢查更新"; }
}

async function versionAction(path, payload, confirmMessage) {
  if (confirmMessage && !window.confirm(confirmMessage)) return;
  try {
    const result = await api(path, { method: "POST", body: JSON.stringify(payload) });
    state.unslothVersion = result;
    renderUnslothVersion();
    toast(result.message || "完成");
  } catch (error) { toast(error.message, true); }
}

function updateUnsloth() {
  versionAction(
    "/api/unsloth/version/update",
    { confirmation: "UPDATE" },
    "確定更新 Unsloth 至遠端最新版？下一個訓練任務將使用新版本。",
  );
}

function rollbackUnsloth() {
  versionAction(
    "/api/unsloth/version/rollback",
    { confirmation: "ROLLBACK" },
    "確定退回上一個 Unsloth 版本？",
  );
}

function pinUnsloth(ref) {
  const value = (ref || $("#unslothRefInput").value).trim();
  if (!value) { toast("請輸入 tag、branch 或 commit", true); return; }
  versionAction(
    "/api/unsloth/version/checkout",
    { confirmation: "CHECKOUT", ref: value },
    `確定切換 Unsloth 至「${value}」？`,
  );
}

async function loadUnslothTags() {
  const button = $("#loadTagsButton");
  button.disabled = true;
  button.textContent = "載入中...";
  try {
    const result = await api("/api/unsloth/version/tags");
    state.unslothTags = result.tags;
    renderUnslothTags();
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; button.textContent = "重新載入清單"; }
}

function renderUnslothTags() {
  const list = $("#unslothTagList");
  if (!state.unslothTags?.length) {
    list.innerHTML = '<div class="empty-state">遠端沒有 release tag。</div>';
    return;
  }
  const current = state.unslothVersion?.pinned_tag;
  list.innerHTML = state.unslothTags.map((tag) => `<div class="tag-row${tag.name === current ? " current" : ""}">
    <code>${escapeHtml(tag.name)}</code>
    <span>${escapeHtml(tag.commit)}</span>
    ${tag.name === current
      ? '<span class="status-badge completed">目前版本</span>'
      : `<button class="secondary tag-switch" data-ref="${escapeHtml(tag.name)}">切換</button>`}
  </div>`).join("");
  $$(".tag-switch").forEach((button) => button.addEventListener("click", () => pinUnsloth(button.dataset.ref)));
}

async function submitTraining(event) {
  event.preventDefault();
  const button = $("#submitButton");
  button.disabled = true;
  button.textContent = "建立任務中...";
  try {
    const job = await api("/api/jobs", { method: "POST", body: JSON.stringify(serializeForm(event.currentTarget)) });
    state.selectedJobId = job.id;
    toast("訓練任務已建立");
    await Promise.all([loadHealth(), loadJobs()]);
    showView("jobs");
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; button.textContent = "開始訓練"; }
}

async function cancelSelected() {
  if (!state.selectedJobId || !window.confirm("確定要取消目前訓練？已儲存的 checkpoint 會保留。")) return;
  try {
    await api(`/api/jobs/${state.selectedJobId}/cancel`, { method: "POST" });
    toast("已送出取消指令");
    await Promise.all([loadJobs(), loadHealth(), loadSystem()]);
  } catch (error) { toast(error.message, true); }
}

async function retrySelected() {
  if (!state.selectedJobId) return;
  try {
    await api(`/api/jobs/${state.selectedJobId}/retry`, { method: "POST" });
    state.logOffset = 0;
    $("#console").textContent = "";
    toast("任務已重新排入，將自動使用可用 checkpoint");
    await Promise.all([loadJobs(), loadHealth()]);
  } catch (error) { toast(error.message, true); }
}

// ---------------------------------------------------------------------------
// 推理補完：用底模（經 Ollama）替資料集補上 think 推理
// ---------------------------------------------------------------------------
const thinkStatusNames = { queued: "排隊中", running: "執行中", completed: "已完成", failed: "失敗", cancelled: "已取消" };
const thinkKindNames = { generate: "產生推理", build: "組出 train.jsonl" };

function serializeThinkForm(form) {
  const data = Object.fromEntries(new FormData(form).entries());
  ["samples", "parallel", "seed", "num_ctx", "num_predict", "retry_on_length", "min_think_chars", "max_think_chars", "min_content_chars"].forEach((key) => { data[key] = Number.parseInt(data[key], 10); });
  return data;
}

async function checkOllama() {
  const url = $("#thinkForm").elements.ollama_url.value.trim();
  const backend = $("#thinkForm").elements.api.value;
  const apiKey = $("#thinkForm").elements.api_key.value.trim();
  const button = $("#ollamaCheckButton");
  button.disabled = true;
  button.textContent = "連線中...";
  try {
    const result = await api(`/api/think/ollama?url=${encodeURIComponent(url)}&api=${encodeURIComponent(backend)}&api_key=${encodeURIComponent(apiKey)}`);
    $("#ollamaModels").innerHTML = result.models.map((model) => `<option value="${escapeHtml(model.name)}"></option>`).join("");
    $("#ollamaStatus").textContent = `已連線 ${result.url}（${backend === "openai" ? "OpenAI 相容 /v1" : "Ollama"}），${result.models.length} 個模型可從下拉選單挑選`;
    toast(`連線成功，${result.models.length} 個模型`);
  } catch (error) {
    $("#ollamaStatus").textContent = error.message;
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "測試連線";
  }
}

async function submitThinkGenerate(event) {
  event.preventDefault();
  const button = $("#thinkSubmitButton");
  button.disabled = true;
  button.textContent = "建立任務中...";
  try {
    const job = await api("/api/think/generate", { method: "POST", body: JSON.stringify(serializeThinkForm(event.currentTarget)) });
    state.selectedThinkJobId = job.id;
    toast("推理補完任務已建立，日誌每 10 筆更新一次");
    await Promise.all([loadHealth(), loadThinkJobs()]);
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; button.textContent = "開始產生"; }
}

async function buildThinkDataset(name) {
  const form = $("#thinkForm");
  const payload = {
    output_name: name,
    min_think_chars: Number.parseInt(form.elements.min_think_chars.value, 10),
    max_think_chars: Number.parseInt(form.elements.max_think_chars.value, 10),
    min_content_chars: Number.parseInt(form.elements.min_content_chars.value, 10),
    only_with_think: $("#thinkOnlyWithThink").checked,
  };
  try {
    const job = await api("/api/think/build", { method: "POST", body: JSON.stringify(payload) });
    state.selectedThinkJobId = job.id;
    toast("開始組 train.jsonl");
    await loadThinkJobs();
  } catch (error) { toast(error.message, true); }
}

function useThinkDataset(path) {
  setFormValue("dataset", path);
  setFormValue("dataset_config", "");
  setFormValue("dataset_split", "train");
  setFormValue("dataset_format", "messages");
  setFormValue("max_samples", 0);
  setFormValue("empty_think", "mask");
  updateDatasetFields();
  updateSummary();
  showView("create");
  toast("已帶入資料集：messages 格式、空 think 遮罩不計 loss，其餘參數請自行調整");
}

async function loadThinkJobs(pollSelected = true) {
  try {
    state.thinkJobs = await api("/api/think/jobs");
    renderThinkJobs();
    renderThinkHistoryOptions();
    if (state.selectedThinkJobId) selectThinkJob(state.selectedThinkJobId, false, pollSelected);
  } catch (error) { toast(error.message, true); }
}

function setThinkFormValue(name, value) {
  const input = $("#thinkForm").elements[name];
  if (!input) return;
  if (input.type === "checkbox") { input.checked = Boolean(value); return; }
  input.value = value ?? "";
}

function applyThinkJobConfig(jobId) {
  const job = state.thinkJobs.find((item) => item.id === jobId);
  if (!job?.request) { toast("找不到該任務的設定", true); return; }
  if (job.kind === "generate") {
    Object.entries(job.request).forEach(([key, value]) => setThinkFormValue(key, value));
  } else {
    setThinkFormValue("output_name", job.request.output_name);
    setThinkFormValue("min_think_chars", job.request.min_think_chars);
    setThinkFormValue("max_think_chars", job.request.max_think_chars);
    setThinkFormValue("min_content_chars", job.request.min_content_chars);
    $("#thinkOnlyWithThink").checked = Boolean(job.request.only_with_think);
  }
  const select = $("#thinkHistorySelect");
  if (select && state.thinkJobs.some((item) => item.id === jobId && item.kind === "generate")) select.value = jobId;
  $("#thinkForm").scrollIntoView({ behavior: "smooth", block: "start" });
  toast(`已載入「${thinkKindNames[job.kind] || job.kind} · ${job.request.output_name}」的設定，可調整後重新送出`);
}

function renderThinkHistoryOptions() {
  const loader = $("#thinkHistoryLoader");
  const select = $("#thinkHistorySelect");
  const generateJobs = state.thinkJobs.filter((job) => job.kind === "generate");
  if (!generateJobs.length) { loader.hidden = true; return; }
  const previous = select.value;
  select.innerHTML = generateJobs.map((job) => {
    const label = `${job.request.output_name} · ${job.request.mode} · ${Number(job.request.samples).toLocaleString("zh-TW")} 筆 · ${formatDate(job.created_at)} · ${thinkStatusNames[job.status]}`;
    return `<option value="${job.id}">${escapeHtml(label)}</option>`;
  }).join("");
  if (generateJobs.some((job) => job.id === previous)) select.value = previous;
  loader.hidden = false;
}

function thinkProgressHtml(job) {
  const progress = job.progress;
  if (!progress || !progress.total) return "";
  const percent = Math.min(100, Math.round(progress.done / progress.total * 100));
  return `<div class="meter"><span style="width:${percent}%"></span></div><small>${progress.done.toLocaleString("zh-TW")} / ${progress.total.toLocaleString("zh-TW")}（${percent}%）</small>`;
}

function renderThinkJobs() {
  const list = $("#thinkJobList");
  if (!state.thinkJobs.length) {
    list.innerHTML = '<div class="empty-state">尚無推理補完任務。</div>';
    return;
  }
  list.innerHTML = state.thinkJobs.map((job) => {
    const detail = job.kind === "generate"
      ? `${job.request.mode} · ${job.request.ollama_model} · ${Number(job.request.samples).toLocaleString("zh-TW")} 筆`
      : `推理 ${job.request.min_think_chars} 到 ${job.request.max_think_chars || "∞"} 字${job.request.only_with_think ? " · 只含推理" : ""}`;
    return `<button class="job-card ${job.id === state.selectedThinkJobId ? "selected" : ""}" data-think-id="${job.id}">
      <div class="job-card-top"><strong>${escapeHtml(thinkKindNames[job.kind] || job.kind)} · ${escapeHtml(job.request.output_name)}</strong><span class="status-badge ${job.status}">${thinkStatusNames[job.status]}</span></div>
      <small>${escapeHtml(detail)}</small>
      ${thinkProgressHtml(job)}
      <div class="job-card-meta"><span>${formatDate(job.created_at)}</span><span>${duration(job)}</span></div>
    </button>`;
  }).join("");
  $$("[data-think-id]").forEach((card) => card.addEventListener("click", () => selectThinkJob(card.dataset.thinkId)));
}

function selectThinkJob(jobId, resetLog = true, startPolling = true) {
  const job = state.thinkJobs.find((item) => item.id === jobId);
  if (!job) return;
  state.selectedThinkJobId = jobId;
  renderThinkJobs();
  $("#thinkStatus").className = `status-badge ${job.status}`;
  $("#thinkStatus").textContent = thinkStatusNames[job.status];
  $("#thinkTitle").textContent = `${thinkKindNames[job.kind] || job.kind} · ${job.request.output_name}`;
  $("#thinkMeta").textContent = `${job.id} · ${duration(job)}${job.error ? ` · ${job.error}` : ""}`;
  $("#thinkCancelButton").hidden = !["queued", "running"].includes(job.status);
  $("#thinkLoadConfigButton").hidden = false;
  if (resetLog) {
    state.thinkLogOffset = 0;
    $("#thinkConsole").textContent = "";
  }
  if (startPolling) pollThinkLog();
}

async function pollThinkLog() {
  clearTimeout(state.thinkLogTimer);
  if (!state.selectedThinkJobId) return;
  const consoleElement = $("#thinkConsole");
  try {
    const stickToBottom = consoleElement.scrollTop + consoleElement.clientHeight >= consoleElement.scrollHeight - 32;
    const result = await api(`/api/think/jobs/${state.selectedThinkJobId}/log?offset=${state.thinkLogOffset}`);
    if (result.content) consoleElement.textContent += result.content;
    state.thinkLogOffset = result.offset;
    $("#thinkLogPosition").textContent = formatBytes(state.thinkLogOffset);
    if (stickToBottom) consoleElement.scrollTop = consoleElement.scrollHeight;
  } catch (error) { /* job list may have been refreshed */ }
  const job = state.thinkJobs.find((item) => item.id === state.selectedThinkJobId);
  if (job && ["queued", "running"].includes(job.status)) {
    state.thinkLogTimer = setTimeout(async () => { await loadThinkJobs(false); pollThinkLog(); }, 2500);
  } else {
    loadThinkDatasets();
  }
}

async function cancelThinkJob() {
  if (!state.selectedThinkJobId || !window.confirm("確定要取消？已完成的樣本會保留，之後用同一個輸出名稱可以續跑。")) return;
  try {
    await api(`/api/think/jobs/${state.selectedThinkJobId}/cancel`, { method: "POST" });
    toast("已送出取消指令");
    await Promise.all([loadThinkJobs(), loadHealth()]);
  } catch (error) { toast(error.message, true); }
}

async function loadThinkDatasets() {
  try {
    state.thinkDatasets = await api("/api/think/datasets");
    renderThinkDatasets();
  } catch (error) { toast(error.message, true); }
}

function renderThinkDatasets() {
  const box = $("#thinkDatasets");
  if (!state.thinkDatasets.length) {
    box.innerHTML = '<div class="empty-state">尚無資料集，先在上方產生推理。</div>';
    return;
  }
  box.innerHTML = state.thinkDatasets.map((item) => {
    const stats = item.build_stats || {};
    const built = Number.isFinite(item.train_rows);
    const withThink = Number.isFinite(stats.with_think) ? stats.with_think.toLocaleString("zh-TW") : "—";
    const source = item.meta?.dataset || "";
    return `<article class="adapter-card">
      <div class="adapter-card-head"><h2>${escapeHtml(item.name)}</h2><span class="status-badge ${built ? "completed" : "queued"}">${built ? "READY" : "未組"}</span></div>
      <dl>
        <div><dt>來源</dt><dd title="${escapeHtml(source)}">${escapeHtml(source.split("/").pop() || "—")}</dd></div>
        <div><dt>模式</dt><dd>${escapeHtml(item.meta?.mode || "—")} · ${escapeHtml(item.meta?.turn || "—")}</dd></div>
        <div><dt>已生成</dt><dd>${Number(item.generated_rows || 0).toLocaleString("zh-TW")} 筆</dd></div>
        <div><dt>train.jsonl</dt><dd>${built ? `${item.train_rows.toLocaleString("zh-TW")} 筆，含推理 ${withThink}` : "尚未組出"}</dd></div>
        <div><dt>更新</dt><dd>${formatDate(item.train_at || item.generated_at)}</dd></div>
      </dl>
      <div class="path-box"><code title="${escapeHtml(item.path)}">${escapeHtml(item.path)}</code><button class="copy-button copy-think-path" data-copy="${escapeHtml(item.path)}">複製</button></div>
      <div class="card-actions"><button class="secondary" data-build="${escapeHtml(item.name)}">組出 train.jsonl</button><button class="primary" data-use="${escapeHtml(item.path)}" ${built ? "" : "disabled"}>用它建立訓練</button></div>
    </article>`;
  }).join("");
  $$(".copy-think-path").forEach((button) => button.addEventListener("click", async () => {
    await navigator.clipboard.writeText(button.dataset.copy);
    toast("已複製資料集路徑");
  }));
  $$("[data-build]").forEach((button) => button.addEventListener("click", () => buildThinkDataset(button.dataset.build)));
  $$("[data-use]").forEach((button) => button.addEventListener("click", () => useThinkDataset(button.dataset.use)));
}

function bindEvents() {
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));
  $$("[data-preset]").forEach((button) => button.addEventListener("click", () => applyPreset(button.dataset.preset)));
  $$(".jump-create").forEach((button) => button.addEventListener("click", () => showView("create")));
  $("#mobileMenu").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  $("#datasetFormat").addEventListener("change", updateDatasetFields);
  $("#trainingForm").addEventListener("input", updateSummary);
  $("#trainingForm").addEventListener("submit", submitTraining);
  $("#refreshSystem").addEventListener("click", loadSystem);
  $("#refreshJobs").addEventListener("click", loadJobs);
  $("#refreshAdapters").addEventListener("click", loadAdapters);
  $("#cancelButton").addEventListener("click", cancelSelected);
  $("#retryButton").addEventListener("click", retrySelected);
  $("#loadConfigButton").addEventListener("click", () => { if (state.selectedJobId) applyJobConfig(state.selectedJobId); });
  $("#historyApplyButton").addEventListener("click", () => { const id = $("#historySelect").value; if (id) applyJobConfig(id); });
  $("#refreshVersionButton").addEventListener("click", () => loadUnslothVersion());
  $("#checkVersionButton").addEventListener("click", checkUnslothUpdate);
  $("#updateVersionButton").addEventListener("click", updateUnsloth);
  $("#rollbackVersionButton").addEventListener("click", rollbackUnsloth);
  $("#pinVersionButton").addEventListener("click", () => pinUnsloth());
  $("#unslothRefInput").addEventListener("keydown", (event) => { if (event.key === "Enter") pinUnsloth(); });
  $("#loadTagsButton").addEventListener("click", loadUnslothTags);
  $("#refreshThink").addEventListener("click", () => { loadThinkJobs(); loadThinkDatasets(); });
  $("#thinkForm").addEventListener("submit", submitThinkGenerate);
  $("#ollamaCheckButton").addEventListener("click", checkOllama);
  $("#thinkCancelButton").addEventListener("click", cancelThinkJob);
  $("#thinkLoadConfigButton").addEventListener("click", () => { if (state.selectedThinkJobId) applyThinkJobConfig(state.selectedThinkJobId); });
  $("#thinkHistoryApplyButton").addEventListener("click", () => { const id = $("#thinkHistorySelect").value; if (id) applyThinkJobConfig(id); });
}

async function initialize() {
  bindEvents();
  updateDatasetFields();
  updateSummary();
  await Promise.all([loadHealth(), loadSystem(), loadJobs(), loadAdapters()]);
  setInterval(async () => { await Promise.all([loadHealth(), loadSystem()]); }, 8000);
}

document.addEventListener("DOMContentLoaded", initialize);

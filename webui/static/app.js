"use strict";

const state = {
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
    assistant_only_loss: true, packing: false, filter_overlength: true,
  },
  "qwen-fable": {
    model: "unsloth/Qwen3.5-9B", model_family: "language", output_name: "qwen3.5-fable-agent",
    dataset: "Crownelius/Complete-FABLE.5-traces-2M", dataset_config: "", dataset_split: "train",
    dataset_format: "fable_trace", source_filter: "greghavens/fable-5-coding-and-debugging-traces",
    text_field: "text", max_samples: 3000, max_seq_length: 8192, load_in_4bit: true,
    lora_r: 16, lora_alpha: 16, lora_dropout: 0, batch_size: 1,
    gradient_accumulation_steps: 8, assistant_only_loss: true, packing: false,
    filter_overlength: true,
  },
  "gemma-finetome": {
    model: "unsloth/gemma-4-12b-it", model_family: "multimodal", output_name: "gemma4-12b-finetome",
    dataset: "mlabonne/FineTome-100k", dataset_config: "", dataset_split: "train",
    dataset_format: "sharegpt", source_filter: "", text_field: "text", max_samples: 3000,
    max_seq_length: 2048, load_in_4bit: true, lora_r: 16, lora_alpha: 16,
    lora_dropout: 0, batch_size: 1, gradient_accumulation_steps: 8,
    assistant_only_loss: false, packing: false, filter_overlength: true,
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
    $("#healthDot").className = `status-dot ${state.health.active_job ? "busy" : "ok"}`;
    $("#healthText").textContent = state.health.active_job ? "GPU 任務執行中" : "服務正常";
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
  return Boolean(state.health?.active_job);
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
}

async function initialize() {
  bindEvents();
  updateDatasetFields();
  updateSummary();
  await Promise.all([loadHealth(), loadSystem(), loadJobs(), loadAdapters()]);
  setInterval(async () => { await Promise.all([loadHealth(), loadSystem()]); }, 8000);
}

document.addEventListener("DOMContentLoaded", initialize);

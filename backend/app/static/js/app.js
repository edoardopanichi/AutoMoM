const state = {
  currentJobId: null,
  eventSource: null,
  latestJobState: null,
  modelDownloads: {},
  modelDownloadPoller: null,
  profileRefreshPoller: null,
  speakerFormFingerprint: null,
  localModelCatalog: { defaults: {}, models: [] },
  diarizationModels: [],
  transcriptionModels: [],
  formatterModels: [],
  jobs: [],
};

const LAST_JOB_STORAGE_KEY = "automom:lastJobId";

const LOCAL_RUNTIME_OPTIONS = {
  diarization: ["pyannote"],
  transcription: ["whisper.cpp", "faster-whisper"],
  formatter: ["ollama", "command"],
};

const LOCAL_MODEL_FIELD_CONFIG = {
  "pyannote": {
    primaryKey: "pipeline_path",
    primaryLabel: "Pipeline path",
    primaryPlaceholder: "/abs/path/to/config.yaml",
    secondaryKey: "embedding_model_ref",
    secondaryLabel: "Embedding model ref",
    secondaryPlaceholder: "pyannote/wespeaker-voxceleb-resnet34-LM",
    help: "Register an existing local pyannote pipeline config and its embedding reference.",
  },
  "whisper.cpp": {
    primaryKey: "binary_path",
    primaryLabel: "Binary path",
    primaryPlaceholder: "/abs/path/to/whisper-cli",
    secondaryKey: "model_path",
    secondaryLabel: "Model path",
    secondaryPlaceholder: "/abs/path/to/model.gguf",
    help: "Register an existing whisper.cpp binary and GGUF model file.",
  },
  "faster-whisper": {
    primaryKey: "model_path",
    primaryLabel: "Model directory",
    primaryPlaceholder: "/abs/path/to/ctranslate2-model",
    secondaryKey: "compute_type",
    secondaryLabel: "Compute type",
    secondaryPlaceholder: "auto | float16 | int8",
    help: "Register an existing faster-whisper model directory. The package must already be installed.",
  },
  "ollama": {
    primaryKey: "tag",
    primaryLabel: "Ollama tag",
    primaryPlaceholder: "phi4-mini",
    secondaryKey: "",
    secondaryLabel: "Secondary setting",
    secondaryPlaceholder: "",
    help: "Register an already-pulled Ollama model tag.",
  },
  "command": {
    primaryKey: "command_template",
    primaryLabel: "Command template",
    primaryPlaceholder: "bash -lc \"... {model} ...\"",
    secondaryKey: "model_path",
    secondaryLabel: "Model path",
    secondaryPlaceholder: "/abs/path/to/model.gguf",
    help: "Register a command-based formatter with an existing local model file.",
  },
};

/**
 * @brief Query the first matching DOM element.
 * @param {*} selector CSS selector used to query the DOM.
 */
const qs = (selector) => document.querySelector(selector);
/**
 * @brief Query all matching DOM elements as an array.
 * @param {*} selector CSS selector used to query the DOM.
 */
const qsa = (selector) => Array.from(document.querySelectorAll(selector));

/**
 * @brief Switch Tab.
 * @param {*} tabId Identifier of the tab to activate.
 */
function switchTab(tabId) {
  document.body.dataset.activeTab = tabId;
  qsa(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabId);
  });
  qsa(".panel").forEach((panel) => panel.classList.toggle("active", panel.id === tabId));
}

/**
 * @brief Reset Job Ui.
 */
function resetJobUi() {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  state.currentJobId = null;
  state.latestJobState = null;
  state.speakerFormFingerprint = null;
  localStorage.removeItem(LAST_JOB_STORAGE_KEY);
  qs("#audio-file").value = "";
  qs("#meeting-title").value = "";
  qs("#start-error").textContent = "";
  qs("#speaker-count").textContent = "Detected speakers: -";
  qs("#speaker-form").innerHTML = "";
  qs("#speaker-naming-section").classList.add("hidden");
  qs("#mom-preview").textContent = "";
  qs("#download-link").removeAttribute("href");
  qs("#overall-value").textContent = "0.0%";
  qs("#stage-value").textContent = "-";
  qs("#stage-percent").textContent = "0.0%";
  qs("#segment-progress").textContent = "-";
  qs("#logs").textContent = "";
  setProgressBars(0, 0);
  switchTab("new-job");
}

/**
 * @brief Fetch JSON.
 * @param {*} url Request URL.
 * @param {*} options Fetch options for the request.
 */
async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

/**
 * @brief Load Jobs.
 */
async function loadJobs() {
  const payload = await fetchJSON("/api/jobs");
  state.jobs = payload.jobs || [];
  renderJobList();
  return state.jobs;
}

/**
 * @brief Render Job List.
 */
function renderJobList() {
  const list = qs("#job-list");
  if (!list) return;
  list.innerHTML = "";
  if (!state.jobs.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No jobs yet.";
    list.appendChild(empty);
    return;
  }

  state.jobs.forEach((job) => {
    const card = document.createElement("article");
    card.className = "job-card";

    const header = document.createElement("div");
    header.className = "job-card-header";
    const title = document.createElement("h3");
    title.className = "job-card-title";
    title.textContent = job.job_id;
    const status = document.createElement("span");
    status.className = "job-status";
    status.textContent = job.status;
    header.append(title, status);

    const meta = document.createElement("div");
    meta.className = "job-card-meta";
    const stage = document.createElement("span");
    stage.textContent = job.current_stage || "No active stage";
    const updated = document.createElement("span");
    updated.textContent = `Updated ${formatDateTime(job.updated_at)}`;
    const progress = document.createElement("span");
    progress.textContent = `${Number(job.overall_percent || 0).toFixed(1)}%`;
    meta.append(stage, updated, progress);

    const actions = document.createElement("div");
    actions.className = "job-card-actions";
    const openBtn = document.createElement("button");
    openBtn.type = "button";
    openBtn.className = "small-btn";
    openBtn.textContent = job.status === "completed" ? "Open Result" : "Open Job";
    openBtn.addEventListener("click", () => reopenJob(job.job_id));
    actions.appendChild(openBtn);

    if (["completed", "failed", "cancelled"].includes(job.status)) {
      const deleteBtn = document.createElement("button");
      deleteBtn.type = "button";
      deleteBtn.className = "small-btn danger-btn";
      deleteBtn.textContent = "Delete";
      deleteBtn.addEventListener("click", () => deleteJob(job.job_id));
      actions.appendChild(deleteBtn);
    }

    card.append(header, meta, actions);
    list.appendChild(card);
  });
}

/**
 * @brief Format Date Time.
 * @param {*} value Date value received from API.
 */
function formatDateTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

/**
 * @brief Load Startup Data.
 */
async function loadStartupData() {
  const [templates, models, profiles, downloads, localModelCatalog] = await Promise.all([
    fetchJSON("/api/templates"),
    fetchJSON("/api/models"),
    fetchJSON("/api/profiles"),
    fetchJSON("/api/models/downloads"),
    fetchJSON("/api/models/local"),
  ]);
  state.localModelCatalog = localModelCatalog;
  state.diarizationModels = (localModelCatalog.models || []).filter((item) => item.stage === "diarization");
  state.transcriptionModels = (localModelCatalog.models || []).filter((item) => item.stage === "transcription");
  state.formatterModels = (localModelCatalog.models || []).filter((item) => item.stage === "formatter");
  state.modelDownloads = Object.fromEntries(downloads.map((item) => [item.model_id, item]));
  renderDiarizationModels();
  renderTranscriptionModels();
  renderFormatterModels();
  syncLocalModelForm();
  renderTemplateSelect(templates);
  renderExecutionModelLabels();
  renderTemplates(templates);
  renderModels(models);
  renderProfiles(profiles);
  startModelDownloadPolling();
}

/**
 * @brief Render Diarization Models.
 * @param {*} payload Payload consumed by the renderer.
 */
function renderDiarizationModels() {
  const select = qs("#local-diarization-model");
  if (!select) return;
  select.innerHTML = "";
  state.diarizationModels.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;
    option.disabled = !model.installed;
    select.appendChild(option);
  });
  select.value = state.localModelCatalog.defaults.diarization || state.diarizationModels[0]?.model_id || "";
}

/**
 * @brief Render Transcription Models.
 */
function renderTranscriptionModels() {
  const select = qs("#local-transcription-model");
  if (!select) return;
  select.innerHTML = "";
  state.transcriptionModels.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.installed ? `${model.name} (${model.runtime})` : `${model.name} (${model.runtime}, unavailable)`;
    option.disabled = !model.installed;
    select.appendChild(option);
  });
  select.value = state.localModelCatalog.defaults.transcription || state.transcriptionModels[0]?.model_id || "";
}

/**
 * @brief Render Formatter Models.
 */
function renderFormatterModels() {
  const select = qs("#local-formatter-model");
  if (!select) return;
  select.innerHTML = "";
  state.formatterModels.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.installed ? `${model.name} (${model.runtime})` : `${model.name} (${model.runtime}, unavailable)`;
    option.disabled = !model.installed;
    select.appendChild(option);
  });
  select.value = state.localModelCatalog.defaults.formatter || state.formatterModels[0]?.model_id || "";
}

/**
 * @brief Render Template Select.
 * @param {*} templates Template records returned by the API.
 */
function renderTemplateSelect(templates) {
  const select = qs("#template-select");
  select.innerHTML = "";
  templates.forEach((template) => {
    const option = document.createElement("option");
    option.value = template.template_id;
    option.textContent = `${template.name} (${template.version})`;
    select.appendChild(option);
  });
}

/**
 * @brief Render Execution Model Labels.
 */
function renderExecutionModelLabels() {
  renderDiarizationModels();
  renderTranscriptionModels();
  renderFormatterModels();
}

/**
 * @brief Reset Template Creator.
 */
function resetTemplateCreator() {
  qs("#new-template-id").value = "";
  qs("#new-template-name").value = "";
  qs("#new-template-version").value = "1.0.0";
  qs("#new-template-description").value = "";
  qs("#new-template-prompt-block").value = "";
}

/**
 * @brief Toggle Template Creator.
 * @param {*} show Whether the template creator should be visible.
 */
function toggleTemplateCreator(show) {
  const panel = qs("#template-creator");
  panel.classList.toggle("hidden", !show);
  if (!show) {
    resetTemplateCreator();
  }
}

/**
 * @brief Set Execution Mode.
 * @param {*} stage Pipeline stage identifier.
 * @param {*} mode Execution mode for the stage.
 */
function setExecutionMode(stage, mode) {
  const hiddenInput = qs(`#${stage}-execution`);
  const card = qsa(".engine-card").find((item) => item.dataset.stage === stage);
  if (!hiddenInput || !card) return;

  hiddenInput.value = mode;
  qsa(`.engine-option[data-stage="${stage}"]`).forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });

  qsa(`.engine-card[data-stage="${stage}"] .engine-model-field`).forEach((field) => {
    const isActive = field.dataset.mode === mode;
    field.classList.toggle("hidden", !isActive);
    const select = field.querySelector("select");
    if (select) {
      // Disable the inactive selector as well as hiding it so only the chosen execution path
      // contributes values when the form is submitted.
      select.disabled = !isActive;
    }
  });

  card.dataset.mode = mode;
}

/**
 * @brief Synchronize Cloud Execution Controls.
 */
function syncCloudExecutionControls() {
  ["diarization", "transcription", "formatter"].forEach((stage) => {
    const input = qs(`#${stage}-execution`);
    if (!input) return;
    setExecutionMode(stage, input.value || "local");
  });

  const apiSelected = ["diarization", "transcription", "formatter"].some(
    (stage) => qs(`#${stage}-execution`)?.value === "api",
  );
  qs("#api-key-panel").classList.toggle("hidden", !apiSelected);
}

/**
 * @brief Render Templates.
 * @param {*} templates Template records returned by the API.
 */
function renderTemplates(templates) {
  const container = qs("#templates");
  container.innerHTML = "";

  if (!templates.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No templates available.";
    container.appendChild(empty);
    return;
  }

  templates.forEach((template) => {
    const card = document.createElement("article");
    card.className = "template-card";

    const eyebrow = document.createElement("p");
    eyebrow.className = "card-eyebrow";
    eyebrow.textContent = "Template";

    const title = document.createElement("div");
    title.className = "card-title";

    const name = document.createElement("strong");
    name.textContent = template.name;

    const id = document.createElement("small");
    id.textContent = template.template_id;

    title.append(name, id);

    const meta = document.createElement("div");
    meta.className = "card-meta";
    meta.textContent = `Version ${template.version}${template.description ? ` | ${template.description}` : ""}`;

    card.append(eyebrow, title, meta);
    container.appendChild(card);
  });
}

/**
 * @brief Render Models.
 * @param {*} models Model records returned by the API.
 */
function renderModels(models) {
  const container = qs("#models");
  container.innerHTML = "";

  const localModels = state.localModelCatalog.models || [];
  if (localModels.length) {
    localModels.forEach((model) => {
      const card = document.createElement("article");
      card.className = "model-card";
      card.dataset.modelId = model.model_id;
      card.dataset.installed = String(Boolean(model.installed));
      card.dataset.downloadStatus = model.installed ? "completed" : "idle";

      const eyebrow = document.createElement("p");
      eyebrow.className = "card-eyebrow";
      eyebrow.textContent = `Local ${model.stage}`;

      const title = document.createElement("div");
      title.className = "card-title";
      const strong = document.createElement("strong");
      strong.textContent = model.name;
      const small = document.createElement("small");
      small.textContent = `[${model.runtime}]`;
      title.append(strong, small);

      const info = document.createElement("div");
      info.className = "card-meta";
      const defaultForStage = state.localModelCatalog.defaults[model.stage] === model.model_id ? " | Default" : "";
      info.textContent = `Installed: ${model.installed ? "yes" : "no"} | Stage: ${model.stage}${defaultForStage}`;

      const details = document.createElement("div");
      details.className = "card-meta";
      const configSummary = Object.entries(model.config || {})
        .map(([key, value]) => `${key}=${value}`)
        .join(" | ");
      details.textContent = model.validation_error
        ? `Validation: ${model.validation_error}`
        : configSummary || "Ready";

      const actionRow = document.createElement("div");
      actionRow.className = "model-action-row";

      const defaultBtn = document.createElement("button");
      defaultBtn.className = "download-btn";
      defaultBtn.textContent = "Set default";
      defaultBtn.disabled = !model.installed || state.localModelCatalog.defaults[model.stage] === model.model_id;
      defaultBtn.addEventListener("click", async () => {
        try {
          await fetchJSON("/api/models/local/defaults", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ stage: model.stage, model_id: model.model_id }),
          });
          await refreshSettings();
        } catch (error) {
          alert(`Unable to set default: ${error.message}`);
        }
      });
      actionRow.append(defaultBtn);

      const deleteBtn = document.createElement("button");
      deleteBtn.className = "download-btn";
      deleteBtn.textContent = "Delete";
      deleteBtn.disabled = state.localModelCatalog.defaults[model.stage] === model.model_id;
      deleteBtn.addEventListener("click", async () => {
        try {
          await fetchJSON(`/api/models/local/${encodeURIComponent(model.model_id)}`, { method: "DELETE" });
          await refreshSettings();
        } catch (error) {
          alert(`Unable to delete model: ${error.message}`);
        }
      });
      actionRow.append(deleteBtn);

      card.append(eyebrow, title, info, details, actionRow);
      container.appendChild(card);
    });
  }

  models.forEach((model) => {
    const downloadState = state.modelDownloads[model.model_id] || {
      status: model.installed ? "completed" : "idle",
      percent: model.installed ? 100 : 0,
      downloaded_bytes: 0,
      total_bytes: null,
      error: null,
    };

    const card = document.createElement("article");
    card.className = "model-card";
    card.dataset.modelId = model.model_id;
    card.dataset.installed = String(Boolean(model.installed));
    card.dataset.downloadStatus = downloadState.status || "idle";

    const eyebrow = document.createElement("p");
    eyebrow.className = "card-eyebrow";
    eyebrow.textContent = model.model_id === "formatter" ? "Formatter Runtime" : "Model Asset";

    const title = document.createElement("div");
    title.className = "card-title";
    const strong = document.createElement("strong");
    strong.textContent = model.name;
    const small = document.createElement("small");
    small.textContent = `[${model.model_id}]`;
    title.append(strong, small);

    const info = document.createElement("div");
    info.className = "card-meta";
    info.textContent = `Installed: ${model.installed ? "yes" : "no"} | Size: ${model.size_mb} MB | Source: ${model.source}`;

    const actionRow = document.createElement("div");
    actionRow.className = "model-action-row";

    const download = document.createElement("button");
    download.className = "download-btn";
    download.textContent = downloadState.status === "running" ? "Downloading..." : "Download";
    download.disabled = model.installed || downloadState.status === "running";
    download.addEventListener("click", async () => {
      try {
        await fetchJSON("/api/models/download", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_id: model.model_id }),
        });
        await refreshModelDownloads();
        startModelDownloadPolling();
      } catch (error) {
        alert(`Download failed: ${error.message}`);
      }
      await refreshSettings();
    });
    actionRow.append(download);

    const downloadStatus = document.createElement("div");
    downloadStatus.className = "model-download-status";
    downloadStatus.textContent = buildDownloadStatusText(downloadState);

    const progressWrap = document.createElement("div");
    progressWrap.className = "download-progress-wrap";
    const progressBar = document.createElement("div");
    progressBar.className = "download-progress-bar";
    progressBar.style.width = `${Math.max(0, Math.min(100, Number(downloadState.percent || 0))).toFixed(1)}%`;
    progressWrap.appendChild(progressBar);

    card.append(eyebrow, title, info, actionRow, downloadStatus, progressWrap);
    container.appendChild(card);
  });
}

/**
 * @brief Render Profiles.
 * @param {*} profiles Voice profile records returned by the API.
 */
function renderProfiles(profiles) {
  const container = qs("#profiles");
  container.innerHTML = "";
  if (!profiles.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No voice profiles saved.";
    container.appendChild(empty);
    return;
  }
  profiles.forEach((profile) => {
    const card = document.createElement("article");
    card.className = "profile-card";
    const modelSummary = Array.from(
      new Set(
        (profile.samples || []).flatMap((sample) => (sample.embeddings || []).map((item) => item.diarization_model_id)),
      ),
    );
    const eyebrow = document.createElement("p");
    eyebrow.className = "card-eyebrow";
    eyebrow.textContent = "Voice Profile";

    const title = document.createElement("div");
    title.className = "card-title";

    const name = document.createElement("strong");
    name.textContent = profile.name;

    const count = document.createElement("small");
    count.textContent = `${profile.sample_count} sample${profile.sample_count === 1 ? "" : "s"}`;

    title.append(name, count);

    const meta = document.createElement("div");
    meta.className = "card-meta";
    meta.textContent = modelSummary.length ? `Embeddings: ${modelSummary.join(", ")}` : "No embeddings stored yet.";

    card.append(eyebrow, title, meta);
    container.appendChild(card);
  });
}

/**
 * @brief Refresh Profiles For Selected Model.
 */
async function refreshProfilesForSelectedModel() {
  const status = qs("#refresh-profiles-status");
  const payload = {
    diarization_execution: qs("#diarization-execution").value,
    local_diarization_model_id: qs("#local-diarization-model").value,
    openai_diarization_model: qs("#openai-diarization-model").value,
  };
  status.textContent = "Starting profile refresh...";
  try {
    const task = await fetchJSON("/api/profiles/rebuild", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    pollProfileRefreshTask(task.task_id);
  } catch (error) {
    status.textContent = error.message;
  }
}

/**
 * @brief Poll Profile Refresh Task.
 * @param {*} taskId Identifier of the background task.
 */
async function pollProfileRefreshTask(taskId) {
  clearInterval(state.profileRefreshPoller);
  const status = qs("#refresh-profiles-status");
  const tick = async () => {
    try {
      const task = await fetchJSON(`/api/profiles/rebuild/${taskId}`);
      const total = task.total_samples || 0;
      status.textContent = `${task.status}: ${task.message || ""}${total ? ` (${task.processed_samples}/${total})` : ""}`.trim();
      if (task.status === "completed" || task.status === "failed") {
        clearInterval(state.profileRefreshPoller);
        state.profileRefreshPoller = null;
        await refreshSettings();
      }
    } catch (error) {
      clearInterval(state.profileRefreshPoller);
      state.profileRefreshPoller = null;
      status.textContent = error.message;
    }
  };
  await tick();
  state.profileRefreshPoller = setInterval(tick, 1500);
}

/**
 * @brief Refresh Settings.
 */
async function refreshSettings() {
  const [templates, models, profiles, downloads, localModelCatalog] = await Promise.all([
    fetchJSON("/api/templates"),
    fetchJSON("/api/models"),
    fetchJSON("/api/profiles"),
    fetchJSON("/api/models/downloads"),
    fetchJSON("/api/models/local"),
  ]);
  state.localModelCatalog = localModelCatalog;
  state.diarizationModels = (localModelCatalog.models || []).filter((item) => item.stage === "diarization");
  state.transcriptionModels = (localModelCatalog.models || []).filter((item) => item.stage === "transcription");
  state.formatterModels = (localModelCatalog.models || []).filter((item) => item.stage === "formatter");
  state.modelDownloads = Object.fromEntries(downloads.map((item) => [item.model_id, item]));
  renderDiarizationModels();
  renderTranscriptionModels();
  renderFormatterModels();
  syncLocalModelForm();
  renderTemplateSelect(templates);
  renderExecutionModelLabels();
  renderTemplates(templates);
  renderModels(models);
  renderProfiles(profiles);
}

/**
 * @brief Synchronize Local Model Form.
 */
function syncLocalModelForm() {
  const stageSelect = qs("#local-model-stage");
  const runtimeSelect = qs("#local-model-runtime");
  const primaryLabel = qs("#local-model-primary-label");
  const secondaryLabel = qs("#local-model-secondary-label");
  const primaryInput = qs("#local-model-primary");
  const secondaryInput = qs("#local-model-secondary");
  const help = qs("#local-model-help");
  if (!stageSelect || !runtimeSelect) return;

  const stage = stageSelect.value || "transcription";
  const runtimes = LOCAL_RUNTIME_OPTIONS[stage] || [];
  const currentRuntime = runtimeSelect.value;
  runtimeSelect.innerHTML = "";
  runtimes.forEach((runtime) => {
    const option = document.createElement("option");
    option.value = runtime;
    option.textContent = runtime;
    runtimeSelect.appendChild(option);
  });
  runtimeSelect.value = runtimes.includes(currentRuntime) ? currentRuntime : runtimes[0];

  const fieldConfig = LOCAL_MODEL_FIELD_CONFIG[runtimeSelect.value];
  primaryLabel.textContent = fieldConfig.primaryLabel;
  secondaryLabel.textContent = fieldConfig.secondaryLabel || "Secondary setting";
  primaryInput.placeholder = fieldConfig.primaryPlaceholder || "";
  secondaryInput.placeholder = fieldConfig.secondaryPlaceholder || "";
  secondaryInput.disabled = !fieldConfig.secondaryKey;
  help.textContent = fieldConfig.help;
}

/**
 * @brief Register Local Model.
 */
async function registerLocalModel() {
  const stage = qs("#local-model-stage").value;
  const runtime = qs("#local-model-runtime").value;
  const name = qs("#local-model-name").value.trim();
  const primary = qs("#local-model-primary").value.trim();
  const secondary = qs("#local-model-secondary").value.trim();
  const languages = qs("#local-model-languages").value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  const notes = qs("#local-model-notes").value.trim();
  const status = qs("#local-model-status");
  const fieldConfig = LOCAL_MODEL_FIELD_CONFIG[runtime];
  if (!name || !primary) {
    status.textContent = "Name and the primary setting are required.";
    return;
  }

  const payload = {
    stage,
    runtime,
    name,
    languages,
    notes,
    set_as_default: qs("#local-model-default").checked,
    config: {
      [fieldConfig.primaryKey]: primary,
    },
  };
  if (fieldConfig.secondaryKey && secondary) {
    payload.config[fieldConfig.secondaryKey] = secondary;
  }

  status.textContent = "Registering...";
  try {
    await fetchJSON("/api/models/local", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    qs("#local-model-name").value = "";
    qs("#local-model-primary").value = "";
    qs("#local-model-secondary").value = "";
    qs("#local-model-languages").value = "";
    qs("#local-model-notes").value = "";
    status.textContent = "Model registered.";
    await refreshSettings();
  } catch (error) {
    status.textContent = error.message;
  }
}

/**
 * @brief Format Bytes.
 * @param {*} value Numeric value to format.
 */
function formatBytes(value) {
  if (value === null || value === undefined) return "? B";
  const num = Number(value);
  if (!Number.isFinite(num)) return "? B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = num;
  let idx = 0;
  while (size >= 1024 && idx < units.length - 1) {
    size /= 1024;
    idx += 1;
  }
  return `${size.toFixed(idx === 0 ? 0 : 1)} ${units[idx]}`;
}

/**
 * @brief Build Download Status Text.
 * @param {*} downloadState Download state snapshot for a model.
 */
function buildDownloadStatusText(downloadState) {
  const status = downloadState.status || "idle";
  if (status === "running") {
    const percent = Number(downloadState.percent || 0).toFixed(1);
    const left = formatBytes(downloadState.downloaded_bytes);
    const right = formatBytes(downloadState.total_bytes);
    return `Downloading ${percent}% (${left} / ${right})`;
  }
  if (status === "completed") {
    return "Download completed";
  }
  if (status === "failed") {
    return `Download failed: ${downloadState.error || "unknown error"}`;
  }
  return "Not downloaded";
}

/**
 * @brief Refresh Model Downloads.
 */
async function refreshModelDownloads() {
  const downloads = await fetchJSON("/api/models/downloads");
  state.modelDownloads = Object.fromEntries(downloads.map((item) => [item.model_id, item]));
}

/**
 * @brief Start Model Download Polling.
 */
function startModelDownloadPolling() {
  if (state.modelDownloadPoller) {
    clearInterval(state.modelDownloadPoller);
    state.modelDownloadPoller = null;
  }

  const hasRunning = Object.values(state.modelDownloads).some((item) => item.status === "running");
  if (!hasRunning) {
    return;
  }

  state.modelDownloadPoller = setInterval(async () => {
    try {
      await refreshSettings();
      const stillRunning = Object.values(state.modelDownloads).some((item) => item.status === "running");
      if (!stillRunning && state.modelDownloadPoller) {
        clearInterval(state.modelDownloadPoller);
        state.modelDownloadPoller = null;
      }
    } catch (error) {
      console.error("Model download polling failed", error);
    }
  }, 1000);
}

/**
 * @brief Set Progress Bars.
 * @param {*} overall Overall completion percentage.
 * @param {*} stagePercent Current stage completion percentage.
 */
function setProgressBars(overall, stagePercent) {
  qs("#overall-bar").style.width = `${overall}%`;
  qs("#stage-bar").style.width = `${stagePercent}%`;
}

/**
 * @brief Update Progress View.
 * @param {*} jobState Job state payload from the API.
 */
function updateProgressView(jobState) {
  qs("#overall-value").textContent = `${jobState.overall_percent.toFixed(1)}%`;
  qs("#stage-value").textContent = jobState.current_stage || "-";
  qs("#stage-percent").textContent = `${jobState.stage_percent.toFixed(1)}%`;
  qs("#segment-progress").textContent = jobState.transcript_segment_progress || jobState.stage_detail || "-";
  setProgressBars(jobState.overall_percent, jobState.stage_percent);
  qs("#logs").textContent = (jobState.logs || []).join("\n");
  qs("#speaker-naming-section").classList.toggle("hidden", jobState.status !== "waiting_speaker_input");
}

/**
 * @brief Build Form Fingerprint.
 * @param {*} jobState Job state payload from the API.
 */
function speakerFormFingerprint(jobState) {
  const speakerInfo = jobState.speaker_info;
  if (!speakerInfo) {
    return null;
  }
  const speakers = speakerInfo.speakers.map((speaker) => ({
    speaker_id: speaker.speaker_id,
    suggested_name: speaker.suggested_name || "",
    matched_profile: speaker.matched_profile || null,
    snippets: (speaker.snippets || []).map((snippet) => snippet.snippet_path),
  }));
  return JSON.stringify({
    job_id: jobState.job_id,
    detected_speakers: speakerInfo.detected_speakers,
    speakers,
  });
}

/**
 * @brief Render Speaker Form.
 * @param {*} jobState Job state payload from the API.
 */
function renderSpeakerForm(jobState) {
  const speakerInfo = jobState.speaker_info;
  if (!speakerInfo) {
    return;
  }

  const existingNames = new Map(
    qsa(".speaker-name").map((input) => [input.dataset.speakerId, input.value]),
  );
  const existingSaveProfile = new Map(
    qsa(".speaker-save-profile").map((input) => [input.dataset.speakerId, input.checked]),
  );

  qs("#speaker-count").textContent = `Detected speakers: ${speakerInfo.detected_speakers}`;
  const form = qs("#speaker-form");
  form.innerHTML = "";

  speakerInfo.speakers.forEach((speaker) => {
    const card = document.createElement("div");
    card.className = "speaker-card";

    const header = document.createElement("div");
    header.className = "speaker-card-header";

    const title = document.createElement("h4");
    title.textContent = speaker.speaker_id;
    header.appendChild(title);

    if (speaker.matched_profile) {
      const badge = document.createElement("span");
      badge.className = `speaker-match-badge ${speaker.matched_profile.status}`;
      if (speaker.matched_profile.status === "matched") {
        badge.textContent = `Auto-recognized: ${speaker.matched_profile.name} (${speaker.matched_profile.score.toFixed(2)})`;
      } else {
        const alternatives = (speaker.matched_profile.ambiguous_names || []).join(", ");
        badge.textContent = alternatives
          ? `Possible match: ${speaker.matched_profile.name} (${speaker.matched_profile.score.toFixed(2)}), also ${alternatives}`
          : `Possible match: ${speaker.matched_profile.name} (${speaker.matched_profile.score.toFixed(2)})`;
      }
      header.appendChild(badge);
    }

    const input = document.createElement("input");
    input.type = "text";
    input.value =
      existingNames.get(speaker.speaker_id) ||
      speaker.suggested_name ||
      speaker.speaker_id;
    input.dataset.speakerId = speaker.speaker_id;
    input.className = "speaker-name";

    const toggleWrap = document.createElement("label");
    toggleWrap.className = "speaker-toggle";
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.dataset.speakerId = speaker.speaker_id;
    toggle.className = "speaker-save-profile";
    toggle.checked = Boolean(existingSaveProfile.get(speaker.speaker_id));
    toggleWrap.append(toggle, speaker.matched_profile ? "Save or update shared voice profile" : "Save as voice profile");

    const snippets = document.createElement("div");
    snippets.className = "speaker-snippets";
    speaker.snippets.forEach((snippet) => {
      const audio = document.createElement("audio");
      audio.controls = true;
      const snippetName = String(snippet.snippet_path || "")
        .replace(/\\/g, "/")
        .split("/")
        .pop();
      if (!snippetName) {
        return;
      }
      audio.src = `/api/jobs/${jobState.job_id}/snippets/${encodeURIComponent(snippetName)}`;
      snippets.appendChild(audio);
    });

    const hint = document.createElement("div");
    hint.className = "speaker-card-hint";
    hint.textContent = speaker.matched_profile
      ? "Matched profiles prefill the name field. You can keep it, correct it, or update the shared profile."
      : "Review the proposed snippets, assign a name, and save a profile only if this sample is representative.";

    card.append(header, input, toggleWrap, hint, snippets);
    form.appendChild(card);
  });
}

/**
 * @brief Submit Speaker Mapping.
 */
async function submitSpeakerMapping() {
  if (!state.currentJobId) return;
  const mappings = qsa(".speaker-name").map((input) => {
    const speakerId = input.dataset.speakerId;
    const save = qsa(".speaker-save-profile").find((item) => item.dataset.speakerId === speakerId);
    return {
      speaker_id: speakerId,
      name: input.value.trim() || speakerId,
      save_voice_profile: Boolean(save && save.checked),
    };
  });

  const grouped = new Map();
  mappings.forEach((item) => {
    const key = item.name.trim().toLowerCase();
    if (!key) return;
    const existing = grouped.get(key) || [];
    existing.push(item);
    grouped.set(key, existing);
  });
  const duplicates = Array.from(grouped.values()).filter((items) => items.length > 1);
  if (duplicates.length) {
    const duplicateNames = duplicates.map((items) => items[0].name).join(", ");
    const confirmed = window.confirm(
      `You assigned the same speaker name to multiple diarized speakers: ${duplicateNames}. This will merge them into one speaker in the final transcript and update a shared voice profile. Continue?`,
    );
    if (!confirmed) {
      return;
    }
  }

  await fetchJSON(`/api/jobs/${state.currentJobId}/speaker-mapping`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mappings }),
  });
  switchTab("progress");
}

/**
 * @brief Load Result.
 * @param {*} jobId Identifier of the active job.
 */
async function loadResult(jobId) {
  qs("#download-link").href = `/api/jobs/${jobId}/download/mom`;
  try {
    let response = await fetch(`/api/jobs/${jobId}/mom`, { cache: "no-store" });
    if (!response.ok) {
      response = await fetch(`/api/jobs/${jobId}/download/mom`, { cache: "no-store" });
    }
    if (!response.ok) throw new Error(`Unable to fetch markdown preview (${response.status})`);
    const text = await response.text();
    qs("#mom-preview").textContent = text;
  } catch (error) {
    qs("#mom-preview").textContent = "Unable to load markdown preview.";
  }
}

/**
 * @brief Apply Job State.
 * @param {*} jobState Job state payload from the API.
 */
async function applyJobState(jobState, { navigate = false } = {}) {
  state.currentJobId = jobState.job_id;
  state.latestJobState = jobState;
  localStorage.setItem(LAST_JOB_STORAGE_KEY, jobState.job_id);
  updateProgressView(jobState);

  if (jobState.status === "waiting_speaker_input") {
    const nextFingerprint = speakerFormFingerprint(jobState);
    if (nextFingerprint !== state.speakerFormFingerprint) {
      renderSpeakerForm(jobState);
      state.speakerFormFingerprint = nextFingerprint;
    }
    switchTab("progress");
    return;
  }

  state.speakerFormFingerprint = null;
  if (jobState.status === "completed") {
    await loadResult(jobState.job_id);
    switchTab("result");
    return;
  }
  if (navigate) {
    switchTab("progress");
  }
}

/**
 * @brief Reopen Job.
 * @param {*} jobId Identifier of the job to reopen.
 */
async function reopenJob(jobId) {
  try {
    const jobState = await fetchJSON(`/api/jobs/${jobId}`);
    await applyJobState(jobState, { navigate: true });
    if (["created", "running", "waiting_speaker_input"].includes(jobState.status)) {
      openJobEventStream(jobId);
    }
  } catch (error) {
    alert(`Unable to open job: ${error.message}`);
  }
}

/**
 * @brief Delete Job.
 * @param {*} jobId Identifier of the job to delete.
 */
async function deleteJob(jobId) {
  const confirmed = window.confirm(`Delete ${jobId} and its local artifacts?`);
  if (!confirmed) return;
  try {
    await fetchJSON(`/api/jobs/${jobId}`, { method: "DELETE" });
    if (state.currentJobId === jobId) {
      resetJobUi();
    }
    await loadJobs();
  } catch (error) {
    alert(`Unable to delete job: ${error.message}`);
  }
}

/**
 * @brief Restore Last Job.
 */
async function restoreLastJob() {
  const jobId = localStorage.getItem(LAST_JOB_STORAGE_KEY);
  if (!jobId) return;
  try {
    await reopenJob(jobId);
  } catch (error) {
    localStorage.removeItem(LAST_JOB_STORAGE_KEY);
  }
}

/**
 * @brief Open Job Event Stream.
 * @param {*} jobId Identifier of the active job.
 */
function openJobEventStream(jobId) {
  if (state.eventSource) {
    state.eventSource.close();
  }
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  state.eventSource = source;

  source.onmessage = async (event) => {
    const jobState = JSON.parse(event.data);
    await applyJobState(jobState);

    if (jobState.status === "completed") {
      source.close();
      state.eventSource = null;
      await loadJobs();
      return;
    }

    if (jobState.status === "failed" || jobState.status === "cancelled") {
      alert(`Job ${jobState.status}: ${jobState.error || "No details"}`);
      source.close();
      state.eventSource = null;
      await loadJobs();
      if (jobState.status === "cancelled") {
        resetJobUi();
      }
    }
  };

  source.onerror = async () => {
    source.close();
    state.eventSource = null;
    if (!state.currentJobId) return;
    try {
      const jobState = await fetchJSON(`/api/jobs/${state.currentJobId}`);
      await applyJobState(jobState);
      await loadJobs();
    } catch (error) {
      localStorage.removeItem(LAST_JOB_STORAGE_KEY);
    }
  };
}

/**
 * @brief Start Job.
 * @param {*} event Browser event object.
 */
async function startJob(event) {
  event.preventDefault();
  qs("#start-error").textContent = "";

  const fileInput = qs("#audio-file");
  if (!fileInput.files.length) {
    qs("#start-error").textContent = "Please select an audio file.";
    return;
  }

  const formData = new FormData();
  formData.append("audio_file", fileInput.files[0]);
  formData.append("template_id", qs("#template-select").value);
  formData.append("title", qs("#meeting-title").value);
  formData.append("diarization_execution", qs("#diarization-execution").value);
  formData.append("transcription_execution", qs("#transcription-execution").value);
  formData.append("formatter_execution", qs("#formatter-execution").value);
  formData.append("local_diarization_model_id", qs("#local-diarization-model").value);
  formData.append("local_transcription_model_id", qs("#local-transcription-model").value);
  formData.append("local_formatter_model_id", qs("#local-formatter-model").value);
  if (
    ["#diarization-execution", "#transcription-execution", "#formatter-execution"].some(
      (selector) => qs(selector).value === "api",
    )
  ) {
    formData.append("openai_api_key", qs("#openai-api-key").value);
  }
  formData.append("openai_diarization_model", qs("#openai-diarization-model").value);
  formData.append("openai_transcription_model", qs("#openai-transcription-model").value);
  formData.append("openai_formatter_model", qs("#openai-formatter-model").value);

  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `HTTP ${response.status}`);
    }
    const job = await response.json();
    state.currentJobId = job.job_id;
    localStorage.setItem(LAST_JOB_STORAGE_KEY, job.job_id);
    switchTab("progress");
    openJobEventStream(job.job_id);
    await loadJobs();
  } catch (error) {
    qs("#start-error").textContent = error.message;
  }
}

/**
 * @brief Cancel Job.
 */
async function cancelJob() {
  if (!state.currentJobId) return;
  await fetchJSON(`/api/jobs/${state.currentJobId}/cancel`, { method: "POST" });
  resetJobUi();
  await loadJobs();
}

/**
 * @brief Save Template Inline.
 */
async function saveTemplateInline() {
  const templateId = qs("#new-template-id").value.trim();
  const name = qs("#new-template-name").value.trim();
  const version = qs("#new-template-version").value.trim() || "1.0.0";
  const description = qs("#new-template-description").value.trim();
  const promptBlock = qs("#new-template-prompt-block").value.trim();

  if (!templateId || !name || !promptBlock) {
    alert("Template ID, Name, and Prompt Block are required.");
    return;
  }

  try {
    await fetchJSON("/api/templates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        template_id: templateId,
        name,
        version,
        description,
        prompt_block: promptBlock,
      }),
    });
    await refreshSettings();
    qs("#template-select").value = templateId;
    toggleTemplateCreator(false);
  } catch (error) {
    alert(`Unable to save template: ${error.message}`);
  }
}

/**
 * @brief Bind Events.
 */
function bindEvents() {
  qsa(".tab").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });
  qs("#job-form").addEventListener("submit", startJob);
  qs("#cancel-job").addEventListener("click", cancelJob);
  qs("#refresh-jobs").addEventListener("click", loadJobs);
  qs("#refresh-profiles-btn").addEventListener("click", refreshProfilesForSelectedModel);
  qs("#submit-speakers").addEventListener("click", submitSpeakerMapping);
  qs("#open-template-creator").addEventListener("click", () => toggleTemplateCreator(true));
  qs("#cancel-template-inline").addEventListener("click", () => toggleTemplateCreator(false));
  qs("#save-template-inline").addEventListener("click", saveTemplateInline);
  qs("#local-model-stage").addEventListener("change", syncLocalModelForm);
  qs("#local-model-runtime").addEventListener("change", syncLocalModelForm);
  qs("#register-local-model").addEventListener("click", registerLocalModel);
  qsa(".engine-option").forEach((button) => {
    button.addEventListener("click", () => {
      setExecutionMode(button.dataset.stage, button.dataset.mode);
      syncCloudExecutionControls();
    });
  });
}

/**
 * @brief Initialize operation.
 */
async function init() {
  bindEvents();
  syncCloudExecutionControls();
  await loadStartupData();
  await loadJobs();
  await restoreLastJob();
}

init();

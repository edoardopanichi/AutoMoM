const state = {
  currentJobId: null,
  eventSource: null,
  latestJobState: null,
  modelDownloads: {},
  modelDownloadPoller: null,
  profileRefreshPoller: null,
  speakerFormFingerprint: null,
  localModelCatalog: { defaults: {}, models: [] },
  localRuntimeDescriptors: [],
  localModelInstallPoller: null,
  diarizationModels: [],
  transcriptionModels: [],
  formatterModels: [],
  jobs: [],
};

const LAST_JOB_STORAGE_KEY = "automom:lastJobId";
const DEFAULT_TEMPLATE_SECTIONS = [
  { heading: "### Title:", required: true, allow_prefix: true, empty_value: "None" },
  { heading: "#### Participants:", required: true, allow_prefix: false, empty_value: "None" },
  { heading: "#### Concise Overview:", required: true, allow_prefix: false, empty_value: "None" },
  { heading: "#### TODO's:", required: true, allow_prefix: false, empty_value: "None" },
  { heading: "#### CONCLUSIONS:", required: true, allow_prefix: false, empty_value: "None" },
  { heading: "#### DECISION/OPEN POINTS:", required: true, allow_prefix: false, empty_value: "None" },
  { heading: "#### RISKS:", required: true, allow_prefix: false, empty_value: "None" },
];

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
  qs("#full-transcript-preview").textContent = "";
  qs("#transcript-download-link").removeAttribute("href");
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
  const [templates, models, profiles, downloads, localModelCatalog, localRuntimeDescriptors] = await Promise.all([
    fetchJSON("/api/templates"),
    fetchJSON("/api/models"),
    fetchJSON("/api/profiles"),
    fetchJSON("/api/models/downloads"),
    fetchJSON("/api/models/local"),
    fetchJSON("/api/models/local/runtimes"),
  ]);
  state.localModelCatalog = localModelCatalog;
  state.localRuntimeDescriptors = localRuntimeDescriptors || [];
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
 * @brief Format Formatter Model Label.
 * @param {*} model Local formatter model record.
 */
function formatFormatterModelLabel(model) {
  if (model.runtime === "ollama") {
    return `${model.config?.tag || model.name} via Ollama`;
  }
  if (model.runtime === "command") {
    return `${model.config?.model_path || model.name} via command`;
  }
  return `${model.name} (${model.runtime})`;
}

/**
 * @brief Format Local Model Details.
 * @param {*} model Local model record.
 */
function formatLocalModelDetails(model) {
  if (model.validation_error) {
    return `Validation: ${model.validation_error}`;
  }
  if (model.stage === "formatter" && model.runtime === "ollama") {
    return `Model: ${model.config?.tag || "-"} | Runtime: Ollama`;
  }
  if (model.stage === "formatter" && model.runtime === "command") {
    return `Model path: ${model.config?.model_path || "-"} | Runtime: command`;
  }
  return Object.entries(model.config || {})
    .map(([key, value]) => `${key}=${value}`)
    .join(" | ") || "Ready";
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
    option.textContent = model.installed
      ? formatFormatterModelLabel(model)
      : `${formatFormatterModelLabel(model)} - unavailable`;
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
  const currentValue = select.value;
  const defaultTemplate = templates.find((template) => template.is_default) || templates[0];
  select.innerHTML = "";
  templates.forEach((template) => {
    const option = document.createElement("option");
    option.value = template.template_id;
    option.textContent = `${template.name} (${template.version})${template.is_default ? " - default" : ""}`;
    select.appendChild(option);
  });
  select.value = templates.some((template) => template.template_id === currentValue)
    ? currentValue
    : defaultTemplate?.template_id || "";
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
  const idInput = qs("#new-template-id");
  qs("#template-creator").dataset.mode = "create";
  qs("#template-editor-title").textContent = "Guided template";
  idInput.disabled = false;
  idInput.value = "";
  qs("#new-template-name").value = "";
  qs("#new-template-version").value = "1.0.0";
  qs("#new-template-description").value = "";
  qs("#new-template-prompt-block").value = "";
  renderTemplateSectionRows(DEFAULT_TEMPLATE_SECTIONS);
}

/**
 * @brief Toggle Template Creator.
 * @param {*} show Whether the template creator should be visible.
 */
function toggleTemplateCreator(show) {
  const panel = qs("#template-creator");
  const form = qs("#job-form");
  panel.classList.toggle("hidden", !show);
  form?.classList.toggle("template-editing", show);
  if (show && !qs("#template-sections")?.children.length) {
    renderTemplateSectionRows(DEFAULT_TEMPLATE_SECTIONS);
  } else if (!show) {
    resetTemplateCreator();
  }
}

/**
 * @brief Edit Template Inline.
 * @param {*} templateId Identifier of the template to edit.
 */
async function editTemplateInline(templateId) {
  try {
    const template = await fetchJSON(`/api/templates/${encodeURIComponent(templateId)}`);
    const panel = qs("#template-creator");
    panel.dataset.mode = "edit";
    qs("#template-editor-title").textContent = `Edit ${template.name}`;
    qs("#new-template-id").value = template.template_id;
    qs("#new-template-id").disabled = true;
    qs("#new-template-name").value = template.name || "";
    qs("#new-template-version").value = template.version || "1.0.0";
    qs("#new-template-description").value = template.description || "";
    qs("#new-template-prompt-block").value = template.prompt_block || "";
    renderTemplateSectionRows(template.sections?.length ? template.sections : DEFAULT_TEMPLATE_SECTIONS);
    toggleTemplateCreator(true);
  } catch (error) {
    alert(`Unable to load template: ${error.message}`);
  }
}

/**
 * @brief Render Template Section Rows.
 * @param {*} sections Template section definitions.
 */
function renderTemplateSectionRows(sections) {
  const container = qs("#template-sections");
  if (!container) return;
  container.innerHTML = "";
  sections.forEach((section) => addTemplateSectionRow(section));
}

/**
 * @brief Add Template Section Row.
 * @param {*} section Optional section defaults.
 */
function addTemplateSectionRow(section = {}) {
  const container = qs("#template-sections");
  if (!container) return;
  const row = document.createElement("div");
  row.className = "template-section-row";

  const heading = document.createElement("input");
  heading.type = "text";
  heading.placeholder = "#### Section heading:";
  heading.value = section.heading || "";
  heading.dataset.sectionField = "heading";

  const requiredLabel = document.createElement("label");
  const required = document.createElement("input");
  required.type = "checkbox";
  required.checked = section.required !== false;
  required.dataset.sectionField = "required";
  requiredLabel.append(required, document.createTextNode("Required"));

  const prefixLabel = document.createElement("label");
  const allowPrefix = document.createElement("input");
  allowPrefix.type = "checkbox";
  allowPrefix.checked = Boolean(section.allow_prefix);
  allowPrefix.dataset.sectionField = "allow_prefix";
  prefixLabel.title = "Accepts content on the same line as the heading, for example: ### Title: Project Sync";
  prefixLabel.append(allowPrefix, document.createTextNode("Allow inline content"));

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "small-btn danger-btn";
  remove.textContent = "Remove";
  remove.addEventListener("click", () => row.remove());

  row.append(heading, requiredLabel, prefixLabel, remove);
  container.appendChild(row);
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
    meta.textContent = `Version ${template.version}${template.is_default ? " | Default" : ""}${template.description ? ` | ${template.description}` : ""}`;

    const actions = document.createElement("div");
    actions.className = "model-action-row";

    const defaultBtn = document.createElement("button");
    defaultBtn.type = "button";
    defaultBtn.className = "small-btn";
    defaultBtn.textContent = template.is_default ? "Default" : "Set default";
    defaultBtn.disabled = Boolean(template.is_default);
    defaultBtn.addEventListener("click", async () => {
      try {
        await fetchJSON("/api/templates/default", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ template_id: template.template_id }),
        });
        await refreshSettings();
      } catch (error) {
        alert(`Unable to set default template: ${error.message}`);
      }
    });
    actions.append(defaultBtn);

    card.append(eyebrow, title, meta, actions);
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
      strong.textContent = model.stage === "formatter" ? formatFormatterModelLabel(model) : model.name;
      const small = document.createElement("small");
      small.textContent = model.stage === "formatter" ? `[${model.name}]` : `[${model.runtime}]`;
      title.append(strong, small);

      const info = document.createElement("div");
      info.className = "card-meta";
      const defaultForStage = state.localModelCatalog.defaults[model.stage] === model.model_id ? " | Default" : "";
      info.textContent = `Installed: ${model.installed ? "yes" : "no"} | Stage: ${model.stage}${defaultForStage}`;

      const details = document.createElement("div");
      details.className = "card-meta";
      details.textContent = formatLocalModelDetails(model);

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
  const [templates, models, profiles, downloads, localModelCatalog, localRuntimeDescriptors] = await Promise.all([
    fetchJSON("/api/templates"),
    fetchJSON("/api/models"),
    fetchJSON("/api/profiles"),
    fetchJSON("/api/models/downloads"),
    fetchJSON("/api/models/local"),
    fetchJSON("/api/models/local/runtimes"),
  ]);
  state.localModelCatalog = localModelCatalog;
  state.localRuntimeDescriptors = localRuntimeDescriptors || [];
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
  const form = qs("#local-model-form");
  const runtimeContainer = qs("#local-runtime-options");
  const fieldsContainer = qs("#local-model-fields");
  const help = qs("#local-model-help");
  const installBtn = qs("#install-local-model");
  const suggestions = qs("#local-model-suggestions");
  if (!stageSelect || !form || !runtimeContainer || !fieldsContainer) return;

  const stage = stageSelect.value || "transcription";
  if (form.dataset.stage !== stage && suggestions) {
    suggestions.innerHTML = "";
  }
  form.dataset.stage = stage;
  const showAdvanced = Boolean(qs("#local-model-advanced")?.checked);
  const descriptors = state.localRuntimeDescriptors.filter((item) => item.stage === stage);
  const visibleDescriptors = descriptors.filter((item) => showAdvanced || !item.advanced);
  const currentRuntime = form.dataset.runtime;
  const selected = visibleDescriptors.find((item) => item.runtime === currentRuntime) || visibleDescriptors[0] || descriptors[0];
  if (!selected) {
    runtimeContainer.innerHTML = "";
    fieldsContainer.innerHTML = "";
    help.textContent = "No runtime descriptors are available.";
    return;
  }
  form.dataset.runtime = selected.runtime;

  runtimeContainer.innerHTML = "";
  visibleDescriptors.forEach((descriptor) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "runtime-option";
    button.dataset.runtime = descriptor.runtime;
    button.classList.toggle("active", descriptor.runtime === selected.runtime);
    const name = document.createElement("strong");
    name.textContent = descriptor.label;
    const description = document.createElement("span");
    description.textContent = descriptor.description;
    button.append(name, description);
    button.addEventListener("click", () => {
      form.dataset.runtime = descriptor.runtime;
      if (suggestions) suggestions.innerHTML = "";
      syncLocalModelForm();
    });
    runtimeContainer.appendChild(button);
  });

  fieldsContainer.innerHTML = "";
  selected.fields.forEach((field) => {
    const label = document.createElement("label");
    const labelText = document.createElement("span");
    labelText.className = "model-field-label";
    const labelCopy = document.createElement("span");
    labelCopy.className = "model-field-copy";
    labelCopy.textContent = field.label;
    labelText.appendChild(labelCopy);
    if (field.required) {
      const required = document.createElement("span");
      required.className = "required-mark";
      required.textContent = "*";
      labelText.appendChild(required);
    }
    if (field.help) {
      const info = document.createElement("span");
      info.className = "field-info";
      info.title = field.help;
      info.setAttribute("aria-label", field.help);
      info.textContent = "?";
      labelText.appendChild(info);
    }
    const input = document.createElement("input");
    input.type = field.input_type || "text";
    input.dataset.configKey = field.key;
    input.placeholder = field.placeholder || "";
    input.required = Boolean(field.required);
    label.append(labelText, input);
    fieldsContainer.appendChild(label);
  });
  help.textContent = selected.description || "";
  installBtn?.classList.toggle("hidden", !selected.supports_install);
}

/**
 * @brief Register Local Model.
 */
async function registerLocalModel() {
  const status = qs("#local-model-status");
  const payload = buildLocalModelPayload();
  if (!payload) {
    status.textContent = "Fill the required fields before registering.";
    return;
  }

  status.textContent = "Registering...";
  try {
    await fetchJSON("/api/models/local", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    clearLocalModelForm();
    status.textContent = "Model registered.";
    await refreshSettings();
  } catch (error) {
    status.textContent = error.message;
  }
}

/**
 * @brief Build Local Model Payload.
 */
function buildLocalModelPayload() {
  const form = qs("#local-model-form");
  const stage = qs("#local-model-stage").value;
  const runtime = form?.dataset.runtime;
  const descriptor = getRuntimeDescriptor(stage, runtime);
  const name = qs("#local-model-name").value.trim();
  if (!descriptor || !name) return null;
  const config = {};
  for (const field of descriptor.fields) {
    const input = qs(`[data-config-key="${CSS.escape(field.key)}"]`);
    const value = input?.value.trim() || "";
    if (field.required && !value) return null;
    if (value) config[field.key] = value;
  }
  return {
    stage,
    runtime,
    name,
    languages: qs("#local-model-languages").value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean),
    notes: qs("#local-model-notes").value.trim(),
    set_as_default: qs("#local-model-default").checked,
    config,
  };
}

/**
 * @brief Get Runtime Descriptor.
 * @param {*} stage Stage id.
 * @param {*} runtime Runtime id.
 */
function getRuntimeDescriptor(stage, runtime) {
  return state.localRuntimeDescriptors.find((item) => item.stage === stage && item.runtime === runtime);
}

/**
 * @brief Clear Local Model Form.
 */
function clearLocalModelForm() {
  qs("#local-model-name").value = "";
  qs("#local-model-languages").value = "";
  qs("#local-model-notes").value = "";
  qsa("[data-config-key]").forEach((input) => {
    input.value = "";
  });
  const suggestions = qs("#local-model-suggestions");
  if (suggestions) suggestions.innerHTML = "";
}

/**
 * @brief Discover local model candidates.
 */
async function scanLocalModels() {
  const form = qs("#local-model-form");
  const stage = qs("#local-model-stage").value;
  const runtime = form?.dataset.runtime;
  const status = qs("#local-model-status");
  const suggestions = qs("#local-model-suggestions");
  if (!stage || !runtime || !suggestions) return;
  status.textContent = "Scanning known locations...";
  suggestions.innerHTML = "";
  try {
    const payload = await fetchJSON(
      `/api/models/local/discovery/${encodeURIComponent(stage)}/${encodeURIComponent(runtime)}`,
    );
    renderLocalModelSuggestions(payload.suggestions || []);
    status.textContent = payload.suggestions?.length ? "Scan complete." : "No candidates found in known locations.";
  } catch (error) {
    status.textContent = error.message;
  }
}

/**
 * @brief Render local model suggestions.
 * @param {*} suggestions Discovery suggestions.
 */
function renderLocalModelSuggestions(suggestions) {
  const container = qs("#local-model-suggestions");
  container.innerHTML = "";
  if (!suggestions.length) return;
  const title = document.createElement("p");
  title.className = "form-label";
  title.textContent = "Discovered candidates";
  container.appendChild(title);
  suggestions.forEach((suggestion) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "model-suggestion";
    const name = document.createElement("strong");
    name.textContent = suggestion.name;
    const details = document.createElement("span");
    details.textContent = `${suggestion.source}${suggestion.details ? ` | ${suggestion.details}` : ""}`;
    button.append(name, details);
    button.addEventListener("click", () => applyLocalModelSuggestion(suggestion));
    container.appendChild(button);
  });
}

/**
 * @brief Apply local model suggestion.
 * @param {*} suggestion Discovery suggestion.
 */
function applyLocalModelSuggestion(suggestion) {
  qs("#local-model-name").value = suggestion.name || "";
  Object.entries(suggestion.config || {}).forEach(([key, value]) => {
    const input = qs(`[data-config-key="${CSS.escape(key)}"]`);
    if (input) input.value = value;
  });
  qs("#local-model-status").textContent = "Candidate loaded. Review the fields, then register it.";
}

/**
 * @brief Install an Ollama model and register it.
 */
async function installLocalModel() {
  const status = qs("#local-model-status");
  const payload = buildLocalModelPayload();
  if (!payload) {
    status.textContent = "Fill the required fields before pulling with Ollama.";
    return;
  }
  status.textContent = "Starting Ollama pull...";
  try {
    const task = await fetchJSON("/api/models/local/installs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    startLocalModelInstallPolling(task.task_id);
  } catch (error) {
    status.textContent = error.message;
  }
}

/**
 * @brief Poll local model install status.
 * @param {*} taskId Install task id.
 */
function startLocalModelInstallPolling(taskId) {
  if (state.localModelInstallPoller) clearInterval(state.localModelInstallPoller);
  const poll = async () => {
    const status = qs("#local-model-status");
    try {
      const task = await fetchJSON(`/api/models/local/installs/${encodeURIComponent(taskId)}`);
      const suffix = task.percent ? ` (${task.percent.toFixed(1)}%)` : "";
      status.textContent = `${task.message || task.status}${suffix}`;
      if (task.status === "completed" || task.status === "failed") {
        clearInterval(state.localModelInstallPoller);
        state.localModelInstallPoller = null;
        if (task.status === "completed") {
          clearLocalModelForm();
          await refreshSettings();
        } else {
          status.textContent = task.error || task.message || "Install failed.";
        }
      }
    } catch (error) {
      clearInterval(state.localModelInstallPoller);
      state.localModelInstallPoller = null;
      status.textContent = error.message;
    }
  };
  poll();
  state.localModelInstallPoller = setInterval(poll, 1500);
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
 * @brief Format Stage Label.
 * @param {*} stage Current stage label from the API.
 */
function formatStageLabel(stage) {
  if (stage === "Validate/Normalize") {
    return "Validating &\nNormalizing";
  }
  return stage || "-";
}

/**
 * @brief Update Progress View.
 * @param {*} jobState Job state payload from the API.
 */
function updateProgressView(jobState) {
  qs("#overall-value").textContent = `${jobState.overall_percent.toFixed(1)}%`;
  qs("#stage-value").textContent = formatStageLabel(jobState.current_stage);
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
    speaker_ids: speaker.speaker_ids || [],
    review_group_id: speaker.review_group_id || "",
    suggested_name: speaker.suggested_name || "",
    matched_profile: speaker.matched_profile || null,
    snippets: (speaker.snippets || []).map((snippet) => snippet.snippet_id || snippet.snippet_path),
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
    const speakerIds = speaker.speaker_ids && speaker.speaker_ids.length ? speaker.speaker_ids : [speaker.speaker_id];
    card.dataset.speakerId = speaker.speaker_id;
    card.dataset.speakerIds = JSON.stringify(speakerIds);

    const header = document.createElement("div");
    header.className = "speaker-card-header";

    const title = document.createElement("h4");
    title.textContent = speakerIds.length > 1 ? `${speaker.speaker_id} group` : speaker.speaker_id;
    header.appendChild(title);

    if (speakerIds.length > 1) {
      const groupBadge = document.createElement("span");
      groupBadge.className = "speaker-match-badge grouped";
      groupBadge.textContent = `${speakerIds.length} diarized speakers`;
      header.appendChild(groupBadge);
    }

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
    const existingName = existingNames.get(speaker.speaker_id);
    input.value = existingName && existingName !== speaker.speaker_id
      ? existingName
      : speaker.suggested_name || existingName || speaker.speaker_id;
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
      const snippetName = String(snippet.snippet_path || "")
        .replace(/\\/g, "/")
        .split("/")
        .pop();
      if (!snippetName) {
        return;
      }
      const snippetId = snippet.snippet_id || snippetName.replace(/\.[^.]+$/, "");
      const row = document.createElement("div");
      row.className = "speaker-snippet-row";
      row.dataset.snippetId = snippetId;
      row.dataset.sourceSpeakerId = snippet.speaker_id || speaker.speaker_id;

      const audio = document.createElement("audio");
      audio.controls = true;
      audio.src = `/api/jobs/${jobState.job_id}/snippets/${encodeURIComponent(snippetName)}`;

      const actions = document.createElement("div");
      actions.className = "speaker-snippet-actions";
      [
        ["keep", "Keep", "Use this clip as evidence for this speaker."],
        ["split", "Move", "This clip belongs to another speaker; create a separate name field."],
        ["exclude", "Skip", "This clip is too unclear to use for naming or profile saving."],
      ].forEach(([value, label, help]) => {
        const option = document.createElement("label");
        option.className = "speaker-snippet-action-pill";
        option.setAttribute("aria-label", label);
        const radio = document.createElement("input");
        radio.type = "radio";
        radio.name = `snippet-action-${snippetId}`;
        radio.value = value;
        radio.checked = value === "keep";
        radio.className = "speaker-snippet-action";
        const text = document.createElement("span");
        text.textContent = label;
        const info = document.createElement("span");
        info.className = "speaker-snippet-info";
        info.tabIndex = 0;
        info.setAttribute("role", "button");
        info.setAttribute("aria-label", help);
        info.textContent = "i";
        const hint = document.createElement("span");
        hint.className = "speaker-snippet-action-help";
        hint.textContent = help;
        info.appendChild(hint);
        option.append(radio, text, info);
        actions.appendChild(option);
      });

      const splitInput = document.createElement("input");
      splitInput.type = "text";
      splitInput.className = "speaker-split-name hidden";
      splitInput.placeholder = "Name for this new speaker";
      splitInput.dataset.snippetId = snippetId;
      splitInput.dataset.syntheticSpeakerId = `SPLIT_${snippetId}`;

      actions.addEventListener("change", () => {
        const selected = actions.querySelector("input:checked");
        const splitSelected = selected && selected.value === "split";
        splitInput.classList.toggle("hidden", !splitSelected);
        row.classList.toggle("snippet-excluded", selected && selected.value === "exclude");
      });

      row.append(audio, actions, splitInput);
      snippets.appendChild(row);
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
  const mappings = qsa(".speaker-card").map((card) => {
    const input = card.querySelector(".speaker-name");
    const speakerId = card.dataset.speakerId;
    const speakerIds = JSON.parse(card.dataset.speakerIds || `["${speakerId}"]`);
    const save = qsa(".speaker-save-profile").find((item) => item.dataset.speakerId === speakerId);
    const snippetRows = Array.from(card.querySelectorAll(".speaker-snippet-row"));
    const assignedSnippetIds = snippetRows
      .filter((row) => {
        const selected = row.querySelector(".speaker-snippet-action:checked");
        return selected && selected.value === "keep";
      })
      .map((row) => row.dataset.snippetId)
      .filter(Boolean);
    const typedName = input.value.trim();
    const hasRealName = Boolean(typedName && !speakerIds.includes(typedName) && typedName !== speakerId);
    const allClipsRejected = snippetRows.length > 0 && assignedSnippetIds.length === 0;
    return {
      speaker_id: speakerId,
      name: typedName || speakerId,
      save_voice_profile: Boolean(save && save.checked),
      speaker_ids: speakerIds,
      assigned_snippet_ids: assignedSnippetIds,
      exclude_from_mom: allClipsRejected && !hasRealName,
    };
  });
  const snippetActions = [];
  qsa(".speaker-snippet-row").forEach((row) => {
    const selected = row.querySelector(".speaker-snippet-action:checked");
    if (!selected) return;
    const action = selected.value;
    const payload = {
      snippet_id: row.dataset.snippetId,
      source_speaker_id: row.dataset.sourceSpeakerId,
      action,
    };
    if (action === "split") {
      const splitInput = row.querySelector(".speaker-split-name");
      payload.target_speaker_id = splitInput.dataset.syntheticSpeakerId;
      mappings.push({
        speaker_id: payload.target_speaker_id,
        name: splitInput.value.trim() || payload.target_speaker_id,
        save_voice_profile: false,
        speaker_ids: [payload.target_speaker_id],
        assigned_snippet_ids: [row.dataset.snippetId],
        exclude_from_mom: false,
      });
    }
    snippetActions.push(payload);
  });

  const grouped = new Map();
  mappings.filter((item) => !item.exclude_from_mom).forEach((item) => {
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
    body: JSON.stringify({ mappings, snippet_actions: snippetActions }),
  });
  switchTab("progress");
}

/**
 * @brief Load Result.
 * @param {*} jobId Identifier of the active job.
 */
async function loadResult(jobId) {
  qs("#download-link").href = `/api/jobs/${jobId}/download/mom`;
  qs("#transcript-download-link").href = `/api/jobs/${jobId}/download/full-meeting-transcript`;
  await Promise.all([
    loadTextPreview({
      primaryUrl: `/api/jobs/${jobId}/mom`,
      fallbackUrl: `/api/jobs/${jobId}/download/mom`,
      selector: "#mom-preview",
      failureText: "Unable to load markdown preview.",
    }),
    loadTextPreview({
      primaryUrl: `/api/jobs/${jobId}/full-meeting-transcript`,
      fallbackUrl: `/api/jobs/${jobId}/artifacts/formatter_user_prompt`,
      selector: "#full-transcript-preview",
      failureText: "Unable to load full meeting transcript.",
    }),
  ]);
}

/**
 * @brief Load text preview from an endpoint.
 * @param {*} options Request and target details.
 */
async function loadTextPreview({ primaryUrl, fallbackUrl, selector, failureText }) {
  try {
    let response = await fetch(primaryUrl, { cache: "no-store" });
    if (!response.ok && fallbackUrl) {
      response = await fetch(fallbackUrl, { cache: "no-store" });
    }
    if (!response.ok) throw new Error(`Unable to fetch preview (${response.status})`);
    const text = await response.text();
    qs(selector).textContent = text;
  } catch (error) {
    qs(selector).textContent = failureText;
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
  const sections = qsa(".template-section-row")
    .map((row) => {
      const heading = row.querySelector('[data-section-field="heading"]')?.value.trim() || "";
      if (!heading) return null;
      return {
        heading,
        required: Boolean(row.querySelector('[data-section-field="required"]')?.checked),
        allow_prefix: Boolean(row.querySelector('[data-section-field="allow_prefix"]')?.checked),
        empty_value: "None",
      };
    })
    .filter(Boolean);

  if (!templateId || !name || !promptBlock) {
    alert("Template ID, Name, and Prompt Block are required.");
    return;
  }
  if (!sections.length) {
    alert("Add at least one section so AutoMoM knows what structure to request and validate.");
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
        sections,
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
  qs("#edit-selected-template").addEventListener("click", () => {
    const templateId = qs("#template-select").value;
    if (templateId) editTemplateInline(templateId);
  });
  qs("#cancel-template-inline").addEventListener("click", () => toggleTemplateCreator(false));
  qs("#save-template-inline").addEventListener("click", saveTemplateInline);
  qs("#add-template-section").addEventListener("click", () => addTemplateSectionRow());
  qs("#local-model-stage").addEventListener("change", syncLocalModelForm);
  qs("#local-model-advanced").addEventListener("change", syncLocalModelForm);
  qs("#scan-local-models").addEventListener("click", scanLocalModels);
  qs("#register-local-model").addEventListener("click", registerLocalModel);
  qs("#install-local-model").addEventListener("click", installLocalModel);
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

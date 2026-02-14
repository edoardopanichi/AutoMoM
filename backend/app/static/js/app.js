const state = {
  currentJobId: null,
  eventSource: null,
  latestJobState: null,
  modelDownloads: {},
  modelDownloadPoller: null,
  speakerFormFingerprint: null,
  formatterModelTag: "",
};

const qs = (selector) => document.querySelector(selector);
const qsa = (selector) => Array.from(document.querySelectorAll(selector));

function switchTab(tabId) {
  qsa(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabId);
  });
  qsa(".panel").forEach((panel) => panel.classList.toggle("active", panel.id === tabId));
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

async function loadStartupData() {
  const [templates, models, profiles, downloads, formatterModel] = await Promise.all([
    fetchJSON("/api/templates"),
    fetchJSON("/api/models"),
    fetchJSON("/api/profiles"),
    fetchJSON("/api/models/downloads"),
    fetchJSON("/api/models/formatter"),
  ]);
  state.formatterModelTag = formatterModel.model_tag || "";
  state.modelDownloads = Object.fromEntries(downloads.map((item) => [item.model_id, item]));
  renderTemplateSelect(templates);
  renderTemplates(templates);
  renderModels(models);
  renderProfiles(profiles);
  startModelDownloadPolling();
}

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

function renderTemplates(templates) {
  const container = qs("#templates");
  container.innerHTML = "";
  templates.forEach((template) => {
    const card = document.createElement("div");
    card.className = "template-card";
    card.textContent = `${template.template_id}: ${template.name} (${template.version})`;
    container.appendChild(card);
  });
}

function renderModels(models) {
  const container = qs("#models");
  container.innerHTML = "";

  models.forEach((model) => {
    const downloadState = state.modelDownloads[model.model_id] || {
      status: model.installed ? "completed" : "idle",
      percent: model.installed ? 100 : 0,
      downloaded_bytes: 0,
      total_bytes: null,
      error: null,
    };

    const card = document.createElement("div");
    card.className = "model-card";
    card.dataset.modelId = model.model_id;

    const title = document.createElement("div");
    title.innerHTML = `<strong>${model.name}</strong> <small>[${model.model_id}]</small>`;

    const info = document.createElement("div");
    info.textContent = `Installed: ${model.installed ? "yes" : "no"} | Size: ${model.size_mb} MB | Source: ${model.source}`;

    const formatterModelWrap = document.createElement("div");
    const actionRow = document.createElement("div");
    actionRow.className = "model-action-row";
    let formatterModelInput = null;
    if (model.model_id === "formatter") {
      const input = document.createElement("input");
      input.type = "text";
      input.value = state.formatterModelTag || "";
      input.placeholder = "ollama model tag (e.g. qwen2.5:3b-instruct-q5_K_M)";
      input.style.width = "100%";
      formatterModelInput = input;

      const saveBtn = document.createElement("button");
      saveBtn.className = "download-btn formatter-set-btn";
      saveBtn.textContent = "Set formatter model";
      saveBtn.addEventListener("click", async () => {
        try {
          const payload = await fetchJSON("/api/models/formatter", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_tag: input.value.trim() }),
          });
          state.formatterModelTag = payload.model_tag;
          await refreshSettings();
        } catch (error) {
          alert(`Unable to set formatter model: ${error.message}`);
        }
      });

      formatterModelWrap.append(input);
      actionRow.append(saveBtn);
    }

    const download = document.createElement("button");
    download.className = "download-btn";
    download.textContent = downloadState.status === "running" ? "Downloading..." : "Download";
    download.disabled = model.installed || downloadState.status === "running";
    download.addEventListener("click", async () => {
      try {
        if (model.model_id === "formatter" && formatterModelInput) {
          const selectedTag = formatterModelInput.value.trim();
          if (!selectedTag) {
            throw new Error("Formatter model tag is required");
          }
          const payload = await fetchJSON("/api/models/formatter", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_tag: selectedTag }),
          });
          state.formatterModelTag = payload.model_tag;
        }
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

    card.append(title, info, formatterModelWrap, actionRow, downloadStatus, progressWrap);
    container.appendChild(card);
  });
}

function renderProfiles(profiles) {
  const container = qs("#profiles");
  container.innerHTML = "";
  if (!profiles.length) {
    container.textContent = "No voice profiles saved.";
    return;
  }
  profiles.forEach((profile) => {
    const card = document.createElement("div");
    card.className = "profile-card";
    card.textContent = `${profile.name} (${profile.profile_id})`;
    container.appendChild(card);
  });
}

async function refreshSettings() {
  const [templates, models, profiles, downloads, formatterModel] = await Promise.all([
    fetchJSON("/api/templates"),
    fetchJSON("/api/models"),
    fetchJSON("/api/profiles"),
    fetchJSON("/api/models/downloads"),
    fetchJSON("/api/models/formatter"),
  ]);
  state.formatterModelTag = formatterModel.model_tag || "";
  state.modelDownloads = Object.fromEntries(downloads.map((item) => [item.model_id, item]));
  renderTemplateSelect(templates);
  renderTemplates(templates);
  renderModels(models);
  renderProfiles(profiles);
}

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

async function refreshModelDownloads() {
  const downloads = await fetchJSON("/api/models/downloads");
  state.modelDownloads = Object.fromEntries(downloads.map((item) => [item.model_id, item]));
}

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

function setProgressBars(overall, stagePercent) {
  qs("#overall-bar").style.width = `${overall}%`;
  qs("#stage-bar").style.width = `${stagePercent}%`;
}

function updateProgressView(jobState) {
  qs("#overall-value").textContent = `${jobState.overall_percent.toFixed(1)}%`;
  qs("#stage-value").textContent = jobState.current_stage || "-";
  qs("#stage-percent").textContent = `${jobState.stage_percent.toFixed(1)}%`;
  qs("#segment-progress").textContent = jobState.transcript_segment_progress || "-";
  setProgressBars(jobState.overall_percent, jobState.stage_percent);
  qs("#logs").textContent = (jobState.logs || []).join("\n");
}

function speakerFormFingerprint(jobState) {
  const speakerInfo = jobState.speaker_info;
  if (!speakerInfo) {
    return null;
  }
  const speakers = speakerInfo.speakers.map((speaker) => ({
    speaker_id: speaker.speaker_id,
    suggested_name: speaker.suggested_name || "",
    snippets: (speaker.snippets || []).map((snippet) => snippet.snippet_path),
  }));
  return JSON.stringify({
    job_id: jobState.job_id,
    detected_speakers: speakerInfo.detected_speakers,
    speakers,
  });
}

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

    const title = document.createElement("h4");
    title.textContent = speaker.speaker_id;

    const input = document.createElement("input");
    input.type = "text";
    input.value =
      existingNames.get(speaker.speaker_id) ||
      speaker.suggested_name ||
      speaker.speaker_id;
    input.dataset.speakerId = speaker.speaker_id;
    input.className = "speaker-name";

    const toggleWrap = document.createElement("label");
    toggleWrap.style.display = "inline-flex";
    toggleWrap.style.gap = "0.4rem";
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.dataset.speakerId = speaker.speaker_id;
    toggle.className = "speaker-save-profile";
    toggle.checked = Boolean(existingSaveProfile.get(speaker.speaker_id));
    toggleWrap.append(toggle, "Save as voice profile");

    const snippets = document.createElement("div");
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

    card.append(title, input, toggleWrap, snippets);
    form.appendChild(card);
  });
}

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

  await fetchJSON(`/api/jobs/${state.currentJobId}/speaker-mapping`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mappings }),
  });
  switchTab("progress");
}

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

function openJobEventStream(jobId) {
  if (state.eventSource) {
    state.eventSource.close();
  }
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  state.eventSource = source;

  source.onmessage = async (event) => {
    const jobState = JSON.parse(event.data);
    state.latestJobState = jobState;
    updateProgressView(jobState);

    if (jobState.status === "waiting_speaker_input") {
      const nextFingerprint = speakerFormFingerprint(jobState);
      if (nextFingerprint !== state.speakerFormFingerprint) {
        renderSpeakerForm(jobState);
        state.speakerFormFingerprint = nextFingerprint;
      }
      switchTab("speaker");
      return;
    }

    state.speakerFormFingerprint = null;

    if (jobState.status === "completed") {
      await loadResult(jobId);
      switchTab("result");
      source.close();
      return;
    }

    if (jobState.status === "failed" || jobState.status === "cancelled") {
      alert(`Job ${jobState.status}: ${jobState.error || "No details"}`);
      source.close();
    }
  };

  source.onerror = () => {
    source.close();
  };
}

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
  formData.append("language_mode", qs("#language-mode").value);
  formData.append("title", qs("#meeting-title").value);

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
    switchTab("progress");
    openJobEventStream(job.job_id);
  } catch (error) {
    qs("#start-error").textContent = error.message;
  }
}

async function cancelJob() {
  if (!state.currentJobId) return;
  await fetchJSON(`/api/jobs/${state.currentJobId}/cancel`, { method: "POST" });
}

function bindEvents() {
  qsa(".tab").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });
  qs("#job-form").addEventListener("submit", startJob);
  qs("#cancel-job").addEventListener("click", cancelJob);
  qs("#submit-speakers").addEventListener("click", submitSpeakerMapping);
}

async function init() {
  bindEvents();
  await loadStartupData();
}

init();

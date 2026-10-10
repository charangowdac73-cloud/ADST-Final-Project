// ---------------------------------------------------------------
// Cyber Threat Detection — frontend logic
// Talks to the Flask backend via same-origin fetch() calls.
// ---------------------------------------------------------------

const LABELS = ["normal", "dos", "probe", "r2l"];
const LABEL_TEXT = { normal: "NORMAL", dos: "DOS ATTACK", probe: "PROBE", r2l: "R2L ATTACK" };

const NUMERIC_FIELDS = [
  "duration", "src_bytes", "dst_bytes", "count", "srv_count",
  "dst_host_count", "serror_rate", "same_srv_rate", "diff_srv_rate",
];

const form = document.getElementById("trafficForm");
const formError = document.getElementById("formError");
const sampleBtn = document.getElementById("sampleBtn");
const sampleType = document.getElementById("sampleType");
const clearLogBtn = document.getElementById("clearLogBtn");

const resultEmpty = document.getElementById("resultEmpty");
const resultBody = document.getElementById("resultBody");
const verdict = document.getElementById("verdict");
const verdictLabel = document.getElementById("verdictLabel");
const verdictConfidence = document.getElementById("verdictConfidence");
const probList = document.getElementById("probList");

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

const logBody = document.getElementById("logBody");
let logHistory = [];

// ---------------------------------------------------------------
// Startup: populate dropdowns and stats from the backend
// ---------------------------------------------------------------
async function init() {
  try {
    const stats = await fetchJSON("/api/stats");
    populateSelect("protocol_type", stats.protocols);
    populateSelect("service", stats.services);
    populateSelect("flag", stats.flags);
    renderStats(stats);
    setStatus(true);
  } catch (err) {
    setStatus(false);
    console.error(err);
  }
  await loadSample("normal");
}

function setStatus(online) {
  statusDot.className = "dot " + (online ? "online" : "offline");
  statusText.textContent = online ? "Model online" : "Model unavailable";
}

function populateSelect(id, options) {
  const el = document.getElementById(id);
  el.innerHTML = "";
  options.forEach((opt) => {
    const o = document.createElement("option");
    o.value = opt;
    o.textContent = opt.toUpperCase();
    el.appendChild(o);
  });
}

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

// ---------------------------------------------------------------
// Sample generation
// ---------------------------------------------------------------
async function loadSample(type) {
  try {
    const record = await fetchJSON(`/api/sample?type=${encodeURIComponent(type)}`);
    Object.entries(record).forEach(([key, value]) => {
      const field = document.getElementById(key);
      if (field) field.value = value;
    });
  } catch (err) {
    formError.textContent = err.message;
  }
}

sampleBtn.addEventListener("click", () => loadSample(sampleType.value));

// ---------------------------------------------------------------
// Analyze traffic
// ---------------------------------------------------------------
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  formError.textContent = "";

  const payload = {};
  const formData = new FormData(form);
  for (const [key, value] of formData.entries()) {
    payload[key] = NUMERIC_FIELDS.includes(key) ? Number(value) : value;
  }

  try {
    const result = await fetchJSON("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderResult(result);
    addLogRow(result, payload);
  } catch (err) {
    formError.textContent = err.message;
  }
});

function renderResult(result) {
  resultEmpty.classList.add("hidden");
  resultBody.classList.remove("hidden");

  verdict.className = "verdict " + result.prediction;
  verdictLabel.textContent = LABEL_TEXT[result.prediction] || result.prediction.toUpperCase();
  verdictConfidence.textContent = (result.confidence * 100).toFixed(1) + "% confidence";

  probList.innerHTML = "";
  LABELS.forEach((label) => {
    const pct = (result.probabilities[label] || 0) * 100;
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-name">${label}</span>
      <span class="prob-track"><span class="prob-fill ${label}" style="width:${pct}%"></span></span>
      <span class="prob-pct">${pct.toFixed(1)}%</span>
    `;
    probList.appendChild(row);
  });
}

// ---------------------------------------------------------------
// Model performance stats + confusion matrix
// ---------------------------------------------------------------
function renderStats(stats) {
  document.getElementById("accuracyValue").textContent = (stats.accuracy * 100).toFixed(2) + "%";
  document.getElementById("trainSizeValue").textContent = stats.n_train;
  document.getElementById("testSizeValue").textContent = stats.n_test;

  const table = document.getElementById("cmTable");
  const labels = stats.labels;
  let html = "<tr><th>Actual \\ Predicted</th>" + labels.map((l) => `<th>${l}</th>`).join("") + "</tr>";
  stats.confusion_matrix.forEach((row, i) => {
    html += `<tr><th>${labels[i]}</th>` + row.map((val, j) => {
      const cls = i === j ? "diag" : (val > 0 ? "off-diag-hot" : "");
      return `<td class="${cls}">${val}</td>`;
    }).join("") + "</tr>";
  });
  table.innerHTML = html;
}

// ---------------------------------------------------------------
// Detection log
// ---------------------------------------------------------------
function addLogRow(result, payload) {
  const time = new Date().toLocaleTimeString();
  logHistory.unshift({ time, result, payload });
  renderLog();
}

function renderLog() {
  if (logHistory.length === 0) {
    logBody.innerHTML = `<tr><td colspan="5" class="log-empty">No traffic analyzed yet.</td></tr>`;
    return;
  }
  logBody.innerHTML = logHistory.map(({ time, result, payload }) => `
    <tr>
      <td>${time}</td>
      <td><span class="tag ${result.prediction}">${LABEL_TEXT[result.prediction]}</span></td>
      <td>${(result.confidence * 100).toFixed(1)}%</td>
      <td>${payload.protocol_type}</td>
      <td>${payload.service}</td>
    </tr>
  `).join("");
}

clearLogBtn.addEventListener("click", () => {
  logHistory = [];
  renderLog();
});

init();

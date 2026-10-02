// script.js - connects the HTML form to the Flask backend

const form = document.getElementById("cropForm");
const errorBox = document.getElementById("error");
const resultBox = document.getElementById("result");

// Allowed ranges (same as backend, for instant feedback)
const RANGES = { N: [0, 200], P: [0, 200], K: [0, 250], temperature: [0, 55], humidity: [0, 100], ph: [0, 14], rainfall: [0, 500] };

// Emoji icons for each of the 22 crops in the dataset
const ICONS = { rice: "🌾", maize: "🌽", chickpea: "🫘", kidneybeans: "🫘", pigeonpeas: "🫛", mothbeans: "🫘",
  mungbean: "🫛", blackgram: "🫘", lentil: "🫘", pomegranate: "🍎", banana: "🍌", mango: "🥭", grapes: "🍇",
  watermelon: "🍉", muskmelon: "🍈", apple: "🍏", orange: "🍊", papaya: "🥭", coconut: "🥥", cotton: "☁️",
  jute: "🌿", coffee: "☕" };

function showError(msg) { errorBox.textContent = msg; errorBox.hidden = false; }

// Validate inputs in the browser before sending
function collectInputs() {
  const data = {}, problems = [];
  for (const [name, [lo, hi]] of Object.entries(RANGES)) {
    const input = form.elements[name];
    const v = input.value.trim();
    input.classList.remove("invalid");
    if (v === "" || isNaN(v) || +v < lo || +v > hi) {
      input.classList.add("invalid");
      problems.push(`${name} (${lo}–${hi})`);
    } else data[name] = parseFloat(v);
  }
  return { data, problems };
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.hidden = true;
  const { data, problems } = collectInputs();
  if (problems.length) return showError("Please enter valid values for: " + problems.join(", "));

  const btn = form.querySelector("button[type=submit]");
  btn.disabled = true; btn.textContent = "Predicting...";
  try {
    // Send inputs to Flask /predict endpoint
    const res = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
    const out = await res.json();
    if (!res.ok) return showError(out.error || "Something went wrong.");
    showResult(out);
  } catch {
    showError("Could not reach the server. Is the Flask app running?");
  } finally {
    btn.disabled = false; btn.textContent = "Recommend Crop";
  }
});

function showResult(out) {
  document.getElementById("cropIcon").textContent = ICONS[out.crop] || "🌱";
  document.getElementById("cropName").textContent = out.crop;
  document.getElementById("confidence").textContent = (out.confidence * 100).toFixed(1) + "%";
  document.getElementById("source").textContent = out.explanation.source;
  document.getElementById("summary").textContent = out.explanation.summary;
  document.getElementById("points").innerHTML = out.explanation.points.map((p) => `<li>${p}</li>`).join("");
  resultBox.hidden = false;
  resultBox.scrollIntoView({ behavior: "smooth" });
}

document.getElementById("resetBtn").addEventListener("click", () => {
  form.reset(); resultBox.hidden = true; errorBox.hidden = true;
  form.querySelectorAll("input").forEach((i) => i.classList.remove("invalid"));
  form.scrollIntoView({ behavior: "smooth" });
});

document.getElementById("sampleBtn").addEventListener("click", () => {
  const s = { N: 90, P: 42, K: 43, temperature: 20.9, humidity: 82, ph: 6.5, rainfall: 203 };
  for (const k in s) form.elements[k].value = s[k];
});

// Load REAL evaluation metrics from the backend
async function loadMetrics() {
  const m = await (await fetch("/metrics")).json();
  const pct = (x) => (x * 100).toFixed(2) + "%";
  document.getElementById("accuracy").textContent = pct(m.accuracy);
  document.getElementById("precision").textContent = pct(m.macro_avg.precision);
  document.getElementById("recall").textContent = pct(m.macro_avg["recall"]);
  document.getElementById("f1").textContent = pct(m.macro_avg["f1-score"]);
  document.getElementById("splitInfo").textContent =
    `Trained on ${m.train_size} records, tested on ${m.test_size} unseen records across ${m.num_classes} crops.`;

  const imp = Object.entries(m.feature_importance).sort((a, b) => b[1] - a[1]);
  const max = imp[0][1];
  document.getElementById("importance").innerHTML = imp.map(([f, v]) =>
    `<div class="bar"><span>${f}</span><div class="track"><div class="fill" style="width:${(v / max) * 100}%"></div></div><span>${(v * 100).toFixed(1)}%</span></div>`).join("");

  let rows = "<tr><th>Crop</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>";
  for (const c of m.labels) {
    const r = m.per_class[c];
    rows += `<tr><td>${c}</td><td>${r.precision.toFixed(2)}</td><td>${r.recall.toFixed(2)}</td><td>${r["f1-score"].toFixed(2)}</td><td>${r.support}</td></tr>`;
  }
  document.getElementById("reportTable").innerHTML = rows;

  let cm = "<tr><th>Actual ↓ / Pred →</th>" + m.labels.map((l) => `<th title="${l}">${l.slice(0, 3)}</th>`).join("") + "</tr>";
  m.confusion_matrix.forEach((row, i) => {
    cm += `<tr><th>${m.labels[i]}</th>` + row.map((v, j) =>
      `<td class="${i === j ? "diag" : v ? "miss" : ""}">${v || ""}</td>`).join("") + "</tr>";
  });
  document.getElementById("cmTable").innerHTML = cm;
}
loadMetrics();

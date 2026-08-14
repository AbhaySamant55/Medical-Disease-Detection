/* ===========================================================================
   Vitalis frontend.

   Vanilla ES modules, no framework and no build step — `python -m web.server`
   is the whole toolchain. State is small enough (which view, which condition,
   the last result) that a store would be ceremony.

   Animation is driven by a two-frame pattern: elements are inserted in their
   "from" state, then a rAF tick flips the class that transitions them in. Doing
   it in one pass would let the browser coalesce both styles into a single
   computation and nothing would move.
   =========================================================================== */
import { initHelix } from "./scene.js";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const api = async (url, opts) => {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
};
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
/** Cut at a word boundary rather than mid-word, and only if it actually helps. */
const trim = (s, n) => {
  s = String(s || "");
  if (s.length <= n) return s;
  const cut = s.slice(0, n);
  return cut.slice(0, cut.lastIndexOf(" ")) + "…";
};
const esc = (s) => String(s).replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const state = { catalog: null, disease: null, imaging: null, helix: null };

/* Let the browser compute the "from" styles before flipping to the "to" ones;
   applied in one pass they would be coalesced and nothing would move.
   setTimeout rather than requestAnimationFrame on purpose: rAF does not fire
   while a tab is in the background, which would leave bars stuck at zero and
   the loading overlay never dismissed. */
const nextTick = (fn) => setTimeout(fn, 32);

/* ----------------------------------------------------------------- theme */
function initTheme() {
  const saved = localStorage.getItem("vitalis-theme");
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("vitalis-theme", next);
    state.helix?.refreshTheme();
  });
}

/* ------------------------------------------------------------ navigation */
function showView(id) {
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${id}`));
  $("#main").scrollTo?.({ top: 0 });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function setActiveNav(key) {
  $$(".nav-item").forEach((b) =>
    b.classList.toggle("active", b.dataset.key === key || b.dataset.view === key));
}

/* ---------------------------------------------- animated number counters */
function countUp(el, target, { suffix = "", decimals = 1, ms = 850 } = {}) {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
    el.textContent = target.toFixed(decimals) + suffix;
    return;
  }
  const final = target.toFixed(decimals) + suffix;
  let done = false;
  const start = performance.now();
  const tick = (now) => {
    if (done) return;
    const t = Math.min((now - start) / ms, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = (target * eased).toFixed(decimals) + suffix;
    if (t < 1) requestAnimationFrame(tick);
    else done = true;
  };
  requestAnimationFrame(tick);
  // rAF is suspended while the tab is in the background, which would leave the
  // figure frozen at 0. This guarantees the true value lands either way.
  setTimeout(() => { done = true; el.textContent = final; }, ms + 60);
}

/* ------------------------------------------------------------------ home */
function renderHome() {
  const { tabular, imaging } = state.catalog;
  const total = tabular.length + imaging.length;
  $("#cond-count").textContent = total;

  const avgAcc = tabular.reduce((s, d) => s + d.metrics.accuracy, 0) / (tabular.length || 1);
  const algos = tabular[0]?.algorithms ?? 10;
  const samples = tabular.reduce((s, d) => s + d.samples, 0);

  $("#home-stats").innerHTML = [
    ["Conditions", total, "clinical + imaging"],
    ["Algorithms each", algos, "weighted consensus"],
    ["Mean accuracy", pct(avgAcc, 1), "held-out test sets"],
    ["Training records", samples.toLocaleString(), "public research data"],
  ].map(([k, v, s], i) => `
    <div class="stat" style="animation-delay:${i * 0.06}s">
      <div class="stat-k">${esc(k)}</div>
      <div class="stat-v">${esc(v)}</div>
      <div class="stat-s">${esc(s)}</div>
    </div>`).join("");

  // one row per condition: icon, description, figures. A grid of coloured
  // tiles would give each condition a different visual weight for no reason.
  const card = (d, kind, i) => `
    <button class="dcard" data-kind="${kind}" data-key="${esc(d.key)}"
            style="animation-delay:${Math.min(i * 0.04, 0.3)}s">
      <span class="ico">${d.icon}</span>
      <span>
        <h4>${esc(d.name)}</h4>
        <p>${esc(trim(d.blurb, 104))}</p>
      </span>
      <span class="meta">
        ${kind === "tabular"
          ? `<span>${pct(d.metrics.accuracy, 0)} <b>acc</b></span>
             <span>${d.metrics.roc_auc.toFixed(3)} <b>auc</b></span>`
          : `<span>${d.classes.length} <b>classes</b></span>
             <span>${d.algorithms} <b>models</b></span>`}
      </span>
    </button>`;

  $("#home-cards").innerHTML =
    tabular.map((d, i) => card(d, "tabular", i)).join("") +
    imaging.map((d, i) => card(d, "imaging", tabular.length + i)).join("");

  $$("#home-cards .dcard").forEach((el) => el.addEventListener("click", () => {
    el.dataset.kind === "tabular" ? openDisease(el.dataset.key)
                                  : openImaging(el.dataset.key);
  }));
}

function renderNav() {
  const item = (d, kind) => `
    <button class="nav-item" data-kind="${kind}" data-key="${esc(d.key)}">
      <span class="nav-ico">${d.icon}</span><span>${esc(d.name)}</span>
    </button>`;
  $("#nav-tabular").innerHTML = state.catalog.tabular.map((d) => item(d, "tabular")).join("");
  $("#nav-imaging").innerHTML = state.catalog.imaging.length
    ? state.catalog.imaging.map((d) => item(d, "imaging")).join("")
    : `<p class="muted small" style="padding:6px 11px">No imaging models installed.</p>`;

  $$("#nav .nav-item").forEach((el) => el.addEventListener("click", () => {
    if (el.dataset.view === "performance") { openPerformance(); return; }
    el.dataset.kind === "tabular" ? openDisease(el.dataset.key)
                                  : openImaging(el.dataset.key);
  }));
}

/* --------------------------------------------------------------- disease */
async function openDisease(key) {
  const data = await api(`/api/disease/${key}`);
  state.disease = data;
  const s = data.summary;

  $("#d-title").textContent = `${s.icon}  ${s.name}`;
  $("#d-blurb").textContent = s.blurb;
  $("#d-metrics").innerHTML = [
    ["Accuracy", pct(s.metrics.accuracy, 1)],
    ["ROC-AUC", s.metrics.roc_auc.toFixed(3)],
    ["Sensitivity", pct(s.metrics.recall, 1)],
    ["Baseline", pct(s.baseline, 1)],
  ].map(([k, v], i) => `<div class="chip" style="animation-delay:${i * .06}s">
      ${esc(k)} <b>${esc(v)}</b></div>`).join("");

  $("#d-form").innerHTML = data.fields.map((f, i) => {
    const delay = `style="animation-delay:${Math.min(i * 0.025, 0.4)}s"`;
    if (f.kind === "categorical") {
      return `<div class="field" ${delay}>
        <label for="f-${esc(f.name)}">${esc(f.label)}</label>
        <select id="f-${esc(f.name)}" name="${esc(f.name)}">
          ${(f.options || []).map((o) =>
            `<option ${o === f.default ? "selected" : ""}>${esc(o)}</option>`).join("")}
        </select></div>`;
    }
    const step = f.integer ? 1 : Math.max(+(f.step ?? 0.01).toFixed(4), 0.0001);
    return `<div class="field" ${delay}>
      <label for="f-${esc(f.name)}">${esc(f.label)}</label>
      <input id="f-${esc(f.name)}" name="${esc(f.name)}" type="number"
             value="${f.default}" step="${step}"
             min="${f.min ?? ""}" max="${f.max ?? ""}">
      ${f.observed_min !== undefined
        ? `<div class="hint">typical ${(+f.observed_min).toFixed(1)}–${(+f.observed_max).toFixed(1)}</div>`
        : ""}
    </div>`;
  }).join("");

  $("#d-empty").hidden = false;
  $("#d-result").hidden = true;
  $("#d-votes-panel").hidden = true;
  setActiveNav(key);
  showView("disease");
}

function resetForm() {
  if (!state.disease) return;
  state.disease.fields.forEach((f) => {
    const el = $(`#f-${CSS.escape(f.name)}`);
    if (el) el.value = f.default;
  });
}

async function runPrediction() {
  if (!state.disease) return;
  const btn = $("#d-run");
  btn.classList.add("loading");

  const values = {};
  state.disease.fields.forEach((f) => {
    const el = $(`#f-${CSS.escape(f.name)}`);
    if (el) values[f.name] = el.value;
  });

  try {
    const r = await api(`/api/predict/${state.disease.summary.key}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values }),
    });
    renderVerdict(r);
    renderVotes(r);
  } catch (err) {
    $("#d-result").hidden = false;
    $("#d-empty").hidden = true;
    $("#d-result").innerHTML = `<div class="note">Prediction failed: ${esc(err.message)}</div>`;
  } finally {
    btn.classList.remove("loading");
  }
}

function gauge(prob, positive) {
  const C = 408.4;                        // 2*pi*65, matches .fill dasharray
  const cls = positive ? "pos" : "neg";
  return `
    <div class="gauge">
      <svg width="158" height="158" viewBox="0 0 158 158">
        <circle class="track" cx="79" cy="79" r="65" fill="none" stroke-width="12"/>
        <circle class="fill" cx="79" cy="79" r="65" fill="none" stroke-width="12"
                stroke="var(--${positive ? "danger" : "ok"})"
                stroke-dashoffset="${C}"/>
      </svg>
      <div class="gauge-mid">
        <div class="gauge-val ${cls}" id="gauge-num">0.0%</div>
        <div class="gauge-lbl">probability</div>
      </div>
    </div>`;
}

function renderVerdict(r) {
  const cls = r.positive ? "pos" : "neg";
  $("#d-empty").hidden = true;
  const box = $("#d-result");
  box.hidden = false;
  box.innerHTML = `
    <div class="verdict ${cls}">
      ${gauge(r.probability, r.positive)}
      <div class="verdict-body">
        <div class="kicker">Final weighted verdict</div>
        <h3 class="${cls}">${esc(r.verdict)}</h3>
        <div class="chips">
          <span class="chip" style="animation-delay:.30s">Agreement
            <b>${r.agreement}/${r.n_models}</b></span>
          <span class="chip" style="animation-delay:.38s">Combined surety
            <b>${pct(r.confidence, 0)}</b></span>
          <span class="chip" style="animation-delay:.46s">Baseline
            <b>${pct(r.baseline, 0)}</b></span>
        </div>
        <p class="muted small" style="margin-top:13px">
          The accuracy-weighted average of all ${r.n_models} probabilities —
          not a simple majority vote.</p>
      </div>
    </div>`;

  // second frame, so the transition has a "from" value to move away from
  nextTick(() => {
    const fill = box.querySelector(".fill");
    if (fill) fill.style.strokeDashoffset = String(408.4 * (1 - r.probability));
    countUp($("#gauge-num"), r.probability * 100, { suffix: "%", decimals: 1 });
  });
}

function renderVotes(r) {
  const panel = $("#d-votes-panel");
  panel.hidden = false;
  $("#d-vote-note").textContent =
    `bar = that model's surety · weight = its share of the verdict`;

  $("#d-votes").innerHTML = r.models.map((m, i) => {
    const cls = m.votes_positive ? "pos" : "neg";
    const grad = m.votes_positive
      ? "linear-gradient(90deg,var(--danger-dim),var(--danger))"
      : "linear-gradient(90deg,var(--ok),var(--accent))";
    return `
      <div class="vote" style="animation-delay:${i * 0.045}s" title="${esc(m.about || "")}">
        <div class="vote-top">
          <div><span class="vote-name">${esc(m.algorithm)}</span>
               <span class="vote-fam">${esc(m.family || "")}</span></div>
          <div class="vote-call ${cls}">${esc(m.vote)} · ${m.surety.toFixed(0)}%</div>
        </div>
        <div class="vote-bar"><i data-w="${Math.max(m.surety, 2)}"
             style="background:${grad}"></i></div>
        <div class="vote-sub">
          <span>accuracy ${m.accuracy.toFixed(1)}%</span>
          <span>weight ${m.weight.toFixed(1)}%</span>
        </div>
      </div>`;
  }).join("");

  nextTick(() => {
    $$("#d-votes .vote-bar i").forEach((el, i) => {
      el.style.transitionDelay = `${i * 0.045}s`;
      el.style.transform = `scaleX(${(+el.dataset.w) / 100})`;
    });
  });
}

/* --------------------------------------------------------------- imaging */
async function openImaging(key) {
  const data = await api(`/api/imaging/${key}`);
  state.imaging = data;
  const s = data.summary;

  $("#i-title").textContent = `${s.icon}  ${s.name}`;
  $("#i-blurb").textContent = s.blurb;
  $("#i-metrics").innerHTML = [
    ["Algorithms", s.algorithms],
    ["Classes", s.classes.length],
    ["Backbone", s.backbone],
  ].map(([k, v], i) => `<div class="chip" style="animation-delay:${i * .06}s">
      ${esc(k)} <b>${esc(v)}</b></div>`).join("");

  $("#i-preview").hidden = true;
  $("#i-drop").querySelector(".dz-inner").style.display = "";
  $("#i-empty").hidden = false;
  $("#i-result").hidden = true;
  $("#i-votes-panel").hidden = true;
  setActiveNav(key);
  showView("imaging");
}

async function uploadScan(file) {
  if (!file || !state.imaging) return;
  const drop = $("#i-drop");
  const preview = $("#i-preview");
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  drop.querySelector(".dz-inner").style.display = "none";
  drop.classList.add("scanning");

  const fd = new FormData();
  fd.append("file", file);
  try {
    const r = await api(`/api/imaging/${state.imaging.summary.key}/predict`,
                        { method: "POST", body: fd });
    renderImageResult(r);
  } catch (err) {
    $("#i-empty").hidden = true;
    $("#i-result").hidden = false;
    $("#i-result").innerHTML = `<div class="note">Failed: ${esc(err.message)}</div>`;
  } finally {
    drop.classList.remove("scanning");
  }
}

function renderImageResult(r) {
  const positive = !r.healthy;
  const cls = positive ? "pos" : "neg";
  $("#i-empty").hidden = true;
  const box = $("#i-result");
  box.hidden = false;
  box.innerHTML = `
    <div class="verdict ${cls}">
      ${gauge(r.probability, positive)}
      <div class="verdict-body">
        <div class="kicker">Final weighted verdict</div>
        <h3 class="${cls}">${esc(r.pretty)}</h3>
        <div class="chips">
          <span class="chip" style="animation-delay:.30s">Agreement
            <b>${r.agreement}/${r.n_models}</b></span>
          <span class="chip" style="animation-delay:.38s">
            <b>${r.classes.length}</b>-way</span>
        </div>
      </div>
    </div>
    <div class="section-title">Class probabilities</div>
    ${r.classes.map((c, i) => `
      <div class="vote" style="animation-delay:${i * .05}s">
        <div class="vote-top">
          <span class="vote-name">${esc(c.replace(/_/g, " "))}</span>
          <span class="vote-call">${(r.probabilities[i] * 100).toFixed(1)}%</span>
        </div>
        <div class="vote-bar"><i data-w="${Math.max(r.probabilities[i] * 100, 1)}"
          style="background:linear-gradient(90deg,var(--primary),var(--accent))"></i></div>
      </div>`).join("")}`;

  nextTick(() => {
    const fill = box.querySelector(".fill");
    if (fill) fill.style.strokeDashoffset = String(408.4 * (1 - r.probability));
    countUp($("#gauge-num"), r.probability * 100, { suffix: "%", decimals: 1 });
    $$("#i-result .vote-bar i").forEach((el, i) => {
      el.style.transitionDelay = `${i * 0.05}s`;
      el.style.transform = `scaleX(${(+el.dataset.w) / 100})`;
    });
  });

  const panel = $("#i-votes-panel");
  panel.hidden = false;
  $("#i-votes").innerHTML = r.models.map((m, i) => `
    <div class="vote" style="animation-delay:${i * .045}s">
      <div class="vote-top">
        <span class="vote-name">${esc(m.algorithm)}</span>
        <span class="vote-call ${m.agrees ? "neg" : "pos"}">
          ${esc(String(m.vote).replace(/_/g, " "))} · ${m.surety.toFixed(0)}%</span>
      </div>
      <div class="vote-bar"><i data-w="${Math.max(m.surety, 2)}"
        style="background:${m.agrees
          ? "linear-gradient(90deg,var(--ok),var(--accent))"
          : "linear-gradient(90deg,var(--danger-dim),var(--danger))"}"></i></div>
      <div class="vote-sub"><span>accuracy ${m.accuracy.toFixed(1)}%</span>
        <span>weight ${m.weight.toFixed(1)}%</span></div>
    </div>`).join("");

  nextTick(() => {
    $$("#i-votes .vote-bar i").forEach((el, i) => {
      el.style.transitionDelay = `${i * 0.045}s`;
      el.style.transform = `scaleX(${(+el.dataset.w) / 100})`;
    });
  });
}

function initDropzone() {
  const drop = $("#i-drop"), input = $("#i-file");
  drop.addEventListener("click", () => input.click());
  input.addEventListener("change", () => uploadScan(input.files[0]));
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => uploadScan(e.dataTransfer.files[0]));
}

/* ----------------------------------------------------------- performance */
async function openPerformance() {
  setActiveNav("performance");
  showView("performance");
  const data = await api("/api/performance");

  if (data.tabular?.summary) {
    const rows = data.tabular.summary;
    const max = Math.max(...rows.map((r) => r["ROC-AUC"] ?? 0));
    $("#perf-tabular").innerHTML = `
      <div class="panel-head"><h3>Clinical data — weighted ensemble</h3></div>
      <div class="table-wrap"><table>
        <thead><tr>
          <th>Condition</th><th class="num">Samples</th><th class="num">Baseline</th>
          <th class="num">Accuracy</th><th class="num">Recall</th>
          <th class="num">ROC-AUC</th><th>Relative</th>
        </tr></thead>
        <tbody>${rows.map((r, i) => `
          <tr style="animation:fadeUp .4s var(--ease) ${i * .05}s both">
            <td><b>${esc(r.Disease)}</b></td>
            <td class="num">${r.Samples}</td>
            <td class="num">${(r.Baseline * 100).toFixed(1)}%</td>
            <td class="num">${(r.Accuracy * 100).toFixed(1)}%</td>
            <td class="num">${(r.Recall * 100).toFixed(1)}%</td>
            <td class="num"><b>${r["ROC-AUC"].toFixed(3)}</b></td>
            <td class="bar-cell"><i data-w="${(r["ROC-AUC"] / max) * 100}"></i>
              <span></span></td>
          </tr>`).join("")}</tbody>
      </table></div>
      <div class="note">Ensemble weights are set by cross-validation on the
        training split only, so these held-out figures are not contaminated by
        the procedure that produced them.</div>`;

    nextTick(() => {
      $$("#perf-tabular .bar-cell i").forEach((el, i) => {
        el.style.transitionDelay = `${i * 0.06}s`;
        el.style.transform = `translateY(-50%) scaleX(${(+el.dataset.w) / 100})`;
      });
    });
  }

  if (data.imaging) {
    $("#perf-image-panel").hidden = false;
    const rows = Object.entries(data.imaging);
    $("#perf-image").innerHTML = `
      <div class="panel-head"><h3>Medical imaging — CNN embedding + 10 models</h3></div>
      <div class="table-wrap"><table>
        <thead><tr><th>Task</th><th class="num">Test images</th>
          <th class="num">Ensemble</th><th class="num">Macro F1</th>
          <th>CV-picked model</th><th class="num">Its accuracy</th>
          <th class="num">Plain CNN</th></tr></thead>
        <tbody>${rows.map(([k, r], i) => `
          <tr style="animation:fadeUp .4s var(--ease) ${i * .06}s both">
            <td><b>${esc(k.replace(/_/g, " "))}</b></td>
            <td class="num">${r.n_test}</td>
            <td class="num"><b>${(r.test_ensemble.accuracy * 100).toFixed(1)}%</b></td>
            <td class="num">${(r.test_ensemble.f1_macro * 100).toFixed(1)}%</td>
            <td>${esc(r.cv_selected_model.name)}</td>
            <td class="num">${(r.cv_selected_model.test_accuracy * 100).toFixed(1)}%</td>
            <td class="num">${(r.plain_cnn_test_accuracy * 100).toFixed(1)}%</td>
          </tr>`).join("")}</tbody>
      </table></div>
      <div class="note">The ensemble does not always win here — on pneumonia a
        plain fine-tuned CNN scores higher. The ten-model approach buys
        robustness and an auditable breakdown, not guaranteed peak accuracy.</div>`;
  }
}

/* ------------------------------------------------------------------ boot */
async function main() {
  initTheme();
  initDropzone();
  state.helix = initHelix($("#helix"));

  $("#d-run").addEventListener("click", runPrediction);
  $("#d-reset").addEventListener("click", resetForm);
  $$("[data-view]").forEach((el) => el.addEventListener("click", () => {
    if (el.dataset.view === "performance") openPerformance();
  }));
  $("[data-goto='first-disease']")?.addEventListener("click", () => {
    const first = state.catalog?.tabular?.[0];
    if (first) openDisease(first.key);
  });

  try {
    state.catalog = await api("/api/catalog");
    renderNav();
    renderHome();
  } catch (err) {
    $("#boot").innerHTML =
      `<div class="boot-inner"><p class="boot-text">Could not reach the API.<br>
       <span class="mono small">${esc(err.message)}</span></p></div>`;
    return;
  }

  $("#shell").hidden = false;
  nextTick(() => $("#boot").classList.add("gone"));
}

main();

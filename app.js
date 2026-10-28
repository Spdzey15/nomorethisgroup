/* Keystay — hotel cancellation intelligence.
   All analysis runs in the browser on the embedded Kaggle "Hotel booking demand" data (data.js). */
(() => {
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const fmt = (x) => Math.round(x).toLocaleString("en-IN");
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function inr(x) {
  const a = Math.abs(x), s = x < 0 ? "−₹" : "₹";
  if (a >= 1e7) return s + (a / 1e7).toFixed(2) + " Cr";
  if (a >= 1e5) return s + (a / 1e5).toFixed(1) + " L";
  return s + fmt(a);
}

// ---------- storage (guarded; private windows can throw) ----------
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
  del(k) { try { localStorage.removeItem(k); } catch {} },
};
const ui = Object.assign({ fHotel: "all", fYear: "all", fx: 95, thr: 0.5, simThr: 0.5, rec: 30, tier: "risk" }, store.get("folio_ui", {}));
const save = () => store.set("folio_ui", ui);

function toast(m) { const t = $("#toast"); t.textContent = m; t.classList.add("show"); clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove("show"), 2400); }

// =====================================================================
// DATA
// =====================================================================
// Arrival years are displayed seven years later (2015–2017 shown as 2022–2024); the data itself is unchanged.
const YEAR_SHIFT = 7;
const H = window.HOTEL, DICT = H.dicts, N = H.n;
const C = {}; for (const k in H.cols) C[k] = H.cols[k].split(",").map(Number);
const ROWS = Array.from({ length: N }, (_, i) => {
  const r = { i }; for (const k in C) r[k] = C[k][i];
  r.y += YEAR_SHIFT;
  r.nights = r.wn + r.nn; r.value = r.adr * Math.max(1, r.nights);
  return r;
});
const di = (d, name) => DICT[d].indexOf(name);
const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const rate = (rows) => rows.length ? rows.reduce((s, r) => s + r.c, 0) / rows.length : 0;

function filtered() {
  return ROWS.filter((r) => (ui.fHotel === "all" || r.h === +ui.fHotel) && (ui.fYear === "all" || r.y === +ui.fYear));
}
const filterLabel = () => `${ui.fHotel === "all" ? "Both hotels" : DICT.hotel[+ui.fHotel]} · ${ui.fYear === "all" ? "2022–2024" : ui.fYear}`;

// =====================================================================
// MODEL — logistic regression, standardised features, seeded SGD
// =====================================================================
const oneHot = (dict, col, skip = []) => DICT[dict].map((v, j) => ({ v, j })).filter(({ v, j }) => j > 0 && !skip.includes(v))
  .map(({ v, j }) => ({ key: `${col}_${j}`, label: `${dict === "segment" ? "Segment" : dict === "channel" ? "Channel" : dict === "ctype" ? "Customer" : dict === "meal" ? "Meal" : "Deposit"}: ${v}`, f: (r) => (r[col] === j ? 1 : 0) }));

const P1 = [
  { key: "resort", label: "Resort hotel", f: (r) => r.h },
  { key: "lead", label: "Lead time (log days)", f: (r) => Math.log1p(r.lt) },
  { key: "ssin", label: "Season (sin)", f: (r) => Math.sin((2 * Math.PI * r.m) / 12) },
  { key: "scos", label: "Season (cos)", f: (r) => Math.cos((2 * Math.PI * r.m) / 12) },
  { key: "wn", label: "Weekend nights", f: (r) => Math.min(r.wn, 8) },
  { key: "nn", label: "Week nights", f: (r) => Math.min(r.nn, 20) },
  { key: "ad", label: "Adults", f: (r) => Math.min(r.ad, 4) },
  { key: "ch", label: "Children", f: (r) => Math.min(r.ch, 3) },
  { key: "adr", label: "Room rate (ADR)", f: (r) => r.adr },
  ...oneHot("meal", "meal", ["Undefined"]),
];
const P2 = [
  ...P1,
  ...oneHot("segment", "seg", ["Undefined"]),
  ...oneHot("channel", "dch", ["Undefined"]),
  ...oneHot("deposit", "dep"),
  ...oneHot("ctype", "ct"),
  { key: "rep", label: "Repeat guest", f: (r) => r.rep },
  { key: "pc", label: "Previous cancellations", f: (r) => Math.min(r.pc, 5) },
  { key: "pb", label: "Previous stays kept", f: (r) => Math.log1p(Math.min(r.pb, 30)) },
  { key: "ag", label: "Booked via agent", f: (r) => r.ag },
  { key: "comp", label: "Company booking", f: (r) => r.comp },
  { key: "wl", label: "Was on waiting list", f: (r) => (r.wl > 0 ? 1 : 0) },
  { key: "pk", label: "Needs parking", f: (r) => (r.pk > 0 ? 1 : 0) },
  { key: "sr", label: "Special requests", f: (r) => Math.min(r.sr, 5) },
];

function mulberry32(a) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

function train(rows, specs, { epochs = 6, lr0 = 0.05, l2 = 1e-5 } = {}) {
  const k = specs.length, n = rows.length, X = new Float32Array(n * k);
  rows.forEach((r, i) => specs.forEach((s, j) => (X[i * k + j] = s.f(r))));
  const mu = new Float64Array(k), sd = new Float64Array(k);
  for (let j = 0; j < k; j++) { let m = 0; for (let i = 0; i < n; i++) m += X[i * k + j]; mu[j] = m / n; }
  for (let j = 0; j < k; j++) { let v = 0; for (let i = 0; i < n; i++) v += (X[i * k + j] - mu[j]) ** 2; sd[j] = Math.sqrt(v / n) || 1; }
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) X[i * k + j] = (X[i * k + j] - mu[j]) / sd[j];
  const y = rows.map((r) => r.c), base = y.reduce((a, b) => a + b, 0) / n;
  const w = new Float64Array(k); let b = Math.log(base / (1 - base));
  const order = Array.from({ length: n }, (_, i) => i), rnd = mulberry32(42);
  for (let e = 0; e < epochs; e++) {
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    const lr = lr0 / (1 + e);
    for (const i of order) {
      let z = b; const o = i * k;
      for (let j = 0; j < k; j++) z += w[j] * X[o + j];
      const g = 1 / (1 + Math.exp(-z)) - y[i];
      b -= lr * g;
      for (let j = 0; j < k; j++) w[j] -= lr * (g * X[o + j] + l2 * w[j]);
    }
  }
  const zrow = (r) => specs.map((s, j) => (s.f(r) - mu[j]) / sd[j]);
  const predict = (r) => { const z = zrow(r); let t = b; for (let j = 0; j < k; j++) t += w[j] * z[j]; return 1 / (1 + Math.exp(-t)); };
  const explain = (r) => zrow(r).map((z, j) => ({ key: specs[j].key, raw: specs[j].f(r), c: w[j] * z }));
  return { specs, w, b, predict, explain };
}

function evaluate(probs, ys, thr) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  probs.forEach((p, i) => { const h = p >= thr, y = ys[i]; if (h && y) tp++; else if (h) fp++; else if (y) fn++; else tn++; });
  const precision = tp / (tp + fp || 1), recall = tp / (tp + fn || 1);
  return { tp, fp, tn, fn, precision, recall, accuracy: (tp + tn) / probs.length, f1: (2 * precision * recall) / (precision + recall || 1) };
}
function rocAuc(probs, ys) {
  const idx = probs.map((p, i) => i).sort((a, b) => probs[b] - probs[a]);
  const P = ys.reduce((a, b) => a + b, 0), Nn = ys.length - P;
  let t = 0, f = 0, auc = 0; const pts = [{ x: 0, y: 0 }], step = Math.max(1, Math.floor(idx.length / 400));
  idx.forEach((i, n) => { if (ys[i]) t++; else { f++; auc += t / P / Nn; } if (n % step === 0) pts.push({ x: f / Nn, y: t / P }); });
  pts.push({ x: 1, y: 1 });
  return { auc, pts };
}

let M = null;
function models() {
  if (M) return M;
  const tr = ROWS.filter((r) => r.y <= 2023), te = ROWS.filter((r) => r.y === 2024);
  const m1 = train(tr, P1), m2 = train(tr, P2);
  const ys = te.map((r) => r.c), p1 = te.map(m1.predict), p2 = te.map(m2.predict);
  M = { tr, te, m1, m2, ys, p1, p2, r1: rocAuc(p1, ys), r2: rocAuc(p2, ys) };
  return M;
}

const TIERS = [
  { key: "safe", name: "Secure", color: "#2f6b4f", wash: "#e9f2ed", lo: 0, hi: 0.3, action: "No intervention. Use the stay to upsell — a room upgrade or breakfast at a member price." },
  { key: "watch", name: "Watch", color: "#b9821a", wash: "#fbf2de", lo: 0.3, hi: 0.6, action: "Send a reconfirmation reminder 7 days before arrival and offer a free date change instead of a cancellation." },
  { key: "risk", name: "At risk", color: "#c2462b", wash: "#fbece7", lo: 0.6, hi: 1.01, action: "Ask for a deposit or reconfirmation, count the room in the overbooking buffer, and send the \"confirm & get breakfast\" offer." },
];
const tierOf = (p) => TIERS.find((t) => p >= t.lo && p < t.hi);

// =====================================================================
// AUTH (demo — accounts live only in this browser)
// =====================================================================
async function hashPw(pw, salt) {
  if (window.crypto?.subtle) {
    const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(pw), "PBKDF2", false, ["deriveBits"]);
    const bits = await crypto.subtle.deriveBits({ name: "PBKDF2", hash: "SHA-256", salt: new TextEncoder().encode(salt), iterations: 120000 }, key, 256);
    return [...new Uint8Array(bits)].map((b) => b.toString(16).padStart(2, "0")).join("");
  }
  let h = 2166136261; for (const ch of salt + pw) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619); } return "f" + (h >>> 0).toString(16);
}
const session = () => store.get("folio_session", null);
function fail(form, msg) { $(".form-error", form).textContent = msg; if (msg) { form.classList.remove("shake"); void form.offsetWidth; form.classList.add("shake"); } }

function initAuth() {
  $$(".eye").forEach((b) => b.addEventListener("click", () => { const i = b.previousElementSibling, s = i.type === "password"; i.type = s ? "text" : "password"; b.textContent = s ? "Hide" : "Show"; }));
  const su = $("#signupForm"), li = $("#loginForm");
  su.password.addEventListener("input", () => {
    const p = su.password.value; let s = 0;
    if (p.length >= 8) s++; if (/\d/.test(p) && /[a-z]/i.test(p)) s++; if (/[A-Z]/.test(p) && /[a-z]/.test(p)) s++; if (p.length >= 12 || /[^a-z0-9]/i.test(p)) s++;
    $$(".meter span", su).forEach((el, i) => (el.style.background = i < s ? ["#c2462b", "#b9821a", "#a6782b", "#2f6b4f"][s - 1] : ""));
  });
  su.addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = su.name.value.trim(), email = su.email.value.trim().toLowerCase(), pw = su.password.value;
    if (name.length < 2) return fail(su, "Please enter your name.");
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return fail(su, "That email address doesn't look right.");
    if (pw.length < 8 || !/\d/.test(pw) || !/[a-z]/i.test(pw)) return fail(su, "Use 8+ characters with at least one letter and one number.");
    if (pw !== su.confirm.value) return fail(su, "The two passwords don't match.");
    const users = store.get("folio_users", {});
    if (users[email]) return fail(su, "An account with this email already exists — log in instead.");
    const salt = crypto.getRandomValues ? [...crypto.getRandomValues(new Uint8Array(12))].join("-") : String(Math.random());
    users[email] = { name, role: su.role.value, salt, hash: await hashPw(pw, salt) };
    store.set("folio_users", users);
    fail(su, ""); su.reset(); $$(".meter span", su).forEach((el) => (el.style.background = ""));
    start(email, users[email], `Welcome to Keystay, ${name.split(" ")[0]}.`);
  });
  li.addEventListener("submit", async (e) => {
    e.preventDefault();
    const email = li.email.value.trim().toLowerCase(), u = store.get("folio_users", {})[email];
    if (!u || (await hashPw(li.password.value, u.salt)) !== u.hash) return fail(li, "Email or password is incorrect.");
    fail(li, ""); li.reset(); start(email, u, `Welcome back, ${u.name.split(" ")[0]}.`);
  });
  $("#logout").addEventListener("click", () => { store.del("folio_session"); toast("You're logged out."); location.hash = "#/login"; });
}
function start(email, u, msg) {
  store.set("folio_session", { email, name: u.name, role: u.role });
  const next = ui.intended || ui.last || "app/overview"; delete ui.intended; save();
  location.hash = "#/" + next; toast(msg);
}

// =====================================================================
// ROUTER — the URL hash is the source of truth, so refresh returns to the same page
// =====================================================================
const VIEWS = ["overview", "drivers", "channels", "model", "scorer", "revenue", "playbook", "upload"];
const FILTERED = new Set(["overview", "drivers", "channels"]);
const drawn = {};

function route() {
  const path = location.hash.replace(/^#\/?/, ""), s = session();
  if (!path) return location.replace("#/" + (s ? ui.last || "app/overview" : "login"));
  if (path.startsWith("app/") && !s) { ui.intended = path; save(); return location.replace("#/login"); }
  if ((path === "login" || path === "signup") && s) return location.replace("#/" + (ui.last || "app/overview"));

  if (path === "login" || path === "signup") {
    $("#app").classList.add("hidden"); $("#auth").classList.remove("hidden");
    $("#loginForm").classList.toggle("hidden", path !== "login"); $("#signupForm").classList.toggle("hidden", path !== "signup");
    document.title = (path === "login" ? "Log in" : "Create account") + " · Keystay";
    countUp($("#auth"));
    return;
  }
  const view = path.split("/")[1];
  if (!VIEWS.includes(view)) return location.replace("#/app/overview");
  ui.last = path; save();
  $("#auth").classList.add("hidden"); $("#app").classList.remove("hidden");
  $("#meName").textContent = s.name; $("#meRole").textContent = s.role || s.email;
  $("#avatar").textContent = s.name.trim()[0].toUpperCase();
  $$(".view").forEach((v) => v.classList.toggle("active", v.dataset.view === view));
  $$("[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === view));
  closeMenu();
  document.title = $(`[data-nav=${view}]`).textContent.trim() + " · Keystay";
  window.scrollTo(0, 0);
  const key = FILTERED.has(view) ? view + filterLabel() + ui.fx : view;
  if (drawn[view] !== key) { RENDER[view](); drawn[view] = key; }
}
function closeMenu() { $(".side").classList.remove("open"); $("#scrim").classList.remove("show"); }

// =====================================================================
// CHARTS
// =====================================================================
const COL = { ink: "#18213a", clay: "#c2462b", sage: "#2f6b4f", brass: "#a6782b", amber: "#b9821a", ink3: "#868b9c", rule: "rgba(24,33,58,0.07)" };
const clayA = (a) => `rgba(194,70,43,${a.toFixed(2)})`;
const OK = !!window.Chart;
if (OK) {
  Object.assign(Chart.defaults, { color: COL.ink3, borderColor: COL.rule, maintainAspectRatio: false });
  Chart.defaults.font.family = '"Plus Jakarta Sans", system-ui, -apple-system, "Segoe UI", sans-serif';
  Chart.defaults.plugins.legend.display = false;
  Object.assign(Chart.defaults.plugins.tooltip, { backgroundColor: COL.ink, padding: 10, cornerRadius: 8, titleFont: { weight: "600" }, displayColors: false });
}
const charts = {};
function chart(id, cfg) {
  const el = document.getElementById(id);
  if (!OK) { el.parentElement.innerHTML = '<p class="note">Charts need an internet connection (Chart.js).</p>'; return; }
  charts[id]?.destroy(); charts[id] = new Chart(el, cfg);
}
const pAxis = (max) => ({ beginAtZero: true, max, ticks: { callback: (v) => Math.round(v * 100) + "%" }, grid: { color: COL.rule } });
const noGrid = { grid: { display: false } };
const rateTip = (rows) => ({ callbacks: { label: (c) => ` ${pct(c.raw)} cancelled · ${fmt(rows[c.dataIndex])} bookings` } });
function rateBars(id, labels, groups, horizontal = false) {
  const vals = groups.map(rate), n = groups.map((g) => g.length), mx = Math.max(...vals) || 1;
  chart(id, { type: "bar", data: { labels, datasets: [{ data: vals, backgroundColor: vals.map((v) => clayA(0.22 + 0.78 * (v / mx))), borderRadius: 6, maxBarThickness: 46 }] },
    options: { indexAxis: horizontal ? "y" : "x", plugins: { tooltip: rateTip(n) }, scales: horizontal ? { x: pAxis(), y: noGrid } : { y: pAxis(), x: noGrid } } });
  return vals;
}

function countUp(root) {
  $$("[data-count]", root).forEach((el) => {
    const t = parseFloat(el.dataset.count), d = (el.dataset.count.split(".")[1] || "").length, suf = el.dataset.suffix || "", t0 = performance.now();
    const f = (now) => { const k = Math.min(1, (now - t0) / 1000), e = 1 - (1 - k) ** 3; el.textContent = (t * e).toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d }) + suf; if (k < 1) requestAnimationFrame(f); };
    requestAnimationFrame(f);
  });
}

// =====================================================================
// VIEWS
// =====================================================================
const RENDER = {
  overview() {
    const now = new Date().getHours();
    $("#greet").textContent = now < 12 ? "morning" : now < 17 ? "afternoon" : "evening";
    $("#hi").textContent = session().name.split(" ")[0];
    const rows = filtered(), canc = rows.filter((r) => r.c);
    const lostEur = canc.reduce((s, r) => s + r.value, 0), nightsLost = canc.reduce((s, r) => s + r.nights, 0);
    $("#kLost").textContent = inr(lostEur * ui.fx);
    $("#kLostSub").textContent = `€${fmt(lostEur)} in the source data · ${filterLabel()}`;
    $("#kSide").innerHTML = [
      [fmt(rows.length), "bookings"], [pct(rate(rows)), "cancellation rate"], [fmt(nightsLost), "room nights lost"],
    ].map(([b, s]) => `<div><b>${b}</b><small>${s}</small></div>`).join("");

    const keys = [...new Set(rows.map((r) => r.y * 100 + r.m))].sort((a, b) => a - b);
    const byK = keys.map((k) => rows.filter((r) => r.y * 100 + r.m === k));
    chart("cMonthly", { data: { labels: keys.map((k) => `${MON[(k % 100) - 1]} '${String(Math.floor(k / 100)).slice(2)}`),
      datasets: [
        { type: "line", label: "Cancellation rate", data: byK.map(rate), yAxisID: "r", borderColor: COL.brass, backgroundColor: COL.brass, tension: 0.35, pointRadius: 2.5, borderWidth: 2, order: 0 },
        { type: "bar", label: "Kept", data: byK.map((g) => g.length - g.filter((r) => r.c).length), backgroundColor: COL.ink, stack: "s", borderRadius: 3, order: 1 },
        { type: "bar", label: "Cancelled", data: byK.map((g) => g.filter((r) => r.c).length), backgroundColor: COL.clay, stack: "s", borderRadius: 3, order: 1 },
      ] },
      options: { interaction: { mode: "index", intersect: false }, plugins: { tooltip: { displayColors: true, callbacks: { label: (c) => ` ${c.dataset.label}: ${c.dataset.yAxisID === "r" ? pct(c.raw) : fmt(c.raw)}` } } },
        scales: { x: { stacked: true, grid: { display: false } }, y: { stacked: true, grid: { color: COL.rule }, ticks: { callback: (v) => fmt(v) } }, r: { position: "right", min: 0, max: 0.7, grid: { display: false }, ticks: { callback: (v) => Math.round(v * 100) + "%" } } } } });

    const far = rate(rows.filter((r) => r.lt > 180)), near = rate(rows.filter((r) => r.lt <= 7));
    const r0 = rate(rows.filter((r) => r.sr === 0)), r1 = rate(rows.filter((r) => r.sr > 0));
    const rep = rate(rows.filter((r) => r.rep)), nw = rate(rows.filter((r) => !r.rep));
    $("#insights").innerHTML = [
      [`${(far / (near || 1)).toFixed(1)}×`, "Long lead times are risky", `Bookings made 180+ days ahead cancel at ${pct(far)}, against ${pct(near)} for last-week bookings.`, COL.clay],
      [`${pct(r0, 0)}`, "Silent guests walk away", `Guests with no special requests cancel at ${pct(r0)}. Just one request cuts that to ${pct(r1)}.`, COL.amber],
      [`${pct(rep, 0)}`, "Loyal guests keep their word", `Repeat guests cancel at ${pct(rep)}, against ${pct(nw)} for first-timers. Loyalty programmes protect revenue.`, COL.sage],
    ].map(([n, h, p, c]) => `<div class="insight" style="--accent:${c}"><span class="big">${n}</span><h4>${h}</h4><p>${p}</p></div>`).join("");
  },

  drivers() {
    const rows = filtered();
    $("#driverFilter").textContent = filterLabel();
    const LB = [["0–7 days", 0, 7], ["8–30", 8, 30], ["31–90", 31, 90], ["91–180", 91, 180], ["181–365", 181, 365], ["365+", 366, 1e9]];
    const lv = rateBars("cLead", LB.map((b) => b[0]), LB.map(([, a, b]) => rows.filter((r) => r.lt >= a && r.lt <= b)));
    $("#nLead").innerHTML = `Risk climbs steadily with lead time, from <b>${pct(lv[0])}</b> to <b>${pct(lv[5])}</b>. Long-lead bookings are where deposits pay off.`;

    const segs = DICT.segment.map((v, j) => ({ v, g: rows.filter((r) => r.seg === j) })).filter((s) => s.v !== "Undefined" && s.g.length > 50).sort((a, b) => rate(b.g) - rate(a.g));
    rateBars("cSeg", segs.map((s) => s.v), segs.map((s) => s.g), true);
    $("#nSeg").innerHTML = `<b>${segs[0].v}</b> is the riskiest segment (${pct(rate(segs[0].g))}); <b>${segs[segs.length - 1].v}</b> the safest (${pct(rate(segs[segs.length - 1].g))}).`;

    const rv = rateBars("cReq", ["0", "1", "2", "3+"], [0, 1, 2, 3].map((k) => rows.filter((r) => (k === 3 ? r.sr >= 3 : r.sr === k))));
    $("#nReq").innerHTML = `Every request signals commitment: <b>${pct(rv[0])}</b> with none, <b>${pct(rv[3])}</b> with three or more.`;

    const dv = rateBars("cDep", DICT.deposit, DICT.deposit.map((_, j) => rows.filter((r) => r.dep === j)));
    $("#nDep").innerHTML = `<b>Data anomaly:</b> "Non Refund" bookings cancel at <b>${pct(dv[di("deposit", "Non Refund")])}</b>, the opposite of what you'd expect. Most of them are bulk group blocks, so we report this honestly rather than build the story on it.`;

    const hist = [["First-time guest", rows.filter((r) => !r.rep && r.pc === 0)], ["Repeat guest", rows.filter((r) => r.rep)], ["Cancelled before", rows.filter((r) => r.pc > 0)]];
    const hv = rateBars("cHist", hist.map((h) => h[0]), hist.map((h) => h[1]));
    $("#nHist").innerHTML = `Past behaviour repeats itself: guests who cancelled before cancel again at <b>${pct(hv[2])}</b>.`;

  },

  model() {
    const { tr, te, ys, p1, p2, m2 } = models();
    
    const e1 = evaluate(p1, ys, 0.5), e2 = evaluate(p2, ys, 0.5), base = rate(te);
    $("#metrics").innerHTML = `<thead><tr><th>Model</th><th>Features</th><th>Accuracy</th><th>Precision</th></tr></thead><tbody>
      <tr><td class="muted">Baseline — predict "nobody cancels"</td><td>0</td><td>${pct(1 - base)}</td><td>—</td></tr>
      <tr><td>Phase 1 — booking basics</td><td>${P1.length}</td><td>${pct(e1.accuracy)}</td><td>${pct(e1.precision)}</td></tr>
      <tr class="win"><td>Phase 2 — + guest history, channel, deposit</td><td>${P2.length}</td><td>${pct(e2.accuracy)}</td><td>${pct(e2.precision)}</td></tr></tbody>`;

    const list = (title, sub, specs) => `<div><h4>${title}</h4><span class="muted">${sub}</span><ol>${specs.map((f) => `<li>${esc(f.label)}</li>`).join("")}</ol></div>`;
    $("#featLists").innerHTML = list(`Phase 1: ${P1.length} booking basics`, "What the booking form alone tells us", P1)
      + list(`Phase 2: ${P2.length} features`, `All ${P1.length} Phase 1 features plus these ${P2.length - P1.length} guest and channel signals`, P2.slice(P1.length));

    const thr = $("#thr"); thr.value = ui.thr;
    const cm = () => {
      ui.thr = +thr.value; save(); const e = evaluate(p2, ys, ui.thr);
      $("#thrLbl").textContent = `Threshold ${ui.thr.toFixed(2)}`;
      $("#cm").innerHTML = `<div class="h"></div><div class="h">Predicted kept</div><div class="h">Predicted cancel</div>
        <div class="h">Actually kept</div><div class="ok"><b>${fmt(e.tn)}</b>correct</div><div class="bad"><b>${fmt(e.fp)}</b>false alarm</div>
        <div class="h">Actually cancelled</div><div class="bad"><b>${fmt(e.fn)}</b>missed</div><div class="ok"><b>${fmt(e.tp)}</b>caught</div>`;
      $("#cmNote").innerHTML = `At ${ui.thr.toFixed(2)}, <b>${pct(e.precision)}</b> of the bookings the model flags really cancel. A lower threshold catches more but asks more loyal guests for deposits.`;
    };
    thr.oninput = cm; cm();

    const co = m2.specs.map((s, j) => ({ l: s.label, w: m2.w[j] })).filter((c) => Math.abs(c.w) > 0.03).sort((a, b) => b.w - a.w);
    chart("cCoef", { type: "bar", data: { labels: co.map((c) => c.l), datasets: [{ data: co.map((c) => c.w), backgroundColor: co.map((c) => (c.w > 0 ? COL.clay : COL.sage)), borderRadius: 4 }] },
      options: { indexAxis: "y", plugins: { tooltip: { callbacks: { label: (c) => ` ${c.raw > 0 ? "+" : ""}${c.raw.toFixed(2)} log-odds per SD` } } }, scales: { x: { grid: { color: COL.rule } }, y: { grid: { display: false }, ticks: { font: { size: 11 } } } } } });
  },

  channels() {
    const rows = filtered();
    $("#chFilter").textContent = filterLabel();
    const NAMES = { "Online TA": "Online travel agents", "Offline TA/TO": "Offline agents & tour operators", "Groups": "Group bookings", "Direct": "Direct (website & phone)",
      "Corporate": "Corporate accounts", "Complementary": "Complimentary stays", "Aviation": "Airline crew" };
    const verdict = (r) => (r < 0.2 ? ["Push", "Grow it with promotions", COL.sage, "#e9f2ed"] : r < 0.4 ? ["Monitor", "Keep it; add reminders", COL.amber, "#fbf2de"] : ["Restrict", "Ask for deposits or limit rooms", COL.clay, "#fbece7"]);
    const ch = DICT.segment.map((v, j) => ({ v, g: rows.filter((r) => r.seg === j) })).filter((c) => c.v !== "Undefined" && c.g.length)
      .map((c) => { const canc = c.g.filter((r) => r.c); return { ...c, name: NAMES[c.v] || c.v, n: c.g.length, rate: rate(c.g),
        lost: canc.reduce((a, r) => a + r.value, 0) * ui.fx, lead: c.g.reduce((a, r) => a + r.lt, 0) / c.g.length }; })
      .sort((a, b) => b.lost - a.lost);
    // Free stays and very small channels get no marketing verdict
    const NOVERDICT = { "Complementary": ["Not sold", "Free stays; no action", COL.ink3, "#f1f2f4"] };
    const verdictOf = (c) => NOVERDICT[c.v] || (c.n < 1000 ? ["Small", "Too few bookings to judge", COL.ink3, "#f1f2f4"] : verdict(c.rate));
    const total = rows.length, lostAll = ch.reduce((a, c) => a + c.lost, 0), maxRate = Math.max(...ch.map((c) => c.rate)) || 1;
    const big = ch.filter((c) => c.n >= 1000 && !NOVERDICT[c.v]);
    const best = [...big].sort((a, b) => a.rate - b.rate)[0], worst = [...big].sort((a, b) => b.rate - a.rate)[0];
    $("#chKpis").innerHTML = [[ch.length, "booking channels"], [inr(lostAll), "revenue lost across channels"],
      [best ? best.name : "—", best ? `most reliable · ${pct(best.rate)} cancel` : "most reliable"], [worst ? worst.name : "—", worst ? `riskiest · ${pct(worst.rate)} cancel` : "riskiest"]]
      .map(([b, t]) => `<div><b>${b}</b><small>${t}</small></div>`).join("");
    $("#chNote").textContent = `${fmt(total)} bookings · sorted by revenue lost`;
    $("#chTable").innerHTML = `<thead><tr><th>Channel</th><th class="r">Bookings</th><th class="r">Share</th><th>Cancellation rate</th><th class="r">Revenue lost</th><th class="r">Avg lead time</th><th>Verdict</th></tr></thead><tbody>` +
      ch.map((c) => { const [vd, adv, col, wash] = verdictOf(c);
        return `<tr><td><b>${c.name}</b><small class="sub-cell">${esc(c.v)}</small></td><td class="r">${fmt(c.n)}</td><td class="r">${pct(c.n / total)}</td>
          <td><span class="meter-cell"><span class="bar"><i style="width:${(c.rate / maxRate) * 100}%;background:${col}"></i></span><b>${pct(c.rate)}</b></span></td>
          <td class="r">${inr(c.lost)}</td><td class="r">${Math.round(c.lead)} days</td>
          <td><span class="tier-pill" style="background:${wash};color:${col}">${vd}</span><small class="sub-cell">${adv}</small></td></tr>`; }).join("") + "</tbody>";
    chart("cChan", { type: "bar", data: { labels: ch.map((c) => c.name), datasets: [{ data: ch.map((c) => c.lost), backgroundColor: ch.map((c) => verdictOf(c)[2]), borderRadius: 6, maxBarThickness: 30 }] },
      options: { indexAxis: "y", plugins: { tooltip: { callbacks: { label: (c) => ` ${inr(c.raw)} lost · ${pct(ch[c.dataIndex].rate)} cancel` } } },
        scales: { x: { grid: { color: COL.rule }, ticks: { callback: (v) => inr(v) } }, y: { grid: { display: false } } } } });
    const push = big.filter((c) => c.rate < 0.2).map((c) => c.name), restrict = big.filter((c) => c.rate >= 0.4).map((c) => c.name);
    $("#chTake").innerHTML = `<b>Marketing takeaway:</b> ${push.length ? `push <b>${push.join(", ")}</b>` : "no large channel is low-risk"}${restrict.length ? `; protect revenue from <b>${restrict.join(", ")}</b> with deposits or room limits` : ""}. Bar colour shows the verdict.`;
  },


  scorer() {
    const { m2 } = models();
    const st = ui.score = Object.assign({ h: 0, lt: 120, m: 8, wn: 1, nn: 3, ad: 2, ch: 0, adr: 110, meal: 0, seg: di("segment", "Online TA"), dch: di("channel", "TA/TO"),
      dep: di("deposit", "No Deposit"), ct: di("ctype", "Transient"), rep: 0, pc: 0, pb: 0, ag: 1, comp: 0, wl: 0, pk: 0, sr: 1, co: di("country", "GBR") }, ui.score || {});
    const sel = (k, label, dict, filt = () => true) => `<label>${label}<select data-k="${k}">${DICT[dict].map((v, j) => filt(v) ? `<option value="${j}">${esc(v)}</option>` : "").join("")}</select></label>`;
    const rng = (k, label, min, max, step, unit = "") => `<div class="slider"><label>${label}<output data-o="${k}"></output></label><input type="range" data-k="${k}" min="${min}" max="${max}" step="${step}" data-unit="${unit}"></div>`;
    const tog = (k, label) => `<button type="button" class="chip" data-t="${k}">${label}</button>`;
    $("#scoreForm").innerHTML = `
      <div class="full"><span class="seg">${DICT.hotel.map((v, j) => `<button type="button" class="chip" data-hotel="${j}">${v}</button>`).join("")}</span></div>
      ${rng("lt", "Lead time", 0, 500, 1, " days")}${rng("adr", "Room rate", 20, 400, 5, " €/night")}
      ${rng("nn", "Week nights", 0, 14, 1)}${rng("wn", "Weekend nights", 0, 6, 1)}
      ${rng("ad", "Adults", 1, 4, 1)}${rng("ch", "Children", 0, 3, 1)}
      ${rng("sr", "Special requests", 0, 5, 1)}${rng("pc", "Previous cancellations", 0, 5, 1)}
      <label>Arrival month<select data-k="m">${MON.map((v, j) => `<option value="${j + 1}">${v}</option>`).join("")}</select></label>
      ${sel("seg", "Market segment", "segment", (v) => v !== "Undefined")}${sel("dch", "Channel", "channel", (v) => v !== "Undefined")}
      ${sel("dep", "Deposit", "deposit")}${sel("ct", "Customer type", "ctype")}${sel("meal", "Meal plan", "meal", (v) => v !== "Undefined")}
      <div class="full"><span class="seg">${tog("rep", "Repeat guest")}${tog("ag", "Via travel agent")}${tog("pk", "Needs parking")}${tog("wl", "Was wait-listed")}</span></div>`;
    const paint = () => {
      $$("[data-k]", $("#scoreForm")).forEach((el) => (el.value = st[el.dataset.k]));
      $$("[data-o]", $("#scoreForm")).forEach((o) => { const el = $(`input[data-k=${o.dataset.o}]`); o.textContent = st[o.dataset.o] + el.dataset.unit; });
      $$("[data-hotel]").forEach((b) => b.classList.toggle("on", +b.dataset.hotel === st.h));
      $$("[data-t]").forEach((b) => b.classList.toggle("on", !!st[b.dataset.t]));
      save();
      const r = { ...st, nights: st.wn + st.nn }, p = m2.predict(r), t = tierOf(p);
      $("#ktPct").textContent = pct(p, 0); $("#ktPct").style.color = t.color;
      $("#bigTag").style.borderLeftColor = t.color;
      $("#ktMeta").textContent = `${DICT.hotel[st.h]} · ${st.lt} days ahead · ${st.wn + st.nn} nights · €${st.adr}/night`;
      const tier = $("#ktTier"); tier.textContent = t.name; tier.style.background = t.wash; tier.style.color = t.color;
      const ex = m2.explain(r).map((e) => ({ ...e, text: describe(e, r) })).filter((e) => e.text).sort((a, b) => b.c - a.c);
      const up = ex.filter((e) => e.c > 0.15).slice(0, 3).map((e) => e.text), dn = ex.filter((e) => e.c < -0.15).slice(-3).reverse().map((e) => e.text);
      const value = st.adr * Math.max(1, st.wn + st.nn) * ui.fx;
      $("#ktAction").innerHTML = `<h3>Recommended action</h3><p>${t.action}</p><ul>
        ${up.length ? `<li>Pushing risk up: ${up.map(esc).join(", ")}</li>` : ""}${dn.length ? `<li>Holding risk down: ${dn.map(esc).join(", ")}</li>` : ""}
        <li>Booking value at stake: <b>${inr(value)}</b> · expected loss ${inr(value * p)}</li></ul>`;
    };
    const form = $("#scoreForm");
    form.oninput = (e) => { const k = e.target.dataset.k; if (k) { st[k] = +e.target.value; paint(); } };
    form.onclick = (e) => {
      const b = e.target.closest("button"); if (!b) return;
      if (b.dataset.hotel) st.h = +b.dataset.hotel;
      if (b.dataset.t) st[b.dataset.t] = st[b.dataset.t] ? 0 : 1;
      paint();
    };
    paint();
  },

  revenue() {
    const { te, p2, ys } = models();
    const sT = $("#sThr"), sR = $("#sRec");
    sT.value = ui.simThr; sR.value = ui.rec;
    const curveT = Array.from({ length: 17 }, (_, i) => +(0.1 + i * 0.05).toFixed(2));
    const draw = () => {
      ui.simThr = +sT.value; ui.rec = Math.max(0, Math.min(100, +sR.value || 0)); save();
      $("#oThr").textContent = ui.simThr.toFixed(2);
      const s = simulate(ui.simThr, p2, ys, te);
      $("#simOut").innerHTML = `
        <div><small>Bookings flagged</small><b>${fmt(s.flagged)}</b><small>of ${fmt(te.length)} in 2024</small></div>
        <div><small>Cancellations caught</small><b>${fmt(s.tp)}</b><small>${pct(s.tp / s.cancels)} of all cancellations</small></div>
        <div class="net"><small>Revenue protected</small><b>${inr(s.protected)}</b><small>${ui.rec}% of the value of caught cancellations, 2024 season (Jan–Aug)</small></div>`;
      const pts = curveT.map((t) => simulate(t, p2, ys, te).protected);
      chart("cSim", { type: "line", data: { labels: curveT.map((t) => t.toFixed(2)), datasets: [{ data: pts, borderColor: COL.ink, borderWidth: 2, tension: 0.3, fill: { target: "origin", above: "rgba(47,107,79,.10)" },
        pointRadius: curveT.map((t) => (t === ui.simThr ? 6 : 0)), pointBackgroundColor: COL.brass, pointBorderColor: "#fff", pointBorderWidth: 2 }] },
        options: { animation: { duration: 300 }, plugins: { tooltip: { callbacks: { title: (c) => `Threshold ${c[0].label}`, label: (c) => ` Revenue protected ${inr(c.raw)}` } } },
          scales: { x: { title: { display: true, text: "Risk threshold" }, grid: { display: false } }, y: { beginAtZero: true, title: { display: true, text: "Revenue protected" }, grid: { color: COL.rule }, ticks: { callback: (v) => inr(v) } } } } });
      $("#nSim").innerHTML = `A lower threshold flags more bookings, so more revenue is protected, but more loyal guests are asked for a deposit. At <b>${ui.simThr.toFixed(2)}</b>, ${fmt(s.flagged)} bookings are flagged and <b>${inr(s.protected)}</b> is protected.`;
    };
    [sT, sR].forEach((el) => (el.oninput = draw)); draw();
  },

  playbook() {
    const { te, p2 } = models();
    $("#tiers").innerHTML = TIERS.map((t) => {
      const g = te.filter((_, i) => tierOf(p2[i]) === t), r = rate(g);
      return `<div class="tier" style="--c:${t.color}"><h3>${t.name}<span class="muted">${t.lo === 0 ? "<" + pct(t.hi, 0) : t.hi > 1 ? "≥" + pct(t.lo, 0) : pct(t.lo, 0) + "–" + pct(t.hi, 0)} risk</span></h3>
        <b class="n">${fmt(g.length)}</b><span class="muted">bookings · ${pct(g.length / te.length, 0)} of 2024</span>
        <div class="bar"><span style="width:${r * 100}%"></span></div><span class="muted">Actually cancelled: <b style="color:var(--ink)">${pct(r)}</b></span><p>${t.action}</p></div>`;
    }).join("");
  },

  upload() {
    models();
    const drop = $("#drop"), file = $("#file");
    const load = (text, name) => {
      try { scoreUpload(text, name); } catch (err) { $("#upMsg").textContent = "Couldn't read that file: " + err.message; $("#upResults").classList.add("hidden"); return; }
      try { sessionStorage.setItem("keystay_upload", JSON.stringify({ text, name })); } catch {}
    };
    file.onchange = () => { const f = file.files[0]; if (!f) return; $("#upMsg").textContent = `Reading ${f.name}…`; f.text().then((t) => load(t, f.name)); file.value = ""; };
    drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
    drop.ondragleave = () => drop.classList.remove("over");
    drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); const f = e.dataTransfer.files[0]; if (f) f.text().then((t) => load(t, f.name)); };
    $("#sample").onclick = () => load(sampleCsv(500), "sample_bookings.csv");
    $("#template").onclick = () => { const t = sampleCsv(3, false); download("keystay_template.csv", t); copyText(t, "Template downloaded and copied to clipboard."); };
    let saved = null; try { saved = JSON.parse(sessionStorage.getItem("keystay_upload")); } catch {}
    if (saved) load(saved.text, saved.name);
  },
};

// =====================================================================
// UPLOAD & SCORE — any CSV in the Kaggle hotel_bookings format
// =====================================================================
const FULLMON = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const UP_COLS = ["hotel", "is_canceled", "lead_time", "arrival_date_year", "arrival_date_month", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
  "meal", "country", "market_segment", "distribution_channel", "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled", "deposit_type", "agent", "company",
  "days_in_waiting_list", "customer_type", "adr", "required_car_parking_spaces", "total_of_special_requests"];
let UPLOAD = null;

function parseCsv(text) {
  const first = text.slice(0, text.indexOf("\n") + 1 || undefined), d = (first.match(/;/g) || []).length > (first.match(/,/g) || []).length ? ";" : ",";
  const rows = []; let row = [], f = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
    else if (c === '"') q = true;
    else if (c === d) { row.push(f); f = ""; }
    else if (c === "\n") { row.push(f); rows.push(row); row = []; f = ""; }
    else if (c !== "\r") f += c;
  }
  if (f !== "" || row.length) { row.push(f); rows.push(row); }
  return rows.filter((r) => r.some((v) => v.trim() !== ""));
}
const csvCell = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
function copyText(text, msg) {
  (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject()).then(() => toast(msg), () => toast("Copy isn't available here. Use the download button instead."));
}
function download(name, text) {
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([text], { type: "text/csv" })), download: name });
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// Turn one uploaded record into the same encoded shape the model was trained on.
function toRow(o) {
  const n = (k, d = 0) => { const v = parseFloat(o[k]); return Number.isFinite(v) ? v : d; };
  const has = (k) => o[k] != null && !["", "NULL", "NA", "nan"].includes(String(o[k]).trim());
  const cat = (dict, k) => { const v = String(o[k] ?? "").trim(), j = DICT[dict].indexOf(v); return j >= 0 ? j : DICT[dict].indexOf("Other") >= 0 ? DICT[dict].indexOf("Other") : -1; };
  const mRaw = String(o.arrival_date_month ?? "").trim(), mNum = parseInt(mRaw, 10);
  const m = mNum >= 1 && mNum <= 12 ? mNum : FULLMON.findIndex((x) => x.toLowerCase().startsWith(mRaw.toLowerCase().slice(0, 3)) && mRaw) + 1 || 7;
  const r = {
    h: /resort/i.test(o.hotel || "") ? 1 : 0, lt: Math.max(0, n("lead_time", 60)), y: n("arrival_date_year", 2024), m,
    wn: n("stays_in_weekend_nights", 1), nn: n("stays_in_week_nights", 2), ad: n("adults", 2), ch: n("children") + n("babies"),
    meal: Math.max(0, cat("meal", "meal")), co: cat("country", "country"), seg: cat("segment", "market_segment"), dch: cat("channel", "distribution_channel"),
    rep: n("is_repeated_guest"), pc: n("previous_cancellations"), pb: n("previous_bookings_not_canceled"),
    dep: has("deposit_type") ? cat("deposit", "deposit_type") : 0, ag: has("agent") ? 1 : 0, comp: has("company") ? 1 : 0,
    wl: n("days_in_waiting_list"), ct: has("customer_type") ? cat("ctype", "customer_type") : 0,
    adr: Math.max(0, Math.min(1000, n("adr", 100))), pk: n("required_car_parking_spaces"), sr: n("total_of_special_requests"),
  };
  r.nights = r.wn + r.nn; r.value = r.adr * Math.max(1, r.nights);
  if (has("is_canceled")) r.c = n("is_canceled") ? 1 : 0;
  return r;
}

function scoreUpload(text, name) {
  const grid = parseCsv(text);
  if (grid.length < 2) throw new Error("it needs a header row and at least one booking.");
  const head = grid[0].map((h) => h.trim().replace(/^﻿/, "").toLowerCase());
  const known = UP_COLS.filter((c) => head.includes(c));
  if (!known.includes("lead_time")) throw new Error("no lead_time column found — use the template's column names.");
  const recs = grid.slice(1).map((cells) => Object.fromEntries(head.map((h, i) => [h, cells[i] ?? ""])));
  const { m2 } = models();
  const rows = recs.map(toRow), probs = rows.map(m2.predict);
  UPLOAD = { name, head, recs, rows, probs };
  const missing = UP_COLS.filter((c) => c !== "is_canceled" && !head.includes(c));
  $("#upMsg").innerHTML = `Scored <b>${fmt(rows.length)}</b> bookings from <b>${esc(name)}</b>.` + (missing.length ? ` ${missing.length} expected column${missing.length > 1 ? "s were" : " was"} missing and filled with typical values: ${missing.map(esc).join(", ")}.` : " All model columns found.");
  renderUpload();
}

function renderUpload() {
  const { rows, probs, recs, head } = UPLOAD, n = rows.length;
  $("#upResults").classList.remove("hidden");
  const exp = probs.reduce((a, b) => a + b, 0), risk = rows.reduce((s, r, i) => s + r.value * probs[i], 0) * ui.fx;
  const flagged = probs.filter((p) => p >= 0.5).length;
  $("#uRisk").textContent = inr(risk);
  $("#uRiskSub").textContent = `Expected value of the room nights likely to be cancelled · €1 = ₹${ui.fx}`;
  $("#uSide").innerHTML = [[fmt(n), "bookings scored"], [fmt(Math.round(exp)), "expected cancellations"], [`${fmt(flagged)}`, `flagged at ≥ 50% risk (${pct(flagged / n, 0)})`]]
    .map(([b, s]) => `<div><b>${b}</b><small>${s}</small></div>`).join("");

  const counts = TIERS.map((t) => probs.filter((p) => tierOf(p) === t).length);
  chart("cUpTier", { type: "bar", data: { labels: TIERS.map((t) => t.name), datasets: [{ data: counts, backgroundColor: TIERS.map((t) => t.color), borderRadius: 6, maxBarThickness: 64 }] },
    options: { plugins: { tooltip: { callbacks: { label: (c) => ` ${fmt(c.raw)} bookings (${pct(c.raw / n)})` } } }, scales: { y: { beginAtZero: true, grid: { color: COL.rule }, ticks: { precision: 0, callback: (v) => fmt(v) } }, x: { grid: { display: false } } } } });

  const labelled = rows.every((r) => r.c != null);
  if (labelled && n > 1) {
    const ys = rows.map((r) => r.c), e = evaluate(probs, ys, 0.5), hasBoth = ys.some((y) => y) && ys.some((y) => !y), auc = hasBoth ? rocAuc(probs, ys).auc : null;
    $("#upEval").innerHTML = `<h3>How the model did on your file</h3><p class="note" style="margin:0 0 14px">Your file includes <code>is_canceled</code>, so predictions were compared with what actually happened (threshold 0.50).</p>
      <div class="eval"><div><small>Accuracy</small><b>${pct(e.accuracy)}</b></div><div><small>ROC-AUC</small><b>${auc == null ? "—" : auc.toFixed(3)}</b></div>
      <div><small>Precision</small><b>${pct(e.precision)}</b></div></div>
      <p class="note">Actual cancellation rate ${pct(rate(rows))} · model expected ${pct(exp / n)}.</p>`;
  } else {
    $("#upEval").innerHTML = `<h3>Recommended actions</h3>${TIERS.map((t, i) => `<p class="note" style="margin-top:12px"><span class="tier-pill" style="background:${t.wash};color:${t.color}">${t.name} · ${fmt(counts[i])}</span><br>${t.action}</p>`).join("")}
      <p class="note">Add an <code>is_canceled</code> column to also see how accurate the model is on your data.</p>`;
  }

  const order = probs.map((p, i) => i).sort((a, b) => probs[b] - probs[a]).slice(0, 100);
  $("#upCount").textContent = `Top ${order.length} of ${fmt(n)} by risk`;
  $("#upTable").innerHTML = `<thead><tr><th>Row</th><th>Hotel</th><th>Lead days</th><th>Arrival</th><th>Nights</th><th>ADR €</th><th>Segment</th><th>Deposit</th><th>Risk</th><th>Tier</th>${rows[0].c != null ? "<th>Actual</th>" : ""}</tr></thead><tbody>` +
    order.map((i) => { const r = rows[i], t = tierOf(probs[i]);
      return `<tr><td class="muted">${i + 1}</td><td>${DICT.hotel[r.h].replace(" Hotel", "")}</td><td>${r.lt}</td><td>${MON[r.m - 1]} ${r.y}</td><td>${r.nights}</td><td>${r.adr}</td>
        <td>${esc(recs[i].market_segment || "—")}</td><td>${esc(recs[i].deposit_type || "—")}</td><td><b>${pct(probs[i], 0)}</b></td>
        <td><span class="tier-pill" style="background:${t.wash};color:${t.color}">${t.name}</span></td>${r.c != null ? `<td>${r.c ? "Cancelled" : "Kept"}</td>` : ""}</tr>`; }).join("") + "</tbody>";

  const scored = () => {
    const out = [[...head, "cancel_probability", "risk_tier", "recommended_action"].map(csvCell).join(",")];
    recs.forEach((o, i) => { const t = tierOf(probs[i]); out.push([...head.map((h) => o[h]), probs[i].toFixed(4), t.name, t.action].map(csvCell).join(",")); });
    return out.join("\n");
  };
  $("#dl").onclick = () => download(UPLOAD.name.replace(/\.csv$/i, "") + "_scored.csv", scored());
  $("#dlCopy").onclick = () => copyText(scored(), "Scored CSV copied. Paste it into Excel or Google Sheets.");
}

// Rebuild bookings from the embedded dataset in the original Kaggle column format.
function sampleCsv(count, withOutcome = true) {
  const pool = ROWS.filter((r) => r.y === 2024), rnd = mulberry32(7), cols = withOutcome ? UP_COLS : UP_COLS.filter((c) => c !== "is_canceled");
  const lines = [cols.join(",")];
  for (let k = 0; k < count; k++) {
    const r = pool[Math.floor(rnd() * pool.length)];
    const rec = { hotel: DICT.hotel[r.h], is_canceled: r.c, lead_time: r.lt, arrival_date_year: r.y, arrival_date_month: FULLMON[r.m - 1], stays_in_weekend_nights: r.wn,
      stays_in_week_nights: r.nn, adults: r.ad, children: r.ch, babies: 0, meal: DICT.meal[r.meal], country: DICT.country[r.co], market_segment: DICT.segment[r.seg],
      distribution_channel: DICT.channel[r.dch], is_repeated_guest: r.rep, previous_cancellations: r.pc, previous_bookings_not_canceled: r.pb, deposit_type: DICT.deposit[r.dep],
      agent: r.ag ? 9 : "NULL", company: r.comp ? 40 : "NULL", days_in_waiting_list: r.wl, customer_type: DICT.ctype[r.ct], adr: r.adr,
      required_car_parking_spaces: r.pk, total_of_special_requests: r.sr };
    lines.push(cols.map((c) => csvCell(rec[c])).join(","));
  }
  return lines.join("\n");
}

// Plain-language reason for one feature's contribution, phrased from the booking's actual value.
function describe(e, r) {
  const NUM = { lead: `${r.lt} days lead time`, wn: `${r.wn} weekend nights`, nn: `${r.nn} week nights`, ad: `${r.ad} adults`, ch: r.ch ? `${r.ch} children` : "no children",
    adr: `€${r.adr}/night rate`, pc: `${r.pc} previous cancellations`, pb: `${r.pb} previous stays`, sr: r.sr ? `${r.sr} special requests` : "no special requests" };
  if (e.key in NUM) return e.key === "pc" && !r.pc ? "no past cancellations" : e.key === "pb" && !r.pb ? "no previous stays" : NUM[e.key];
  if (e.key === "ssin" || e.key === "scos" || !e.raw) return null;
  return P2.find((s) => s.key === e.key).label;
}

function simulate(t, probs, ys, rows) {
  let flagged = 0, tp = 0, caughtVal = 0, cancels = 0;
  probs.forEach((p, i) => { if (ys[i]) cancels++; if (p >= t) { flagged++; if (ys[i]) { tp++; caughtVal += rows[i].value; } } });
  return { flagged, tp, cancels, protected: caughtVal * ui.fx * (ui.rec / 100) };
}

// ---------- filters (shared by Overview and Drivers) ----------
function initFilters() {
  if (!["all", "2022", "2023", "2024"].includes(ui.fYear)) ui.fYear = "all";
  $("#fHotel").value = ui.fHotel; $("#fYear").value = ui.fYear; $("#fx").value = ui.fx;
  const apply = () => {
    ui.fHotel = $("#fHotel").value; ui.fYear = $("#fYear").value; ui.fx = Math.max(1, +$("#fx").value || 95); save();
    drawn.overview = drawn.drivers = drawn.channels = drawn.revenue = drawn.scorer = drawn.upload = null; route();
  };
  ["#fHotel", "#fYear"].forEach((s) => $(s).addEventListener("change", apply));
  $("#fx").addEventListener("change", apply);
}

// ---------- boot ----------
initAuth(); initFilters();
$("#menuBtn").addEventListener("click", () => { $(".side").classList.add("open"); $("#scrim").classList.add("show"); });
$("#scrim").addEventListener("click", closeMenu);
window.addEventListener("hashchange", route);
route();
$("#loading").classList.add("done");
})();

let data, frame = 0, playing = true, timer;
const $ = (id) => document.getElementById(id);
const number = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 1 });
const currency = new Intl.NumberFormat("en-GB", { style: "currency", currency: "GBP" });

async function load() {
  const live = new URLSearchParams(location.search).get("live") === "1";
  const sources = live ? ["latest.json", "demo-run.json"] : ["demo-run.json"];
  for (const source of sources) {
    try {
      const response = await fetch(`${source}?t=${Date.now()}`);
      if (!response.ok) continue;
      data = await response.json();
      $("mode").textContent = sourceLabel(source, live, data);
      break;
    } catch (_) { /* Try stable replay after unavailable live feed. */ }
  }
  if (!data) throw new Error(unavailableMessage(live));
  const run = data.optimized || data;
  $("mode").textContent += ` · ${providerLabel(run)}`;
  renderSummary();
  drawPower();
  drawSetpoints();
  start();
}

function providerLabel(run) {
  const metadata = run.metadata || {};
  const primary = metadata.primary_model || metadata.model || "unknown model";
  const provider = metadata.primary_provider || metadata.provider;
  const providerName = provider === "groq" ? "GroqCloud" : provider === "ollama" ? "Ollama" : "unknown provider";
  if (metadata.fallback_calls == null) return `${primary} via ${providerName}; fallback usage unavailable`;
  const fallbacks = Number(metadata.fallback_calls) || 0;
  return fallbacks
    ? `${primary} via ${providerName}; ${fallbacks} local fallback call${fallbacks === 1 ? "" : "s"}`
    : `${primary} via ${providerName}; no fallback calls`;
}

function sourceLabel(source, liveRequested, run) {
  if (source === "latest.json") return "Live simulation feed";
  const replay = run?.metadata?.verified === true
    ? "Verified 7-day EnergyPlus replay"
    : "7-day EnergyPlus replay (unverified)";
  return liveRequested ? `${replay}; live feed unavailable` : replay;
}

function unavailableMessage(liveRequested) {
  return liveRequested
    ? "Live feed and verified replay unavailable. Start the local server from the repository root."
    : "Verified replay unavailable. Start the local server from the repository root.";
}

function format(value, suffix = "") { return `${number.format(Number(value))}${suffix}`; }
function merged() {
  return data.optimized ? data : {
    optimized: data,
    baseline: data,
    comparison: {
      energy_change_pct: 0,
      cost_savings_gbp: 0,
      peak_change_pct: 0,
      carbon_change_pct: 0,
      pmv_compliance_delta_pct: 0,
      thermal_comfort_pass: false,
      comfort_budget_pass: false,
    },
  };
}

function renderSummary() {
  const d = merged(), o = d.optimized.summary, b = d.baseline.summary, c = d.comparison;
  const actions = d.optimized.actions?.length ?? 0;
  const errors = d.optimized.errors?.length ?? 0;
  const costSavings = c.cost_savings_gbp ?? (b.total_kwh - o.total_kwh) * .28;
  const thermalComfortPass = c.thermal_comfort_pass ?? (o.pmv_compliance_pct >= 95);
  const comfortBudgetPass = c.comfort_budget_pass
    ?? (thermalComfortPass && c.pmv_compliance_delta_pct >= -1);
  const cards = [
    ["Electricity", -c.energy_change_pct, "% saved"],
    ["Estimated cost", costSavings, `${currency.format(costSavings)} saved at £0.28/kWh`],
    ["Peak demand", -c.peak_change_pct, "% reduced"],
    ["Carbon", -c.carbon_change_pct, "% reduced"],
    ["Comfort boundary", o.pmv_compliance_pct, `% compliant; ${thermalComfortPass ? "PASS" : "MISS"}`],
    ["PMV vs baseline", c.pmv_compliance_delta_pct, ` points; strict −1.0 target ${comfortBudgetPass ? "PASS" : "MISS"}`],
    ["Validated actions", actions, " autonomous setpoint updates"],
    ["Runtime errors", errors, errors === 0 ? " clean run" : " inspect audit"],
  ];
  $("metrics").innerHTML = cards.map(([name, value, label]) =>
    `<article class="metric"><small>${name}</small><strong>${format(value)}</strong><span>${label}</span></article>`
  ).join("");
}

function points(values, max) {
  return values.map((value, index) => `${index / (values.length - 1 || 1) * 1000},${280 - value / max * 250}`).join(" ");
}

function drawPower() {
  const d = merged();
  const baseline = d.baseline.telemetry.map((sample) => sample.facility_demand_w);
  const optimized = d.optimized.telemetry.map((sample) => sample.facility_demand_w);
  const max = Math.max(...baseline, ...optimized, 1);
  $("power-chart").innerHTML = `<polyline class="baseline" points="${points(baseline, max)}"/><polyline class="optimized" points="${points(optimized, max)}"/>`;
}

function drawSetpoints() {
  const rows = merged().optimized.telemetry;
  const heat = rows.map((sample) => sample.current_heating_c);
  const cool = rows.map((sample) => sample.current_cooling_c);
  $("setpoint-chart").innerHTML = `<polyline class="baseline" points="${points(heat, 32)}"/><polyline class="optimized" points="${points(cool, 32)}"/>`;
}

function renderFrame() {
  const run = merged().optimized;
  const sample = run.telemetry[Math.min(frame, run.telemetry.length - 1)];
  if (!sample) return;
  $("clock").textContent = `Simulated hour ${sample.hour}`;
  $("zones").innerHTML = sample.zones.map((zone) =>
    `<article class="zone ${Math.abs(zone.pmv) <= .5 && zone.co2_ppm <= 1000 ? "ok" : "warn"}"><strong>${escapeHtml(zone.name)}</strong><dl><dt>Temperature</dt><dd>${format(zone.temperature_c, " °C")}</dd><dt>PMV</dt><dd>${format(zone.pmv)}</dd><dt>CO₂</dt><dd>${format(zone.co2_ppm, " ppm")}</dd><dt>Occupants</dt><dd>${format(zone.occupants)}</dd></dl></article>`
  ).join("");
  const events = run.audit.filter((event, index) => {
    if (Number.isFinite(event.hour)) return event.hour <= sample.hour;
    return Math.floor(index * run.telemetry.length / Math.max(run.audit.length, 1)) <= frame;
  }).slice(-8).reverse();
  $("audit").innerHTML = events.length
    ? events.map((event) => `<div class="event"><code>${escapeHtml(event.tool)}</code><div>${escapeHtml(JSON.stringify(event.result || event.input || {}).slice(0, 180))}</div></div>`).join("")
    : '<p class="empty">No agent action at this timestep.</p>';
}

function escapeHtml(text) { const paragraph = document.createElement("p"); paragraph.textContent = text; return paragraph.innerHTML; }
function start() {
  clearInterval(timer);
  timer = setInterval(() => {
    if (playing) {
      frame = (frame + 1) % merged().optimized.telemetry.length;
      renderFrame();
    }
  }, 1000 / Number($("speed").value));
  renderFrame();
}

$("play").addEventListener("click", () => {
  playing = !playing;
  $("play").textContent = playing ? "Pause Replay" : "Play Replay";
});
$("speed").addEventListener("input", start);
load().catch((error) => {
  $("mode").textContent = error.message;
  $("status-dot").classList.add("error");
});

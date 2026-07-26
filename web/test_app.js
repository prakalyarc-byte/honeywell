const assert = require("node:assert/strict");
const fs = require("node:fs");
const test = require("node:test");
const vm = require("node:vm");

const source = `${fs.readFileSync(__dirname + "/app.js", "utf8")}\nmodule.exports = { providerLabel, sourceLabel, unavailableMessage };`;
const elements = new Map();
const context = {
  module: { exports: {} },
  document: {
    createElement: () => ({ textContent: "", innerHTML: "" }),
    getElementById: (id) => {
      if (!elements.has(id)) elements.set(id, { addEventListener() {}, classList: { add() {} } });
      return elements.get(id);
    },
  },
  fetch: () => new Promise(() => {}),
  Intl,
  location: { search: "" },
  setInterval: () => 0,
  clearInterval() {},
  URLSearchParams,
};
vm.runInNewContext(source, context);
const { providerLabel, sourceLabel, unavailableMessage } = context.module.exports;

test("provider evidence handles absent, primary-only, and fallback metadata", () => {
  assert.equal(providerLabel({}), "unknown model via unknown provider; fallback usage unavailable");
  assert.equal(
    providerLabel({ metadata: { primary_provider: "groq", primary_model: "openai/gpt-oss-120b", fallback_calls: 0 } }),
    "openai/gpt-oss-120b via GroqCloud; no fallback calls",
  );
  assert.equal(
    providerLabel({ metadata: { primary_provider: "groq", primary_model: "openai/gpt-oss-120b", fallback_calls: 2 } }),
    "openai/gpt-oss-120b via GroqCloud; 2 local fallback calls",
  );
});

test("provider evidence preserves long untrusted model text for safe textContent rendering", () => {
  const model = `<img src=x onerror=alert(1)>${"x".repeat(120)}`;
  assert.equal(
    providerLabel({ metadata: { primary_provider: "ollama", primary_model: model, fallback_calls: 0 } }),
    `${model} via Ollama; no fallback calls`,
  );
  assert.match(source, /\$\("mode"\)\.textContent/);
});

test("source status distinguishes live, replay fallback, and missing data", () => {
  assert.equal(sourceLabel("latest.json", true, {}), "Live simulation feed");
  assert.equal(sourceLabel("demo-run.json", false, {}), "7-day EnergyPlus replay (unverified)");
  assert.equal(sourceLabel("demo-run.json", false, { metadata: { verified: false } }), "7-day EnergyPlus replay (unverified)");
  assert.equal(sourceLabel("demo-run.json", false, { metadata: { verified: true } }), "Verified 7-day EnergyPlus replay");
  assert.equal(sourceLabel("demo-run.json", true, {}), "7-day EnergyPlus replay (unverified); live feed unavailable");
  assert.equal(sourceLabel("demo-run.json", true, { metadata: { verified: true } }), "Verified 7-day EnergyPlus replay; live feed unavailable");
  assert.equal(unavailableMessage(true), "Live feed and verified replay unavailable. Start the local server from the repository root.");
  assert.equal(unavailableMessage(false), "Verified replay unavailable. Start the local server from the repository root.");
});

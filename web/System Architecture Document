# System Architecture Document

Eco-Loop turns EnergyPlus into a safe autonomous building sandbox. A hosted LLM reasons over a compact tool surface; Python enforces every safety constraint. This report explains the tool-calling architecture, prompt engineering strategies, prompt latency management, and the technical approach to handling lengthy simulation logs.

## Closed Loop

`EnergyPlus -> telemetry -> Ollama qwen3:8b -> local FastMCP tools -> deterministic validation -> setpoint actuators -> measured dashboard`

Ollama `qwen3:8b` is the primary decision model. EnergyPlus, the six MCP tools, validation, and actuation remain local. Run metadata records the actual provider and fallback count on every decision.

## Tool-Calling Architecture

Six FastMCP tools expose the minimum inspect–reason–act–verify surface:

| Tool | Arguments | Role |
|---|---|---|
| `inspect_model` | none | Return zones, schedules, sensors, actuators, targets |
| `read_telemetry` | none | Return latest sample, rolling window, and targets |
| `read_runtime_errors` | none | Return error count and recent unique messages |
| `evaluate_action` | `heating_c`, `cooling_c`, `reason` | Validate setpoints and issue a single-use token |
| `apply_setpoints` | `token` | Queue a previously validated action |
| `compare_runs` | none | Compare baseline and optimized summaries |

Sequence is enforced server-side, not by the model. A deterministic state machine (`_next_required_tool`) names the next required tool each turn: inspect, then the two observations, then evaluate, then apply. The model is free to emit any tool, but out-of-sequence calls are acknowledged and redirected rather than executed, so the model can never skip validation.

Native structured tool calls are preferred. When a provider returns parallel tool calls, `_select_model_call` selects the one matching the expected tool and preserves its call ID; `_surplus_tool_messages` acknowledges every other call ID with a skipped result so the conversation history stays complete and provider-valid. Missing native call IDs receive stable `call-turn-{n}-{i}` IDs. A content-JSON fallback parses `{"name","arguments"}` from assistant text for models that cannot emit structured calls.

Provider history is normalized at the transport boundary (`_messages_for_provider`): private annotations, reasoning content, and provider-only fields are stripped; Qwen receives stringified arguments and a `type: "function"` discriminator; Ollama receives object arguments and a `tool_name` field. The model never sees its own private metadata.

Single-use action tokens prevent replay. `evaluate_action` mints a `secrets.token_urlsafe(12)` token stored against the proposed action; `apply_setpoints` pops it once. A second apply with the same token is rejected.

## Prompt Engineering

The system prompt fixes a strict priority order: comfort and air quality first, then hard physical limits, then peak demand, then energy and carbon. Temperature is zero for Ollama and low (0.2–0.4) for Qwen retries, so the model proposes rather than improvises.

The model is invoked only when a trigger fires, not every timestep. Triggers: first controllable timestep, occupancy change, baseline schedule change, new PMV or CO2 violation, peak target crossing, carbon band change, and a three-hour heartbeat. Between decisions the last validated action stays active, so the model does not re-litigate steady state.

Each decision opens with one system message and one user message containing the trigger string, an `inspect_required` flag, and the required observation list. No telemetry dump, no history preamble. The model replies with one tool call per turn.

Tool results are compacted before they re-enter the conversation (`_model_tool_result`): telemetry is reduced to the few fields the model needs (hour, occupied, current and baseline setpoints, demand, carbon, and per-zone name/PMV/CO2/temperature); error logs collapse to a count and the two most recent entries; inspect returns only zones, schedules, targets, and run days. Full results stay in `state.audit` for the dashboard; the model sees only what it needs to decide.

When `evaluate_action` rejects a proposal, the rejection message returns the exact violations, the current observed state, the legal heating/cooling ranges, the 1 C movement limit, the 2 C deadband, and a one-line instruction. One correction is allowed; a second rejection restores the baseline and ends the turn.

## Prompt Latency Management

Triggered cadence is the primary latency control: the model is called only on state change, not on every EnergyPlus timestep, so most hours incur zero model cost.

Each decision is bounded:

- 12-turn conversational cap per decision.
- One validation correction before baseline restoration.
- Ollama request timeout: 30 seconds.

## Handling Lengthy Simulation Logs

EnergyPlus produces voluminous telemetry, error streams, and audit trails. The architecture keeps the model context small while preserving full evidence for the dashboard.

- `read_runtime_errors` returns a count and the ten most recent unique messages, not the full error log.
- `read_telemetry` returns the latest sample and a one-hour rolling window, not the full series.
- `inspect_model` returns a curated subset (zones, schedules, targets, run days), not the full IDF.
- `_model_tool_result` compacts results further before they re-enter the conversation, so a 672-hour replay never reaches the model as context.
- Full tool results and every model call are appended to `state.audit`, which the dashboard renders; the model never re-reads the audit.

Conversation history is preserved as complete assistant/tool-call/tool-result pairs (no raw-count slicing), so provider replay stays valid, but each compacted result keeps only decision-relevant fields.

## Safety Boundary

Python, not the LLM, enforces occupied/unoccupied bounds, 2 C deadband, 1 C hourly movement, PMV and CO2 gating, weather-run-only actuation, finite values, single-use tokens, baseline restoration on context-change failure, and the 12-turn cap. Provider failures retain the last safe action and never stop the EnergyPlus simulation.

## Deployment

Docker runs EnergyPlus, FastMCP, the worker, and Ollama locally. Vercel serves static verified replay data only.

## Measured Results

Values come directly from the frozen `web/demo-run.json`. Absolute PMV comfort passes 95%.

| Metric | Baseline | Optimized | Delta |
|---|---:|---:|---:|
| Total electricity (kWh) | 853.05 | 850.05 | -0.35% |
| Estimated cost at £0.28/kWh | £238.85 | £238.01 | £0.84 saved |
| HVAC electricity (kWh) | 28.36 | 28.10 | -0.95% |
| Peak demand (kW) | 15.68 | 15.61 | -0.47% |
| Operational carbon (kgCO2e) | 111.60 | 111.22 | -0.34% |
| Occupied PMV compliance (%) | 99.24 | 98.03 | 0.8 points |
| Absolute 95% PMV boundary | — | — | PASS |

HVAC is roughly 3% of facility electricity in this five-zone model, so total electricity reductions are structurally small even when cooling and fan energy fall.

## Limitations

One five-zone building, one London weather week, shared thermostat schedules, triggered supervisory control, fixed hourly carbon data, one local Ollama model, and operational carbon only.

# Eco-Loop Building Agents

Eco-Loop uses GroqCloud `openai/gpt-oss-120b` as the primary decision model. EnergyPlus, six FastMCP tools, deterministic validation, and actuation stay local. Ollama `qwen3:8b` is a bounded local fallback only.

## Architecture

`EnergyPlus -> telemetry -> Qwen3 8B -> local MCP tools -> deterministic validation -> active setpoint actuators -> measured dashboard`


## Host Setup

Python 3.12+, Node.js 18+, and Docker Compose required. GroqCloud access requires a user-supplied API key; `.env` is ignored by Git.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest worker.test_worker -v
```

## Run

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen3:8b
read -rsp "Groq API key: " KEY
printf 'GROQ_API_KEY=%s\nLLM_PROVIDER=groq\nGROQ_MODEL=openai/gpt-oss-120b\nOLLAMA_MODEL=qwen3:8b\nOLLAMA_TIMEOUT_SECONDS=120\n' "$KEY" > .env
unset KEY
docker compose build worker
docker compose run --rm worker python3 -m worker.run preflight
docker compose run --rm worker python3 -m worker.run optimized --hours 1
docker compose run --rm worker python3 -m worker.prepare_model prepare
docker compose run --rm worker python3 -m worker.run baseline
docker compose run --rm worker python3 -m worker.run optimized
docker compose run --rm worker python3 -m worker.prepare_model finalize
python3 -m http.server 4173 --directory web
```

## Results

Measured results live in `web/demo-run.json`; raw EnergyPlus artifacts are reproducible with the commands above. The dashboard reports 0.35% electricity reduction, estimated cost savings at an explicit £0.28/kWh assumption, 98.03% occupied PMV compliance, 168 validated actions, and zero runtime errors.

## Status

- Implemented and tested: Groq-primary transport, bounded local Qwen3 fallback, provider/model defaults, source code, executable baseline/control/optimized EnergyPlus models, six MCP tools, active PyEnergyPlus loop, preflight, dashboard, architecture report, demo script, presentation outline, requirement matrix, and submission checklist.
- Previously measured replay: `web/demo-run.json` contains a verified run produced before these Task 4 provider/model defaults. No Groq GPT-OSS 120B plus Qwen3 8B live run is claimed yet.
- External/pending: recorded demo video, official presentation template file, published GitHub URL, submission PDF or ZIP bundle.

These external artifacts are not in the workspace and must be produced from the local artifacts above.

## Limitations

One building, one London weather week, shared zone schedules, trigger-based agent decisions, bundled grid-intensity data, estimated flat electricity tariff, and operational carbon only. See `docs/requirement-matrix.md` for full compliance status.

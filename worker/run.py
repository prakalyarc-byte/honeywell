from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from fastmcp import Client

from worker.prepare_model import MODELS
from worker.tools import EcoLoopState, compare, create_server, summarize

ROOT = Path(__file__).resolve().parents[1]
WEB_LIVE = ROOT / "web" / "latest.json"
RUNS = ROOT / "runs"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120"))

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")

MAX_AGENT_TURNS = 12
REQUIRED_OBSERVATIONS = ("read_telemetry", "read_runtime_errors")

SYSTEM_PROMPT = """You control a simulated five-zone office.
Comfort and safety outrank peak demand, energy, and carbon savings.

Reply with one JSON object per turn describing one tool call:
{"name": "<tool>", "arguments": { ... }}

Tools (use these exact names):
- inspect_model: {}
- read_telemetry: {}
- read_runtime_errors: {}
- compare_runs: {}
- evaluate_action: {"heating_c": <number>, "cooling_c": <number>, "reason": "<short>"}
- apply_setpoints: {"token": "<token from evaluate_action>"}

Sequence each decision:
1. If inspect_required is true, call inspect_model.
2. Call read_telemetry. Read current_heating_c, current_cooling_c, occupied, and the per-zone pmv and co2_ppm.
3. Call read_runtime_errors.
4. Call evaluate_action with heating_c and cooling_c:
   - If occupied is true or baseline_heating_c is above 20 C (preconditioning), use heating [20, 22] and cooling [23, 26].
   - Otherwise use unoccupied heating [16, 20] and cooling [26, 30].
   - cooling_c - heating_c must be at least 2.
   - new heating_c must be within 1 C of current_heating_c; same for cooling.
   - If a zone pmv is outside [-0.5, 0.5] or co2_ppm above 1000, you may not relax (lower heating_c or raise cooling_c).
5. Call apply_setpoints with the exact token string returned by evaluate_action.

Each tool is at most once per decision. Reply with one JSON object only.
"""


def _zone_violation(zone: dict) -> bool:
    return abs(float(zone["pmv"])) > 0.5 or float(zone["co2_ppm"]) > 1000.0


def decision_trigger(latest: dict, previous: dict | None, hours_since: int, peak_kw: float | None) -> str | None:
    if previous is None:
        return "first controllable timestep"
    if latest["occupied"] != previous["occupied"]:
        return "occupancy changed"
    for key in ("baseline_heating_c", "baseline_cooling_c"):
        if latest.get(key) != previous.get(key):
            return "baseline schedule changed"
    latest_bad = any(_zone_violation(zone) for zone in latest["zones"])
    previous_bad = any(_zone_violation(zone) for zone in previous["zones"])
    if latest_bad and not previous_bad:
        return "comfort target violated"
    if peak_kw is not None:
        target_w = peak_kw * 1000.0
        if latest["facility_demand_w"] > target_w >= previous["facility_demand_w"]:
            return "peak target crossed"
    if int(latest["carbon_g_per_kwh"] // 50) != int(previous["carbon_g_per_kwh"] // 50):
        return "carbon band changed"
    if hours_since >= 3:
        return "three-hour heartbeat"
    return None


def tool_schema(tool: Any) -> dict:
    schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", {"type": "object"})
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description or "", "parameters": schema},
    }


def _default_open(request: urllib.request.Request, timeout: int) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _messages_for_provider(messages: list[dict], provider: str) -> list[dict]:
    normalized = []
    for source in messages:
        message = {
            key: value
            for key, value in source.items()
            if not key.startswith("_") and key not in {"tool_name", "reasoning_content"}
        }
        if provider == "ollama" and message.get("role") == "tool" and message.get("name"):
            message["tool_name"] = message["name"]
        if isinstance(message.get("tool_calls"), list):
            calls = []
            for source_call in message["tool_calls"]:
                call = dict(source_call)
                if provider == "groq":
                    call.setdefault("type", "function")
                function = dict(call.get("function") or {})
                arguments = function.get("arguments", {})
                if provider == "groq" and not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                elif provider == "ollama" and isinstance(arguments, str):
                    try:
                        parsed = json.loads(arguments)
                        arguments = parsed if isinstance(parsed, dict) else {}
                    except json.JSONDecodeError:
                        arguments = {}
                function["arguments"] = arguments
                call["function"] = function
                calls.append(call)
            message["tool_calls"] = calls
        normalized.append(message)
    return normalized


class ProviderHTTPError(RuntimeError):
    def __init__(self, provider: str, status: int, code: str | None, detail: str):
        super().__init__(f"{provider} HTTP {status}: {detail}")
        self.provider = provider
        self.status = status
        self.code = code


def _annotate_reply(
    message: dict,
    provider: str,
    attempts: int,
    fallback: bool,
    error_codes: list[str] | None = None,
) -> dict:
    return {
        **message,
        "_provider": provider,
        "_attempts": attempts,
        "_fallback": fallback,
        "_error_codes": list(error_codes or []),
    }


def ollama_request(messages: list[dict], tools: list[dict], opener: Callable = _default_open) -> dict:
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": _messages_for_provider(messages, "ollama"),
        "tools": tools,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 200},
    }).encode()
    request = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body, headers={"Content-Type": "application/json"})
    payload = json.loads(opener(request, OLLAMA_TIMEOUT_SECONDS))
    if "message" not in payload:
        raise ValueError("Ollama response has no message")
    return payload["message"]


def groq_request(messages: list[dict], tools: list[dict], opener: Callable = _default_open, temperature=0.2) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for Groq inference")
    body = json.dumps({
        "model": GROQ_MODEL,
        "messages": _messages_for_provider(messages, "groq"),
        "tools": tools,
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "reasoning_effort": "low",
        "include_reasoning": False,
        "max_completion_tokens": 1024,
        "temperature": temperature,
    }).encode()
    request = urllib.request.Request(
        GROQ_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "eco-loop-worker/1.0",
        },
    )
    try:
        raw = opener(request, 30)
    except urllib.error.HTTPError as error:
        try:
            payload = json.loads(error.read())
        except (ValueError, json.JSONDecodeError):
            payload = {}
        details = payload.get("error") if isinstance(payload, dict) else {}
        details = details if isinstance(details, dict) else {}
        raise ProviderHTTPError(
            "groq",
            error.code,
            details.get("code"),
            str(details.get("message") or error.reason or "request failed")[:300],
        ) from None
    payload = json.loads(raw)
    choices = payload.get("choices") or []
    if not choices or "message" not in choices[0]:
        raise ValueError("Groq response has no message")
    return choices[0]["message"]


def model_request(messages: list[dict], tools: list[dict]) -> dict:
    if LLM_PROVIDER == "ollama":
        return _annotate_reply(ollama_request(messages, tools), "ollama", 1, False)

    last_error = None
    error_codes = []
    for attempt, temperature in enumerate((0.2, 0.3, 0.4), start=1):
        try:
            return _annotate_reply(
                groq_request(messages, tools, temperature=temperature),
                "groq", attempt, False, error_codes,
            )
        except ProviderHTTPError as error:
            last_error = error
            if error.code != "tool_use_failed":
                raise
            error_codes.append("tool_use_failed")
        except (OSError, TimeoutError) as error:
            error_codes.append("transport_error")
            last_error = error
            break

    try:
        return _annotate_reply(
            ollama_request(messages, tools),
            "ollama",
            min(3, max(1, len(error_codes))),
            True,
            error_codes,
        )
    except Exception as fallback_error:
        primary_code = error_codes[-1] if error_codes else "unknown_primary_error"
        raise RuntimeError(
            f"Groq primary failed ({primary_code}); Ollama fallback failed: {fallback_error}"
        ) from last_error


CONTEXT_CHANGE_TRIGGERS = {"occupancy changed", "baseline schedule changed"}


def validate_run_acceptance(mode: str, state: EcoLoopState) -> None:
    if mode != "optimized":
        return
    if not state.actions:
        raise RuntimeError("optimized run has no validated autonomous actions")
    blocking = [
        error for error in state.errors
        if error.get("source") not in {"validation", "baseline_fallback"}
    ]
    if blocking:
        sources = sorted({error.get("source", "unknown") for error in blocking})
        raise RuntimeError(f"optimized run has agent errors: {', '.join(sources)}")


def _persisted_agent_error(error: Exception) -> dict:
    if isinstance(error, ProviderHTTPError):
        code = error.code if error.code in {"tool_use_failed"} else "http_error"
        return {
            "source": "agent",
            "provider": error.provider,
            "status": error.status,
            "code": code,
            "message": f"{error.provider} request failed (HTTP {error.status})",
        }
    return {"source": "agent", "message": str(error)}


def restore_baseline_after_failure(state: EcoLoopState, trigger: str) -> bool:
    if trigger not in CONTEXT_CHANGE_TRIGGERS or not state.telemetry:
        return False
    latest = state.telemetry[-1]
    state.pending_action = {
        "heating_c": float(latest["baseline_heating_c"]),
        "cooling_c": float(latest["baseline_cooling_c"]),
        "reason": f"baseline fallback after {trigger}",
    }
    state.audit.append({"tool": "baseline_fallback", "result": state.pending_action.copy()})
    state.audit[-1]["hour"] = latest.get("hour")
    return True


def _next_required_tool(inspected: bool, observed: set[str], has_token: bool) -> str:
    if not inspected:
        return "inspect_model"
    if "read_telemetry" not in observed:
        return "read_telemetry"
    if "read_runtime_errors" not in observed:
        return "read_runtime_errors"
    if not has_token:
        return "evaluate_action"
    return "apply_setpoints"


def _tool_message(name: str, content: dict, call_id: str | None, provider: str) -> dict:
    message = {
        "role": "tool",
        "name": name,
        "content": json.dumps(content),
    }
    if provider == "ollama":
        message["tool_name"] = name
    if call_id:
        message["tool_call_id"] = call_id
    return message


def _ensure_tool_call_ids(calls: list[dict], turn: int) -> list[dict]:
    normalized = []
    for index, source in enumerate(calls):
        call = dict(source) if isinstance(source, dict) else {}
        call["id"] = str(call.get("id") or f"call-turn-{turn}-{index}")
        normalized.append(call)
    return normalized


def _surplus_tool_messages(
    calls: list[dict],
    selected_id: str | None,
    provider: str,
    expected: str,
) -> list[dict]:
    messages = []
    for call in calls:
        if call.get("id") == selected_id:
            continue
        function = call.get("function") if isinstance(call, dict) else {}
        function = function if isinstance(function, dict) else {}
        messages.append(_tool_message(
            str(function.get("name") or "unknown_tool"),
            {
                "skipped": "parallel tool call not executed",
                "next_required_tool": expected,
                "instruction": "Emit one tool call per response.",
            },
            call.get("id") if isinstance(call, dict) else None,
            provider,
        ))
    return messages


async def agent_decision(state: EcoLoopState, trigger: str) -> bool:
    server = create_server(state)
    async with Client(server, timeout=10) as client:
        definitions = await client.list_tools()
        schemas = [tool_schema(tool) for tool in definitions]
        known = {tool.name for tool in definitions}
        zero_argument = {
            tool.name
            for tool in definitions
            if not (getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", {}))
            .get("required")
        }
        observed: set[str] = set()
        rejected = 0
        applied_token = None
        inspected = any(entry.get("tool") == "inspect_model" for entry in state.audit)
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({
                "trigger": trigger,
                "inspect_required": not inspected,
                "required_observations": list(REQUIRED_OBSERVATIONS),
            })},
        ]
        for turn in range(MAX_AGENT_TURNS):
            expected = _next_required_tool(inspected, observed, applied_token is not None)
            message = await asyncio.to_thread(model_request, messages, schemas)
            provider = message.get("_provider", LLM_PROVIDER)
            state.audit.append({
                "tool": "model_provider",
                "result": {
                    "provider": provider,
                    "model": GROQ_MODEL if provider == "groq" else OLLAMA_MODEL,
                    "attempts": int(message.get("_attempts", 1)),
                    "fallback": bool(message.get("_fallback", False)),
                    "error_codes": list(message.get("_error_codes") or []),
                },
                "hour": state.telemetry[-1].get("hour") if state.telemetry else None,
            })
            native_calls = _ensure_tool_call_ids(message.get("tool_calls") or [], turn)
            message["tool_calls"] = native_calls
            call = _select_model_call(message, expected, known, zero_argument)
            if call is None:
                if native_calls:
                    messages.append({
                        "role": "assistant",
                        "content": message.get("content", "") or "",
                        "tool_calls": native_calls,
                    })
                    messages.extend(_surplus_tool_messages(native_calls, None, provider, expected))
                    messages.append({
                        "role": "user",
                        "content": f"Out-of-sequence tool. Call {expected} next with required arguments.",
                    })
                    continue
                preview = (message.get("content") or "")[:120]
                state.audit.append({
                    "tool": "model_retry",
                    "result": {"reason": "unparseable response", "preview": preview},
                    "hour": state.telemetry[-1].get("hour") if state.telemetry else None,
                })
                messages.append({"role": "user", "content": "Reply with one JSON object describing a single tool call from the allowed set."})
                if turn >= MAX_AGENT_TURNS - 2:
                    state.errors.append({
                        "source": LLM_PROVIDER,
                        "message": "model emitted no parseable tool call before turn limit",
                    })
                    restore_baseline_after_failure(state, trigger)
                    return False
                continue
            assistant_message = {
                "role": "assistant",
                "content": message.get("content", "") or "",
            }
            if message.get("tool_calls"):
                assistant_message["tool_calls"] = message["tool_calls"]
            messages.append(assistant_message)
            call_id = call.get("call_id")
            messages.extend(_surplus_tool_messages(native_calls, call_id, provider, expected))
            name = call["name"]
            arguments = call["arguments"]
            if not isinstance(arguments, dict):
                state.errors.append({"source": "ollama", "message": "tool arguments must be an object"})
                restore_baseline_after_failure(state, trigger)
                return False
            if name not in known:
                state.errors.append({"source": "ollama", "message": f"unknown tool: {name}"})
                restore_baseline_after_failure(state, trigger)
                return False
            if name != expected:
                messages.append(_tool_message(name, {
                    "skipped": "out of sequence",
                    "next_required_tool": expected,
                }, call_id, provider))
                messages.append({
                    "role": "user",
                    "content": f"Out-of-sequence tool. Call {expected} next with required arguments.",
                })
                continue
            try:
                result = (await client.call_tool(name, arguments)).data
            except Exception as error:
                messages.append(_tool_message(name, {"error": str(error)}, call_id, provider))
                continue
            if state.audit and state.telemetry:
                state.audit[-1].setdefault("hour", state.telemetry[-1].get("hour"))
            messages.append(_tool_message(name, _model_tool_result(name, result), call_id, provider))
            messages[:] = _trim_messages(messages)
            if name == "inspect_model":
                inspected = True
            elif name in REQUIRED_OBSERVATIONS:
                observed.add(name)
            elif name == "evaluate_action" and not result["valid"]:
                rejected += 1
                if rejected > 1:
                    state.errors.append({"source": "validation", "message": "second invalid proposal; retaining safe action"})
                    restore_baseline_after_failure(state, trigger)
                    return False
                latest = state.telemetry[-1]
                occupied = bool(latest["occupied"])
                preconditioning = not occupied and float(latest.get("baseline_heating_c", 0.0)) > 20.0
                heat_range = (20.0, 22.0) if occupied or preconditioning else (16.0, 20.0)
                cool_range = (23.0, 26.0) if occupied or preconditioning else (26.0, 30.0)
                messages.append({"role": "user", "content": json.dumps({
                    "rejected_violations": result["violations"],
                    "observed_state": {
                        "occupied": occupied,
                        "current_heating_c": latest["current_heating_c"],
                        "current_cooling_c": latest["current_cooling_c"],
                        "heating_range_c": heat_range,
                        "cooling_range_c": cool_range,
                        "max_movement_c": 1.0,
                        "minimum_deadband_c": 2.0,
                    },
                    "instruction": "Choose different numeric setpoints satisfying every listed constraint, then call evaluate_action. Do not repeat observations or call apply_setpoints.",
                })})
            elif name == "evaluate_action" and result.get("valid"):
                applied_token = result.get("token")
            elif name == "apply_setpoints":
                if not result["applied"]:
                    token = _last_evaluate_token(state)
                    if token and (not isinstance(arguments.get("token"), str) or not state.action_tokens.get(arguments["token"])):
                        retry = (await client.call_tool("apply_setpoints", {"token": token})).data
                        if retry["applied"]:
                            return True
                return bool(result["applied"])
        state.errors.append({"source": "ollama", "message": "agent turn limit reached; retaining safe action"})
        restore_baseline_after_failure(state, trigger)
        return False


def _last_evaluate_token(state: EcoLoopState) -> str | None:
    for entry in reversed(state.audit):
        if entry.get("tool") == "evaluate_action":
            token = (entry.get("result") or {}).get("token")
            if token and token in state.action_tokens:
                return token
    return None


def _model_tool_result(name: str, result: dict) -> dict:
    """Keep Qwen context small while preserving full results in state.audit."""
    if name == "read_telemetry":
        latest = result.get("latest") or {}
        return {
            "latest": {
                key: latest.get(key)
                for key in (
                    "hour", "occupied", "current_heating_c", "current_cooling_c",
                    "baseline_heating_c", "baseline_cooling_c", "facility_demand_w",
                    "carbon_g_per_kwh",
                )
            },
            "zones": [
                {key: zone.get(key) for key in ("name", "pmv", "co2_ppm", "temperature_c")}
                for zone in latest.get("zones", [])
            ],
            "targets": result.get("targets", {}),
        }
    if name == "read_runtime_errors":
        return {"count": result.get("count", 0), "recent": result.get("recent", [])[-2:]}
    if name == "inspect_model":
        return {
            key: result.get(key)
            for key in ("zones", "heating_schedule", "cooling_schedule", "targets", "run_days")
        }
    return result


def _trim_messages(messages: list[dict]) -> list[dict]:
    # A decision has a 12-turn cap; preserving complete call/result pairs is safer than raw-count slicing.
    return messages


def _select_model_call(
    message: dict,
    expected: str,
    known: set[str],
    zero_argument: set[str],
) -> dict | None:
    calls = message.get("tool_calls") or []
    if isinstance(calls, list):
        selected = next(
            (
                call for call in calls
                if isinstance(call, dict)
                and isinstance(call.get("function"), dict)
                and call["function"].get("name") == expected
            ),
            None,
        )
        if selected:
            parsed = _parse_model_response(
                {"tool_calls": [selected]}, known, zero_argument
            )
            if parsed:
                parsed["call_id"] = selected.get("id")
            return parsed
    if calls:
        return None
    parsed = _parse_model_response(message, known, zero_argument)
    if parsed:
        parsed["call_id"] = None
    return parsed


def _parse_model_response(message: dict, known: set[str], zero_argument: set[str] | None = None) -> dict | None:
    zero_argument = zero_argument or set()
    calls = message.get("tool_calls") or []
    if isinstance(calls, list) and len(calls) == 1:
        function = calls[0].get("function", {}) if isinstance(calls[0], dict) else {}
        name = function.get("name")
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if name in zero_argument:
            arguments = {}
        if name == "evaluate_action" and isinstance(arguments, dict):
            arguments.setdefault("reason", "model proposal")
        if name in known and isinstance(arguments, dict):
            return {"name": name, "arguments": arguments}
    content = (message.get("content") or "").strip()
    if not content:
        return None
    payload = _extract_json_object(content)
    if not isinstance(payload, dict):
        return None
    name = payload.get("name") or payload.get("tool")
    arguments = payload.get("arguments")
    if name in zero_argument:
        arguments = {}
    if arguments is None:
        setpoints = payload.get("setpoints") or payload.get("proposed_setpoints")
        if isinstance(setpoints, dict):
            arguments = {
                "heating_c": setpoints.get("heating_c"),
                "cooling_c": setpoints.get("cooling_c"),
                "reason": payload.get("reason") or "model proposal",
            }
    if name not in known or not isinstance(arguments, dict):
        return None
    if name == "evaluate_action":
        arguments.setdefault("reason", "model proposal")
    return {"name": name, "arguments": arguments}


def _extract_json_object(text: str) -> object | None:
    fence = text.find("```")
    if fence != -1:
        text = text[fence:]
        end = text.find("```", 3)
        if end != -1:
            text = text[3:end]
        text = text.split("json", 1)[-1] if text.lstrip().startswith("json") else text
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
        next_start = text.find("{", start + 1)
        if next_start == start:
            break
        start = next_start
    return None


def atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(data, handle, separators=(",", ":"))
        temporary = Path(handle.name)
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


class Simulation:
    def __init__(self, mode: str, max_hours: int | None = None):
        # ponytail: host without pyenergyplus still imports for unit tests
        from pyenergyplus.api import EnergyPlusAPI

        self.mode = mode
        self.max_hours = max_hours
        self.api = EnergyPlusAPI()
        self.state = EcoLoopState()
        self.handles: dict[str, Any] = {}
        self.initialized = False
        self.last_control_hour = -1
        self.last_decision_hour = -1
        self.last_decision_sample: dict | None = None
        self.callback_error: Exception | None = None
        self.carbon = json.loads((MODELS / "carbon-intensity.json").read_text())
        baseline_path = RUNS / "baseline" / "run.json"
        if baseline_path.exists():
            self.state.baseline_summary = json.loads(baseline_path.read_text()).get("summary")

    def initialize_handles(self, ep_state) -> None:
        if self.initialized or not self.api.exchange.api_data_fully_ready(ep_state):
            return
        exchange = self.api.exchange
        definition_path = MODELS / ("control-ready.epJSON" if self.mode == "optimized" else "baseline.epJSON")
        definition = json.loads(definition_path.read_text())
        zones = list(definition["Zone"])
        people_by_zone = {
            person["zone_or_zonelist_or_space_or_spacelist_name"]: name
            for name, person in definition["People"].items()
        }
        people = [people_by_zone.get(zone) for zone in zones]
        baseline_path = RUNS / "baseline" / "run.json"
        baseline_data = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
        baseline_peak = (baseline_data.get("summary") or {}).get("peak_kw")
        self.state.model = {"zones": zones, "people": people, "heating_schedule": "HTGSETP_SCH", "cooling_schedule": "CLGSETP_SCH", "baseline_heating_schedule": "Htg-SetP-Sch", "baseline_cooling_schedule": "Clg-SetP-Sch", "run_days": 7, "targets": {"pmv": [-0.5, 0.5], "co2_ppm_max": 1000, "peak_kw": baseline_peak * 0.9 if baseline_peak else None}}
        self.handles["zones"] = {
            zone: {
                "temperature": exchange.get_variable_handle(ep_state, "Zone Mean Air Temperature", zone),
                "humidity": exchange.get_variable_handle(ep_state, "Zone Air Relative Humidity", zone),
                "co2": exchange.get_variable_handle(ep_state, "Zone Air CO2 Concentration", zone),
                "occupancy": exchange.get_variable_handle(ep_state, "Zone People Occupant Count", zone),
                "pmv": exchange.get_variable_handle(ep_state, "Zone Thermal Comfort Fanger Model PMV", person) if person else -1,
            }
            for zone, person in zip(zones, people)
        }
        self.handles["hvac_energy"] = exchange.get_meter_handle(ep_state, "Electricity:HVAC")
        self.handles["demand"] = exchange.get_variable_handle(ep_state, "Facility Total Electricity Demand Rate", "Whole Building")
        self.handles["outdoor"] = exchange.get_variable_handle(ep_state, "Site Outdoor Air Drybulb Temperature", "Environment")
        thermostat_zone = next((z for z, p in zip(zones, people) if p), zones[0])
        self.handles["heating_sp"] = exchange.get_variable_handle(ep_state, "Zone Thermostat Heating Setpoint Temperature", thermostat_zone)
        self.handles["cooling_sp"] = exchange.get_variable_handle(ep_state, "Zone Thermostat Cooling Setpoint Temperature", thermostat_zone)
        self.handles["baseline_heating"] = exchange.get_variable_handle(ep_state, "Schedule Value", "Htg-SetP-Sch")
        self.handles["baseline_cooling"] = exchange.get_variable_handle(ep_state, "Schedule Value", "Clg-SetP-Sch")
        flat = [self.handles["hvac_energy"], self.handles["demand"], self.handles["outdoor"], self.handles["heating_sp"], self.handles["cooling_sp"], self.handles["baseline_heating"], self.handles["baseline_cooling"]]
        if self.mode == "optimized":
            self.handles["heating"] = exchange.get_actuator_handle(ep_state, "Schedule:Compact", "Schedule Value", "HTGSETP_SCH")
            self.handles["cooling"] = exchange.get_actuator_handle(ep_state, "Schedule:Compact", "Schedule Value", "CLGSETP_SCH")
            flat += [self.handles["heating"], self.handles["cooling"]]
        flat += [handle for zone in self.handles["zones"].values() for handle in zone.values() if handle != -1]
        if any(handle == -1 for handle in flat):
            self.api.runtime.issue_severe(ep_state, "Eco-Loop required sensor or actuator handle missing")
            self.api.runtime.stop_simulation(ep_state)
            raise RuntimeError("required EnergyPlus handle missing")
        self.initialized = True

    def _hour(self, ep_state) -> int:
        return max(0, math.floor(self.api.exchange.current_sim_time(ep_state)))

    def _finished(self, hour: int) -> bool:
        limit = self.max_hours if self.max_hours is not None else int(self.state.model.get("run_days", 0)) * 24
        return bool(limit and hour >= limit)

    def _guard_callback(self, callback: Callable) -> Callable:
        def guarded(ep_state):
            try:
                callback(ep_state)
            except Exception as error:
                self.callback_error = error
                self.api.runtime.stop_simulation(ep_state)

        return guarded

    def collect(self, ep_state) -> None:
        if self.api.exchange.warmup_flag(ep_state) or self.api.exchange.kind_of_sim(ep_state) != 3:
            return
        self.initialize_handles(ep_state)
        exchange = self.api.exchange
        hour = self._hour(ep_state)
        if self.max_hours is not None and hour >= self.max_hours:
            self.api.runtime.stop_simulation(ep_state)
            return
        zones = []
        occupied = False
        for name, handles in self.handles["zones"].items():
            count = exchange.get_variable_value(ep_state, handles["occupancy"]) if handles["occupancy"] != -1 else 0.0
            occupied = occupied or count > 0.1
            # ponytail: zones without People use neutral occupancy and PMV values.
            zones.append({"name": name, "temperature_c": exchange.get_variable_value(ep_state, handles["temperature"]), "humidity_pct": exchange.get_variable_value(ep_state, handles["humidity"]), "co2_ppm": exchange.get_variable_value(ep_state, handles["co2"]), "occupants": count, "pmv": exchange.get_variable_value(ep_state, handles["pmv"]) if handles["pmv"] != -1 else 0.0})
        sample = {
            "hour": hour,
            "occupied": occupied,
            "outdoor_c": exchange.get_variable_value(ep_state, self.handles["outdoor"]),
            # ponytail: Electricity:Facility meter handle is -1 in 5ZoneAirCooled; compute from demand (W) * 900s
            "facility_energy_j": exchange.get_variable_value(ep_state, self.handles["demand"]) * 900,
            "hvac_energy_j": exchange.get_meter_value(ep_state, self.handles["hvac_energy"]),
            "facility_demand_w": exchange.get_variable_value(ep_state, self.handles["demand"]),
            "carbon_g_per_kwh": self.carbon["hourly"][hour % len(self.carbon["hourly"])],
            "current_heating_c": exchange.get_variable_value(ep_state, self.handles["heating_sp"]),
            "current_cooling_c": exchange.get_variable_value(ep_state, self.handles["cooling_sp"]),
            "baseline_heating_c": exchange.get_variable_value(ep_state, self.handles["baseline_heating"]),
            "baseline_cooling_c": exchange.get_variable_value(ep_state, self.handles["baseline_cooling"]),
            "zones": zones,
        }
        self.state.telemetry.append(sample)
        atomic_write(WEB_LIVE, self.payload())

    def control(self, ep_state) -> None:
        if self.mode != "optimized" or self.api.exchange.warmup_flag(ep_state) or self.api.exchange.kind_of_sim(ep_state) != 3:
            return
        self.initialize_handles(ep_state)
        hour = self._hour(ep_state)
        if self._finished(hour):
            if self.max_hours is not None:
                self.api.runtime.stop_simulation(ep_state)
            return
        if hour != self.last_control_hour and self.state.telemetry:
            self.last_control_hour = hour
            previous = self.last_decision_sample
            hours_since = hour - self.last_decision_hour if self.last_decision_hour >= 0 else hour + 1
            peak_kw = self.state.model.get("targets", {}).get("peak_kw")
            trigger = decision_trigger(self.state.telemetry[-1], previous, hours_since, peak_kw)
            if trigger:
                self.state.audit.append({"tool": "decision_trigger", "result": {"reason": trigger, "hour": hour}})
                try:
                    asyncio.run(agent_decision(self.state, trigger))
                except Exception as error:
                    self.state.errors.append(_persisted_agent_error(error))
                self.last_decision_hour = hour
                self.last_decision_sample = dict(self.state.telemetry[-1])
        action = self.state.pending_action
        if action:
            self.api.exchange.set_actuator_value(ep_state, self.handles["heating"], action["heating_c"])
            self.api.exchange.set_actuator_value(ep_state, self.handles["cooling"], action["cooling_c"])
            if self.state.telemetry:
                self.state.telemetry[-1]["current_heating_c"] = action["heating_c"]
                self.state.telemetry[-1]["current_cooling_c"] = action["cooling_c"]

    def message(self, message: bytes) -> None:
        text = message.decode("utf-8", errors="replace").strip()
        if not text:
            return
        severity = "fatal" if "**  Fatal  **" in text else "severe" if "** Severe  **" in text else "warning" if "** Warning **" in text else None
        if severity:
            self.state.errors.append({"source": "energyplus", "severity": severity, "message": text[:500]})

    def current_payload(self) -> dict:
        provider = LLM_PROVIDER
        model = GROQ_MODEL if provider == "groq" else OLLAMA_MODEL
        fallback_calls = sum(
            1 for entry in self.state.audit
            if entry.get("tool") == "model_provider"
            and (entry.get("result") or {}).get("fallback")
        )
        metadata = {
            "mode": self.mode,
            "provider": provider,
            "model": model,
            "primary_provider": provider,
            "primary_model": model,
            "fallback_provider": "ollama" if provider == "groq" else None,
            "fallback_model": OLLAMA_MODEL if provider == "groq" else None,
            "fallback_calls": fallback_calls,
            "verified": False,
        }
        current = {"metadata": metadata, "telemetry": self.state.telemetry, "actions": self.state.actions, "audit": self.state.audit, "errors": self.state.errors, "summary": summarize(self.state.telemetry) if self.state.telemetry else None}
        return current

    def payload(self) -> dict:
        current = self.current_payload()
        baseline_path = RUNS / "baseline" / "run.json"
        if self.mode == "optimized" and baseline_path.exists() and current["summary"]:
            baseline = json.loads(baseline_path.read_text())
            if baseline.get("summary"):
                return {"metadata": current["metadata"], "baseline": baseline, "optimized": current, "comparison": compare(baseline["summary"], current["summary"])}
        return current

    def run(self) -> dict:
        ep_state = self.api.state_manager.new_state()
        self.api.runtime.callback_end_zone_timestep_after_zone_reporting(ep_state, self._guard_callback(self.collect))
        self.api.runtime.callback_begin_system_timestep_before_predictor(ep_state, self._guard_callback(self.control))
        self.api.runtime.callback_message(ep_state, self.message)
        output = RUNS / self.mode
        output.mkdir(parents=True, exist_ok=True)
        model = MODELS / ("control-ready.idf" if self.mode == "optimized" else "baseline.idf")
        exit_code = self.api.runtime.run_energyplus(ep_state, ["-w", str(MODELS / "weather.epw"), "-d", str(output), str(model)])
        self.api.state_manager.delete_state(ep_state)
        if self.callback_error:
            raise RuntimeError("EnergyPlus callback failed") from self.callback_error
        if exit_code != 0:
            raise RuntimeError(f"EnergyPlus exited {exit_code}")
        if any(error.get("severity") in {"fatal", "severe"} for error in self.state.errors):
            raise RuntimeError("EnergyPlus reported fatal or severe errors")
        validate_run_acceptance(self.mode, self.state)
        result = self.current_payload()
        result["metadata"]["verified"] = True
        atomic_write(output / "run.json", result)
        return result


def run_simulation(mode: str, max_hours: int | None = None) -> dict:
    return Simulation(mode, max_hours=max_hours).run()


def preflight(opener: Callable = _default_open) -> dict[str, bool]:
    api_key = os.environ.get("GROQ_API_KEY", "")
    required_assets = (
        "baseline.idf", "control-ready.idf", "optimized.idf",
        "weather.epw", "carbon-intensity.json",
    )
    schedule_path = MODELS / "optimized-setpoints.csv"
    replay_path = ROOT / "web" / "demo-run.json"
    try:
        replay = json.loads(replay_path.read_text())
        verified_replay = bool(replay.get("metadata", {}).get("verified"))
    except (OSError, json.JSONDecodeError):
        verified_replay = False
    checks = {
        "groq_key": bool(api_key),
        "groq_reachable": False,
        "ollama_reachable": False,
        "local_assets": (
            all((MODELS / name).is_file() and (MODELS / name).stat().st_size > 0 for name in required_assets)
            and schedule_path.is_file()
            and len(schedule_path.read_text().splitlines()) > 1
        ),
        "baseline_run": (
            (RUNS / "baseline" / "run.json").is_file()
            and (RUNS / "baseline" / "run.json").stat().st_size > 0
        ),
        "verified_replay": verified_replay,
    }
    if api_key:
        try:
            message = groq_request(
                [{"role": "user", "content": "Reply with an inspect_model call."}],
                [{
                    "type": "function",
                    "function": {"name": "inspect_model", "parameters": {"type": "object"}},
                }],
                opener=opener,
            )
            calls = message.get("tool_calls") if isinstance(message, dict) else None
            function = calls[0].get("function") if isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            checks["groq_reachable"] = bool(
                isinstance(function, dict)
                and function.get("name") == "inspect_model"
                and arguments == {}
            )
        except Exception:
            pass
    try:
        request = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        payload = json.loads(opener(request, 5))
        available = {
            model.get("name") or model.get("model")
            for model in payload.get("models", [])
            if isinstance(model, dict)
        }
        checks["ollama_reachable"] = OLLAMA_MODEL in available
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    provider_ready = (
        checks["ollama_reachable"]
        if LLM_PROVIDER == "ollama"
        else checks["groq_key"] and checks["groq_reachable"] and checks["ollama_reachable"]
    )
    checks["ready"] = bool(
        provider_ready
        and checks["local_assets"]
        and checks["baseline_run"]
        and checks["verified_replay"]
    )
    return checks


def serve_mcp() -> None:
    create_server(EcoLoopState()).run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("baseline", "optimized", "mcp", "preflight"))
    parser.add_argument("--hours", type=int)
    args = parser.parse_args()
    if args.mode == "mcp":
        serve_mcp()
    elif args.mode == "preflight":
        checks = preflight()
        print(json.dumps(checks, indent=2))
        raise SystemExit(0 if checks["ready"] else 1)
    else:
        run_simulation(args.mode, args.hours)


if __name__ == "__main__":
    main()

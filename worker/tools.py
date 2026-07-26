from __future__ import annotations

import math
import secrets
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from fastmcp import FastMCP

JOULES_PER_KWH = 3_600_000.0
ELECTRICITY_PRICE_GBP_PER_KWH = 0.28


@dataclass
class EcoLoopState:
    model: dict[str, Any] = field(default_factory=dict)
    telemetry: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)
    action_tokens: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_action: dict[str, Any] | None = None
    baseline_summary: dict[str, Any] | None = None
    optimized_summary: dict[str, Any] | None = None


def _limits(occupied: bool) -> tuple[tuple[float, float], tuple[float, float]]:
    return ((20.0, 22.0), (23.0, 26.0)) if occupied else ((16.0, 20.0), (26.0, 30.0))


def validate_action(state: EcoLoopState, heating_c: float, cooling_c: float, reason: str) -> dict:
    latest = state.telemetry[-1]
    occupied = bool(latest["occupied"])
    preconditioning = not occupied and float(latest.get("baseline_heating_c", 0.0)) > 20.0
    heat_range, cool_range = _limits(occupied or preconditioning)
    current_heat = float(latest["current_heating_c"])
    current_cool = float(latest["current_cooling_c"])
    violations: list[str] = []

    if not all(math.isfinite(v) for v in (heating_c, cooling_c)):
        violations.append("setpoints must be finite")
    if not heat_range[0] <= heating_c <= heat_range[1]:
        violations.append(f"heating must be within {heat_range}")
    if not cool_range[0] <= cooling_c <= cool_range[1]:
        violations.append(f"cooling must be within {cool_range}")
    if cooling_c - heating_c < 2.0:
        violations.append("deadband must be at least 2 C")
    heat_in_range = heat_range[0] <= current_heat <= heat_range[1]
    cool_in_range = cool_range[0] <= current_cool <= cool_range[1]
    if (heat_in_range and abs(heating_c - current_heat) > 1.0) or (cool_in_range and abs(cooling_c - current_cool) > 1.0):
        violations.append("setpoint movement must be at most 1 C per decision")

    zones = latest.get("zones", [])
    if occupied and any(abs(float(zone["pmv"])) > 0.5 for zone in zones):
        if heating_c < current_heat or cooling_c > current_cool:
            violations.append("occupied PMV outside -0.5 to +0.5 blocks more relaxed conditioning")
    if occupied and any(float(zone["co2_ppm"]) > 1000.0 for zone in zones):
        if heating_c < current_heat or cooling_c > current_cool:
            violations.append("CO2 above 1000 ppm blocks more relaxed conditioning")

    peak_target = state.model.get("targets", {}).get("peak_kw")
    result: dict[str, Any] = {
        "valid": not violations,
        "violations": violations,
        "objective_terms": {
            "current_peak_kw": float(latest["facility_demand_w"]) / 1000.0,
            "target_peak_kw": peak_target,
            "carbon_g_per_kwh": float(latest["carbon_g_per_kwh"]),
            "worst_abs_pmv": max((abs(float(zone["pmv"])) for zone in zones), default=0.0),
        },
    }
    if not violations:
        token = secrets.token_urlsafe(12)
        action = {"heating_c": heating_c, "cooling_c": cooling_c, "reason": reason}
        state.action_tokens[token] = action
        result.update(token=token, action=action)
    state.audit.append({"tool": "evaluate_action", "input": {"heating_c": heating_c, "cooling_c": cooling_c, "reason": reason}, "result": result})
    return result


def summarize(samples: list[dict[str, Any]]) -> dict[str, float]:
    if not samples:
        raise ValueError("cannot summarize an empty run")
    total_j = sum(float(sample["facility_energy_j"]) for sample in samples)
    hvac_j = sum(float(sample["hvac_energy_j"]) for sample in samples)
    carbon_kg = sum(
        float(sample["facility_energy_j"]) / JOULES_PER_KWH
        * float(sample["carbon_g_per_kwh"]) / 1000.0
        for sample in samples
    )
    occupied_zones = [
        zone
        for sample in samples if sample["occupied"]
        for zone in sample.get("zones", [])
    ]
    pmv_ok = [abs(float(zone["pmv"])) <= 0.5 for zone in occupied_zones]
    temp_ok = [20.0 <= float(zone["temperature_c"]) <= 26.0 for zone in occupied_zones]
    total_kwh = total_j / JOULES_PER_KWH
    return {
        "total_kwh": total_kwh,
        "estimated_cost_gbp": total_kwh * ELECTRICITY_PRICE_GBP_PER_KWH,
        "hvac_kwh": hvac_j / JOULES_PER_KWH,
        "peak_kw": max(float(sample["facility_demand_w"]) for sample in samples) / 1000.0,
        "carbon_kg": carbon_kg,
        "pmv_compliance_pct": 100.0 * mean(pmv_ok) if pmv_ok else 100.0,
        "temperature_compliance_pct": 100.0 * mean(temp_ok) if temp_ok else 100.0,
    }


def _percent(new: float, old: float) -> float:
    return 0.0 if old == 0 else 100.0 * (new - old) / old


def compare(baseline: dict[str, float], optimized: dict[str, float]) -> dict[str, float]:
    pmv_delta = optimized["pmv_compliance_pct"] - baseline["pmv_compliance_pct"]
    baseline_cost = baseline.get(
        "estimated_cost_gbp",
        baseline["total_kwh"] * ELECTRICITY_PRICE_GBP_PER_KWH,
    )
    optimized_cost = optimized.get(
        "estimated_cost_gbp",
        optimized["total_kwh"] * ELECTRICITY_PRICE_GBP_PER_KWH,
    )
    return {
        "energy_change_pct": _percent(optimized["total_kwh"], baseline["total_kwh"]),
        "cost_change_pct": _percent(optimized_cost, baseline_cost),
        "cost_savings_gbp": baseline_cost - optimized_cost,
        "peak_change_pct": _percent(optimized["peak_kw"], baseline["peak_kw"]),
        "carbon_change_pct": _percent(optimized["carbon_kg"], baseline["carbon_kg"]),
        "pmv_compliance_delta_pct": pmv_delta,
        "temperature_compliance_delta_pct": optimized["temperature_compliance_pct"] - baseline["temperature_compliance_pct"],
        "thermal_comfort_pass": optimized["pmv_compliance_pct"] >= 95.0,
        "comfort_budget_pass": optimized["pmv_compliance_pct"] >= 95.0 and pmv_delta >= -1.0,
    }


def create_server(state: EcoLoopState) -> FastMCP:
    mcp = FastMCP("Eco-Loop", instructions="Inspect, read telemetry/errors, validate, then apply safe setpoints.")

    @mcp.tool
    def inspect_model() -> dict:
        """Return zones, People objects, schedules, sensors, actuators, and run period."""
        result = state.model
        state.audit.append({"tool": "inspect_model", "result": result})
        return result

    @mcp.tool
    def read_telemetry() -> dict:
        """Return latest telemetry and the previous hour of observations."""
        latest = state.telemetry[-1] if state.telemetry else None
        rolling = state.telemetry[-1:] if state.telemetry else []
        result = {"latest": latest, "rolling": rolling, "targets": state.model.get("targets", {})}
        state.audit.append({"tool": "read_telemetry", "result": result})
        return result

    @mcp.tool
    def read_runtime_errors() -> dict:
        """Return unique recent runtime errors and fallback events."""
        result = {"count": len(state.errors), "recent": state.errors[-10:]}
        state.audit.append({"tool": "read_runtime_errors", "result": result})
        return result

    @mcp.tool
    def evaluate_action(heating_c: float, cooling_c: float, reason: str) -> dict:
        """Validate proposed setpoints and issue a single-use action token."""
        return validate_action(state, heating_c, cooling_c, reason)

    @mcp.tool
    def apply_setpoints(token: str) -> dict:
        """Queue a previously validated action for the active EnergyPlus callback."""
        action = state.action_tokens.pop(token, None)
        if action is None:
            result = {"applied": False, "error": "invalid or already-used action token"}
        else:
            state.pending_action = action
            state.actions.append(action)
            result = {"applied": True, "action": action}
        state.audit.append({"tool": "apply_setpoints", "input": {"token": token}, "result": result})
        return result

    @mcp.tool
    def compare_runs() -> dict:
        """Compare measured baseline and optimized summaries."""
        if state.baseline_summary is None or state.optimized_summary is None:
            return {"ready": False}
        result = {"ready": True, **compare(state.baseline_summary, state.optimized_summary)}
        state.audit.append({"tool": "compare_runs", "result": result})
        return result

    return mcp

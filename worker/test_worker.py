import copy
import unittest

from worker.prepare_model import configure_epjson
from worker.tools import EcoLoopState, compare, create_server, summarize, validate_action


class PrepareModelTests(unittest.TestCase):
    def setUp(self):
        self.model = {
            "Version": {"Version 1": {"version_identifier": "25.1"}},
            "Timestep": {"Timestep 1": {"number_of_timesteps_per_hour": 1}},
            "RunPeriod": {"Annual": {}},
            "Building": {},
            "Schedule:Compact": {
                "Htg-SetP-Sch": {"schedule_type_limits_name": "Temperature", "data": [16.7, 22.2]},
                "Clg-SetP-Sch": {"schedule_type_limits_name": "Temperature", "data": [29.4, 23.9]},
            },
            "ThermostatSetpoint:DualSetpoint": {
                "DualSetPoint": {
                    "heating_setpoint_temperature_schedule_name": "Htg-SetP-Sch",
                    "cooling_setpoint_temperature_schedule_name": "Clg-SetP-Sch",
                }
            },
            "People": {
                "SPACE1-1 People": {
                    "zone_or_zonelist_or_space_or_spacelist_name": "SPACE1-1",
                    "activity_level_schedule_name": "ACTIVITY_SCH",
                }
            },
        }

    def test_configure_model_adds_required_outputs(self):
        result = configure_epjson(copy.deepcopy(self.model), control_ready=False)
        self.assertEqual(result["Timestep"]["Timestep 1"]["number_of_timesteps_per_hour"], 4)
        self.assertEqual(result["RunPeriod"]["Annual"]["begin_month"], 7)
        names = {item["variable_name"] for item in result["Output:Variable"].values()}
        self.assertIn("Zone Mean Air Temperature", names)
        self.assertIn("Zone Thermal Comfort Fanger Model PMV", names)
        self.assertIn("Zone Air CO2 Concentration", names)

    def test_control_ready_model_keeps_same_physics(self):
        baseline = configure_epjson(copy.deepcopy(self.model), control_ready=False)
        controlled = configure_epjson(copy.deepcopy(self.model), control_ready=True)
        self.assertEqual(baseline["Building"], controlled["Building"])
        schedules = controlled["Schedule:Compact"]
        self.assertEqual(schedules["HTGSETP_SCH"], schedules["Htg-SetP-Sch"])
        self.assertEqual(schedules["CLGSETP_SCH"], schedules["Clg-SetP-Sch"])
        thermostat = controlled["ThermostatSetpoint:DualSetpoint"]["DualSetPoint"]
        self.assertEqual(thermostat["heating_setpoint_temperature_schedule_name"], "HTGSETP_SCH")
        self.assertEqual(thermostat["cooling_setpoint_temperature_schedule_name"], "CLGSETP_SCH")

    def test_build_replay_model_rejects_empty_schedule(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from worker.prepare_model import build_replay_model

        with TemporaryDirectory() as directory:
            models = Path(directory)
            schedule = models / "optimized-setpoints.csv"
            schedule.write_text("hour,heating_c,cooling_c,reason\n")
            with patch("worker.prepare_model.MODELS", models):
                with self.assertRaisesRegex(ValueError, "no setpoint rows"):
                    build_replay_model(schedule)

    def test_annual_schedule_aligns_replay_to_july_15(self):
        from worker.prepare_model import annualize_schedule

        week = [{"hour": str(hour)} for hour in range(168)]
        annual = annualize_schedule(week)
        self.assertEqual(len(annual), 8760)
        self.assertEqual(annual[4680], week[0])
        self.assertEqual(annual[4679], week[-1])


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.state = EcoLoopState(
            model={"zones": ["SPACE1-1"], "people": ["SPACE1-1 People"]},
            telemetry=[{
                "occupied": True,
                "current_heating_c": 21.0,
                "current_cooling_c": 24.0,
                "baseline_heating_c": 22.2,
                "baseline_cooling_c": 23.9,
                "facility_energy_j": 3_600_000.0,
                "hvac_energy_j": 1_800_000.0,
                "facility_demand_w": 10_000.0,
                "carbon_g_per_kwh": 150.0,
                "zones": [{"pmv": 0.1, "co2_ppm": 800.0, "temperature_c": 24.0}],
            }],
        )

    def test_validate_action_accepts_safe_change(self):
        result = validate_action(self.state, 21.0, 25.0, "Reduce cooling")
        self.assertTrue(result["valid"])
        self.assertIn(result["token"], self.state.action_tokens)

    def test_validate_action_rejects_deadband_and_jump(self):
        result = validate_action(self.state, 22.0, 23.0, "Unsafe")
        self.assertFalse(result["valid"])
        self.assertIn("deadband", " ".join(result["violations"]))

    def test_high_co2_blocks_more_relaxed_cooling(self):
        self.state.telemetry[-1]["zones"][0]["co2_ppm"] = 1200.0
        result = validate_action(self.state, 21.0, 25.0, "Save energy")
        self.assertFalse(result["valid"])
        self.assertIn("CO2", " ".join(result["violations"]))

    def test_bad_pmv_allows_corrective_conditioning(self):
        self.state.telemetry[-1]["zones"][0]["pmv"] = 0.8
        result = validate_action(self.state, 21.0, 23.0, "Restore comfort")
        self.assertTrue(result["valid"])

    def test_validate_action_allows_transition_from_out_of_range(self):
        # ponytail: occupied-to-unoccupied transition leaves current=22, unoccupied max=20; range re-entry must skip movement gate.
        state = EcoLoopState(
            model={"targets": {}},
            telemetry=[{
                "occupied": False,
                "current_heating_c": 22.0,
                "current_cooling_c": 24.0,
                "facility_demand_w": 10_000.0,
                "carbon_g_per_kwh": 150.0,
                "zones": [{"pmv": 0.0, "co2_ppm": 800.0}],
            }],
        )
        result = validate_action(state, 20.0, 26.0, "transition to unoccupied range")
        self.assertTrue(result["valid"])

    def test_validate_action_allows_compact_schedule_preconditioning(self):
        state = EcoLoopState(
            model={"targets": {}},
            telemetry=[{
                "occupied": False,
                "current_heating_c": 16.7,
                "current_cooling_c": 29.4,
                "baseline_heating_c": 22.2,
                "baseline_cooling_c": 23.9,
                "facility_demand_w": 10_000.0,
                "carbon_g_per_kwh": 150.0,
                "zones": [{"pmv": 0.0, "co2_ppm": 800.0}],
            }],
        )
        self.assertTrue(validate_action(state, 22.0, 24.1, "baseline preconditioning")["valid"])

    def test_summarize_and_compare_use_measured_values(self):
        baseline = summarize(self.state.telemetry * 2)
        optimized_samples = [copy.deepcopy(self.state.telemetry[0]) for _ in range(2)]
        for sample in optimized_samples:
            sample["facility_energy_j"] *= 0.9
            sample["facility_demand_w"] *= 0.8
        result = compare(baseline, summarize(optimized_samples))
        self.assertAlmostEqual(result["energy_change_pct"], -10.0)
        self.assertAlmostEqual(result["peak_change_pct"], -20.0)
        self.assertAlmostEqual(baseline["estimated_cost_gbp"], 0.56)
        self.assertAlmostEqual(result["cost_savings_gbp"], 0.056)
        self.assertAlmostEqual(result["cost_change_pct"], -10.0)
        self.assertTrue(result["thermal_comfort_pass"])

    def test_compare_reports_comfort_budget(self):
        baseline = {
            "total_kwh": 100, "peak_kw": 10, "carbon_kg": 10,
            "pmv_compliance_pct": 99, "temperature_compliance_pct": 98,
        }
        valid = {**baseline, "total_kwh": 95, "pmv_compliance_pct": 98}
        low = {**valid, "pmv_compliance_pct": 94}
        self.assertTrue(compare(baseline, valid)["comfort_budget_pass"])
        self.assertFalse(compare(baseline, low)["comfort_budget_pass"])

    def test_finalize_rejects_comfort_regression_before_replacing_demo(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from worker.prepare_model import finalize_artifacts

        summary = {
            "total_kwh": 100.0, "hvac_kwh": 10.0, "peak_kw": 10.0,
            "carbon_kg": 10.0, "pmv_compliance_pct": 99.0,
            "temperature_compliance_pct": 98.0,
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs/baseline").mkdir(parents=True)
            (root / "runs/optimized").mkdir(parents=True)
            (root / "web").mkdir()
            (root / "runs/baseline/run.json").write_text(json.dumps({"summary": summary, "metadata": {"verified": True}}))
            optimized = {
                "summary": {**summary, "total_kwh": 95.0, "pmv_compliance_pct": 94.0},
                "actions": [{"heating_c": 21.0, "cooling_c": 25.0}],
                "metadata": {"verified": True},
            }
            (root / "runs/optimized/run.json").write_text(json.dumps(optimized))
            demo = root / "web/demo-run.json"
            demo.write_text("old verified data")
            with patch("worker.prepare_model.ROOT", root):
                with self.assertRaisesRegex(ValueError, "comfort boundary"):
                    finalize_artifacts()
            self.assertEqual(demo.read_text(), "old verified data")

    def test_finalize_rejects_no_savings_or_actions_before_replacing_demo(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from worker.prepare_model import finalize_artifacts

        summary = {
            "total_kwh": 100.0, "hvac_kwh": 10.0, "peak_kw": 10.0,
            "carbon_kg": 10.0, "pmv_compliance_pct": 99.0,
            "temperature_compliance_pct": 98.0,
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs/baseline").mkdir(parents=True)
            (root / "runs/optimized").mkdir(parents=True)
            (root / "web").mkdir()
            baseline = {"summary": summary, "metadata": {"verified": True}}
            optimized = {
                "summary": summary,
                "actions": [],
                "metadata": {"verified": True},
            }
            (root / "runs/baseline/run.json").write_text(json.dumps(baseline))
            (root / "runs/optimized/run.json").write_text(json.dumps(optimized))
            demo = root / "web/demo-run.json"
            demo.write_text("old verified data")
            with patch("worker.prepare_model.ROOT", root):
                with self.assertRaisesRegex(ValueError, "electricity savings"):
                    finalize_artifacts()
            self.assertEqual(demo.read_text(), "old verified data")

    def test_finalize_preserves_replay_when_model_generation_fails(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from worker.prepare_model import finalize_artifacts

        baseline_summary = {
            "total_kwh": 100.0, "hvac_kwh": 10.0, "peak_kw": 10.0,
            "carbon_kg": 10.0, "pmv_compliance_pct": 99.0,
            "temperature_compliance_pct": 98.0,
        }
        optimized_summary = {**baseline_summary, "total_kwh": 95.0, "pmv_compliance_pct": 98.0}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            (root / "runs/baseline").mkdir(parents=True)
            (root / "runs/optimized").mkdir(parents=True)
            (root / "web").mkdir()
            models.mkdir()
            (root / "runs/baseline/run.json").write_text(json.dumps({
                "summary": baseline_summary, "metadata": {"verified": True},
            }))
            (root / "runs/optimized/run.json").write_text(json.dumps({
                "summary": optimized_summary,
                "metadata": {"verified": True},
                "actions": [{"heating_c": 21.0, "cooling_c": 25.0}],
                "telemetry": [{"hour": 0, "current_heating_c": 21.0, "current_cooling_c": 25.0}],
            }))
            demo = root / "web/demo-run.json"
            demo.write_text("old verified data")
            with patch("worker.prepare_model.ROOT", root), \
                 patch("worker.prepare_model.MODELS", models), \
                 patch("worker.prepare_model.build_replay_model", side_effect=RuntimeError("conversion failed")):
                with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                    finalize_artifacts()
            self.assertEqual(demo.read_text(), "old verified data")

    def test_fastmcp_in_memory_tool_call(self):
        import asyncio
        from fastmcp import Client

        async def call():
            async with Client(create_server(self.state)) as client:
                result = await client.call_tool("read_telemetry", {})
                return result.data

        result = asyncio.run(call())
        self.assertEqual(result["latest"]["facility_demand_w"], 10_000.0)

    def test_server_exposes_exact_tool_surface(self):
        import asyncio
        from fastmcp import Client

        expected = {
            "inspect_model",
            "read_telemetry",
            "read_runtime_errors",
            "evaluate_action",
            "apply_setpoints",
            "compare_runs",
        }

        async def names():
            async with Client(create_server(self.state)) as client:
                tools = await client.list_tools()
                return {tool.name for tool in tools}

        self.assertEqual(asyncio.run(names()), expected)


class AgentTests(unittest.TestCase):
    def test_provider_defaults_are_groq_primary_and_qwen3_fallback(self):
        import json
        import os
        import subprocess
        import sys

        env = os.environ.copy()
        for key in ("LLM_PROVIDER", "GROQ_MODEL", "OLLAMA_MODEL", "OLLAMA_TIMEOUT_SECONDS"):
            env.pop(key, None)
        output = subprocess.check_output([
            sys.executable,
            "-c",
            "import json, worker.run as r; print(json.dumps([r.LLM_PROVIDER, r.GROQ_MODEL, r.OLLAMA_MODEL, r.OLLAMA_TIMEOUT_SECONDS]))",
        ], env=env, text=True)

        self.assertEqual(json.loads(output), ["groq", "openai/gpt-oss-120b", "qwen3:8b", 120])

    def test_tool_schema_maps_mcp_definition(self):
        from worker.run import tool_schema

        class Tool:
            name = "read_telemetry"
            description = "Read data"
            inputSchema = {"type": "object", "properties": {}}

        self.assertEqual(tool_schema(Tool())["function"]["name"], "read_telemetry")

    def test_groq_request_uses_native_tools_without_leaking_key(self):
        import json
        from unittest.mock import patch
        from worker.run import groq_request

        captured = {}

        def open_request(request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.headers["Authorization"]
            captured["body"] = json.loads(request.data)
            return json.dumps({"choices": [{"message": {
                "tool_calls": [{"function": {"name": "read_telemetry", "arguments": "{}"}}]
            }}]}).encode()

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-secret"}):
            message = groq_request([{"role": "user", "content": "observe"}], [{
                "type": "function", "function": {
                    "name": "read_telemetry", "parameters": {"type": "object"}
                }
            }], opener=open_request)
        self.assertEqual(captured["url"], "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(captured["authorization"], "Bearer test-secret")
        self.assertEqual(captured["body"]["tool_choice"], "required")
        self.assertFalse(captured["body"]["parallel_tool_calls"])
        self.assertEqual(captured["body"]["reasoning_effort"], "low")
        self.assertFalse(captured["body"]["include_reasoning"])
        self.assertEqual(captured["body"]["max_completion_tokens"], 1024)
        self.assertEqual(captured["body"]["temperature"], 0.2)
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "read_telemetry")
        self.assertNotIn("test-secret", repr(message))

    def test_groq_request_requires_key(self):
        import os
        from unittest.mock import patch
        from worker.run import groq_request

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY"):
                groq_request([], [])

    def test_ollama_request_disables_thinking_and_normalizes_history(self):
        import json

        from worker.run import ollama_request

        captured = {}

        def open_request(request, timeout):
            captured["timeout"] = timeout
            captured.update(json.loads(request.data))
            return b'{"message":{"role":"assistant","content":"ok"}}'

        history = [{
            "role": "tool",
            "name": "inspect_model",
            "content": "{}",
            "tool_call_id": "call-1",
        }, {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-2",
                "function": {"name": "read_telemetry", "arguments": "{}"},
            }],
        }]
        tools = [{"type": "function", "function": {"name": "inspect_model"}}]

        ollama_request(history, tools, opener=open_request)
        self.assertFalse(captured["think"])
        self.assertEqual(captured["timeout"], 120)
        self.assertEqual(captured["messages"][0]["tool_name"], "inspect_model")
        self.assertEqual(captured["messages"][0]["tool_call_id"], "call-1")
        self.assertEqual(captured["messages"][1]["tool_calls"][0]["function"]["arguments"], {})
        self.assertEqual(captured["tools"], tools)

    def test_groq_request_strips_ollama_only_history_fields(self):
        import json
        from unittest.mock import patch

        from worker.run import groq_request

        captured = {}

        def open_request(request, timeout):
            captured.update(json.loads(request.data))
            return b'{"choices":[{"message":{"content":"ok"}}]}'

        history = [{
            "role": "tool",
            "name": "inspect_model",
            "tool_name": "inspect_model",
            "content": "{}",
            "tool_call_id": "call-1",
        }, {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-2",
                "function": {"name": "read_telemetry", "arguments": {}},
            }],
        }]
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-secret"}):
            groq_request(history, [], opener=open_request)

        self.assertNotIn("tool_name", captured["messages"][0])
        self.assertEqual(captured["messages"][0]["tool_call_id"], "call-1")
        self.assertEqual(captured["messages"][1]["tool_calls"][0]["type"], "function")
        self.assertEqual(captured["messages"][1]["tool_calls"][0]["function"]["arguments"], "{}")

    def test_groq_request_classifies_empty_tool_use_failed_response(self):
        import json
        from io import BytesIO
        from urllib.error import HTTPError
        from unittest.mock import patch

        from worker.run import ProviderHTTPError, groq_request

        body = json.dumps({"error": {
            "message": "Failed to call a function.",
            "type": "invalid_request_error",
            "code": "tool_use_failed",
            "failed_generation": "",
        }}).encode()

        def open_request(request, timeout):
            raise HTTPError(request.full_url, 400, "Bad Request", {}, BytesIO(body))

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-secret"}):
            with self.assertRaises(ProviderHTTPError) as raised:
                groq_request([], [], opener=open_request)

        self.assertEqual(raised.exception.code, "tool_use_failed")
        self.assertNotIn("test-secret", str(raised.exception))

    def test_model_request_retries_tool_use_failed_before_ollama(self):
        from unittest.mock import patch

        from worker.run import ProviderHTTPError, model_request

        success = {"content": "", "tool_calls": [{
            "id": "call-1",
            "function": {"name": "inspect_model", "arguments": "{}"},
        }]}
        with patch("worker.run.LLM_PROVIDER", "groq"), \
             patch("worker.run.groq_request", side_effect=[
                 ProviderHTTPError("groq", 400, "tool_use_failed", "invalid tool call"),
                 success,
             ]) as groq, \
             patch("worker.run.ollama_request") as ollama:
            reply = model_request([], [])

        self.assertEqual(groq.call_count, 2)
        self.assertEqual([call.kwargs["temperature"] for call in groq.call_args_list], [0.2, 0.3])
        ollama.assert_not_called()
        self.assertEqual(reply["_provider"], "groq")
        self.assertEqual(reply["_attempts"], 2)
        self.assertFalse(reply["_fallback"])
        self.assertEqual(reply["_error_codes"], ["tool_use_failed"])

    def test_model_request_uses_ollama_once_after_three_tool_failures(self):
        from unittest.mock import patch

        from worker.run import ProviderHTTPError, model_request

        failure = ProviderHTTPError("groq", 400, "tool_use_failed", "invalid tool call")
        fallback = {"content": '{"name":"inspect_model","arguments":{}}'}
        with patch("worker.run.LLM_PROVIDER", "groq"), \
             patch("worker.run.groq_request", side_effect=[failure, failure, failure]) as groq, \
             patch("worker.run.ollama_request", return_value=fallback) as ollama:
            reply = model_request([], [{"type": "function"}])

        self.assertEqual(groq.call_count, 3)
        self.assertEqual([call.kwargs["temperature"] for call in groq.call_args_list], [0.2, 0.3, 0.4])
        ollama.assert_called_once_with([], [{"type": "function"}])
        self.assertEqual(reply["_provider"], "ollama")
        self.assertEqual(reply["_attempts"], 3)
        self.assertTrue(reply["_fallback"])
        self.assertEqual(reply["_error_codes"], [
            "tool_use_failed", "tool_use_failed", "tool_use_failed",
        ])

    def test_model_request_does_not_hide_non_retryable_groq_error(self):
        from unittest.mock import patch

        from worker.run import ProviderHTTPError, model_request

        error = ProviderHTTPError("groq", 400, None, "unsupported property")
        with patch("worker.run.LLM_PROVIDER", "groq"), \
             patch("worker.run.groq_request", side_effect=error) as groq, \
             patch("worker.run.ollama_request") as ollama:
            with self.assertRaisesRegex(ProviderHTTPError, "unsupported property"):
                model_request([], [])

        groq.assert_called_once()
        ollama.assert_not_called()

    def test_model_request_transport_failure_falls_back_once(self):
        from unittest.mock import patch

        from worker.run import model_request

        fallback = {"content": '{"name":"inspect_model","arguments":{}}'}
        with patch("worker.run.LLM_PROVIDER", "groq"), \
             patch("worker.run.groq_request", side_effect=TimeoutError("slow")) as groq, \
             patch("worker.run.ollama_request", return_value=fallback) as ollama:
            reply = model_request([], [])

        groq.assert_called_once()
        ollama.assert_called_once()
        self.assertEqual(reply["_provider"], "ollama")

    def test_model_request_preserves_primary_failure_when_fallback_fails(self):
        from unittest.mock import patch

        from worker.run import ProviderHTTPError, model_request

        failure = ProviderHTTPError("groq", 400, "tool_use_failed", "invalid tool call")
        with patch("worker.run.LLM_PROVIDER", "groq"), \
             patch("worker.run.groq_request", side_effect=[failure, failure, failure]), \
             patch("worker.run.ollama_request", side_effect=RuntimeError("local unavailable")):
            with self.assertRaisesRegex(
                RuntimeError,
                r"Groq primary failed \(tool_use_failed\); Ollama fallback failed: local unavailable",
            ) as raised:
                model_request([], [])

        self.assertIs(raised.exception.__cause__, failure)

    def test_ollama_request_rejects_missing_message(self):
        from worker.run import ollama_request

        with self.assertRaises(ValueError):
            ollama_request([], [], opener=lambda *_: b"{}")

    def test_ollama_request_sends_native_tool_schemas(self):
        import json

        from worker.run import ollama_request

        captured = {}

        def open_request(request, timeout):
            captured.update(json.loads(request.data))
            return b'{"message":{"role":"assistant","content":""}}'

        tools = [{"type": "function", "function": {"name": "read_telemetry"}}]
        ollama_request([], tools, opener=open_request)
        self.assertEqual(captured["tools"], tools)

    @staticmethod
    def tool_call(name, arguments, provider="groq"):
        return {"role": "assistant", "content": "", "tool_calls": [{
            "id": f"call-{name}",
            "function": {"name": name, "arguments": arguments}
        }], "_provider": provider, "_attempts": 1, "_fallback": provider == "ollama"}

    @staticmethod
    def agent_state():
        return EcoLoopState(
            model={"zones": ["SPACE1-1"], "targets": {"peak_kw": 12.0}},
            telemetry=[{
                "occupied": True,
                "current_heating_c": 21.0,
                "current_cooling_c": 24.0,
                "baseline_heating_c": 22.2,
                "baseline_cooling_c": 23.9,
                "facility_demand_w": 10_000.0,
                "carbon_g_per_kwh": 150.0,
                "zones": [{"pmv": 0.1, "co2_ppm": 800.0, "temperature_c": 24.0}],
            }],
        )

    def test_agent_chooses_setpoints_through_bounded_tool_loop(self):
        import asyncio
        from unittest.mock import patch

        from worker.run import agent_decision

        state = self.agent_state()
        responses = iter([
            self.tool_call("inspect_model", {}),
            self.tool_call("read_telemetry", {}),
            self.tool_call("read_runtime_errors", {}),
            self.tool_call("evaluate_action", {
                "heating_c": 21.0,
                "cooling_c": 25.0,
                "reason": "Use comfort headroom",
            }),
            self.tool_call("apply_setpoints", {"token": "TOKEN"}),
        ])

        def respond(messages, tools):
            self.assertEqual(
                {tool["function"]["name"] for tool in tools},
                {"inspect_model", "read_telemetry", "read_runtime_errors",
                 "evaluate_action", "apply_setpoints", "compare_runs"},
            )
            if len(messages) > 2:
                tool_messages = [message for message in messages if message["role"] == "tool"]
                self.assertTrue(tool_messages)
                self.assertIn("tool_call_id", tool_messages[-1])
            response = next(responses)
            if response["tool_calls"][0]["function"]["name"] == "apply_setpoints":
                token = next(iter(state.action_tokens))
                response["tool_calls"][0]["function"]["arguments"]["token"] = token
            return response

        with patch("worker.run.model_request", side_effect=respond):
            self.assertTrue(asyncio.run(agent_decision(state, "first timestep")))
        self.assertEqual(state.actions[-1]["heating_c"], 21.0)
        self.assertEqual(state.actions[-1]["cooling_c"], 25.0)

    def test_groq_tool_message_omits_ollama_tool_name(self):
        from worker.run import _tool_message

        message = _tool_message("inspect_model", {"zones": []}, "call-1", "groq")

        self.assertNotIn("tool_name", message)
        self.assertEqual(message["name"], "inspect_model")
        self.assertEqual(message["tool_call_id"], "call-1")

    def test_agent_responds_to_every_parallel_tool_call_id(self):
        from worker.run import _surplus_tool_messages

        calls = [
            {"id": "call-first", "function": {"name": "read_telemetry"}},
            {"id": "call-surplus", "function": {"name": "read_runtime_errors"}},
        ]

        messages = _surplus_tool_messages(
            calls,
            selected_id="call-first",
            provider="groq",
            expected="read_telemetry",
        )

        self.assertEqual([message["tool_call_id"] for message in messages], ["call-surplus"])
        self.assertEqual(messages[0]["name"], "read_runtime_errors")
        self.assertIn("one tool call", messages[0]["content"])

    def test_agent_completes_parallel_history_but_executes_only_required_tool(self):
        import asyncio
        from unittest.mock import patch

        from worker.run import agent_decision

        state = self.agent_state()
        responses = iter([
            self.tool_call("inspect_model", {}),
            {
                **self.tool_call("read_telemetry", {}),
                "tool_calls": [
                    {"id": "call-read", "function": {"name": "read_telemetry", "arguments": {}}},
                    {"id": "call-extra", "function": {"name": "read_runtime_errors", "arguments": {}}},
                ],
            },
            self.tool_call("read_runtime_errors", {}),
            self.tool_call("evaluate_action", {
                "heating_c": 21.0,
                "cooling_c": 25.0,
                "reason": "Use comfort headroom",
            }),
            self.tool_call("apply_setpoints", {"token": "TOKEN"}),
        ])
        saw_complete_pair = False

        def respond(messages, tools):
            nonlocal saw_complete_pair
            ids = {m.get("tool_call_id") for m in messages if m["role"] == "tool"}
            if {"call-read", "call-extra"}.issubset(ids):
                saw_complete_pair = True
            response = next(responses)
            if response["tool_calls"][0]["function"]["name"] == "apply_setpoints":
                response["tool_calls"][0]["function"]["arguments"]["token"] = next(iter(state.action_tokens))
            return response

        with patch("worker.run.model_request", side_effect=respond):
            self.assertTrue(asyncio.run(agent_decision(state, "first timestep")))

        self.assertTrue(saw_complete_pair)
        self.assertEqual(
            [entry["tool"] for entry in state.audit].count("read_runtime_errors"),
            1,
        )

    def test_agent_redirects_unmatched_native_calls_instead_of_executing_content_json(self):
        import asyncio
        import json
        from unittest.mock import patch

        from worker.run import agent_decision

        state = self.agent_state()
        responses = iter([
            {
                **self.tool_call("read_telemetry", {}),
                "content": json.dumps({"name": "inspect_model", "arguments": {}}),
                "tool_calls": [
                    {"id": "call-wrong-1", "function": {"name": "read_telemetry", "arguments": {}}},
                    {"id": "call-wrong-2", "function": {"name": "read_runtime_errors", "arguments": {}}},
                ],
            },
            self.tool_call("inspect_model", {}),
            self.tool_call("read_telemetry", {}),
            self.tool_call("read_runtime_errors", {}),
            self.tool_call("evaluate_action", {
                "heating_c": 21.0,
                "cooling_c": 25.0,
                "reason": "Use comfort headroom",
            }),
            self.tool_call("apply_setpoints", {"token": "TOKEN"}),
        ])
        saw_redirect = False

        def respond(messages, tools):
            nonlocal saw_redirect
            if len([message for message in messages if message["role"] == "assistant"]) == 1:
                ids = {m.get("tool_call_id") for m in messages if m["role"] == "tool"}
                saw_redirect = {"call-wrong-1", "call-wrong-2"}.issubset(ids)
                self.assertFalse(any(entry["tool"] == "inspect_model" for entry in state.audit))
            response = next(responses)
            if response["tool_calls"][0]["function"]["name"] == "apply_setpoints":
                response["tool_calls"][0]["function"]["arguments"]["token"] = next(iter(state.action_tokens))
            return response

        with patch("worker.run.model_request", side_effect=respond):
            self.assertTrue(asyncio.run(agent_decision(state, "first timestep")))

        self.assertTrue(saw_redirect)

    def test_missing_native_tool_call_id_gets_stable_turn_id(self):
        from worker.run import _ensure_tool_call_ids

        calls = [{"function": {"name": "inspect_model", "arguments": {}}}]
        normalized = _ensure_tool_call_ids(calls, turn=2)

        self.assertEqual(normalized[0]["id"], "call-turn-2-0")
        self.assertNotIn("id", calls[0])

    def test_decision_history_keeps_complete_tool_pairs(self):
        from worker.run import _trim_messages

        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "start"},
        ]
        for index in range(8):
            call_id = f"call-{index}"
            messages.extend([
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": call_id,
                    "function": {"name": "inspect_model", "arguments": "{}"},
                }]},
                {"role": "tool", "name": "inspect_model", "tool_call_id": call_id, "content": "{}"},
            ])

        self.assertEqual(_trim_messages(messages), messages)

    def test_agent_audits_primary_and_fallback_model_calls(self):
        import asyncio
        from unittest.mock import patch

        from worker.run import agent_decision

        state = self.agent_state()
        fallback = self.tool_call("inspect_model", {}, provider="ollama")
        fallback["_attempts"] = 3
        fallback["_error_codes"] = ["tool_use_failed", "tool_use_failed", "tool_use_failed"]
        responses = iter([
            fallback,
            self.tool_call("read_telemetry", {}),
            self.tool_call("read_runtime_errors", {}),
            self.tool_call("evaluate_action", {
                "heating_c": 21.0,
                "cooling_c": 25.0,
                "reason": "Use comfort headroom",
            }),
            self.tool_call("apply_setpoints", {"token": "TOKEN"}),
        ])

        def respond(messages, tools):
            response = next(responses)
            if response["tool_calls"][0]["function"]["name"] == "apply_setpoints":
                response["tool_calls"][0]["function"]["arguments"]["token"] = next(iter(state.action_tokens))
            return response

        with patch("worker.run.OLLAMA_MODEL", "qwen3:8b"), \
             patch("worker.run.model_request", side_effect=respond):
            self.assertTrue(asyncio.run(agent_decision(state, "first timestep")))

        provider_events = [entry for entry in state.audit if entry["tool"] == "model_provider"]
        self.assertEqual(provider_events[0]["result"], {
            "provider": "ollama",
            "model": "qwen3:8b",
            "attempts": 3,
            "fallback": True,
            "error_codes": ["tool_use_failed", "tool_use_failed", "tool_use_failed"],
        })
        self.assertEqual(provider_events[1]["result"]["provider"], "groq")

    def test_agent_rejects_evaluate_before_observation(self):
        import asyncio
        import json
        from unittest.mock import patch

        from worker.run import agent_decision

        state = self.agent_state()
        bad_call = json.dumps({"name": "evaluate_action", "arguments": {"heating_c": 21.0, "cooling_c": 25.0, "reason": "too early"}})
        with patch("worker.run.model_request", return_value={"role": "assistant", "content": bad_call}):
            self.assertFalse(asyncio.run(agent_decision(state, "first timestep")))
        self.assertEqual(state.actions, [])
        self.assertIn("turn limit", state.errors[-1]["message"])

    def test_agent_allows_one_validation_correction(self):
        import asyncio
        import json
        from unittest.mock import patch

        from worker.run import agent_decision

        state = self.agent_state()
        queue = [
            json.dumps({"name": "inspect_model", "arguments": {}}),
            json.dumps({"name": "read_telemetry", "arguments": {}}),
            json.dumps({"name": "read_runtime_errors", "arguments": {}}),
            json.dumps({"name": "evaluate_action", "arguments": {"heating_c": 22.0, "cooling_c": 23.0, "reason": "invalid deadband"}}),
            json.dumps({"name": "evaluate_action", "arguments": {"heating_c": 21.0, "cooling_c": 25.0, "reason": "corrected"}}),
            None,
        ]

        def respond(messages, _tools):
            assistant_messages = [m for m in messages if m["role"] == "assistant"]
            index = min(len(assistant_messages), len(queue) - 1)
            content = queue[index]
            if content is None:
                token = next(iter(state.action_tokens))
                content = json.dumps({"name": "apply_setpoints", "arguments": {"token": token}})
            return {"role": "assistant", "content": content}

        with patch("worker.run.model_request", side_effect=respond):
            self.assertTrue(asyncio.run(agent_decision(state, "first timestep")))
        evaluations = [entry for entry in state.audit if entry["tool"] == "evaluate_action"]
        self.assertEqual(len(evaluations), 2)
        self.assertEqual(len(state.actions), 1)

    def test_agent_recovers_from_format_errors_before_valid_correction(self):
        import asyncio
        import json
        from unittest.mock import patch

        from worker.run import agent_decision

        state = self.agent_state()
        queue = [
            {"content": "not json"},
            {"content": json.dumps({"name": "inspect_model", "arguments": {}})},
            {"content": json.dumps({"name": "read_telemetry", "arguments": {}})},
            {"content": json.dumps({"name": "read_runtime_errors", "arguments": {}})},
            {"content": json.dumps({"name": "evaluate_action", "arguments": {
                "heating_c": 22.0, "cooling_c": 23.0, "reason": "invalid"
            }})},
            {"content": "still not json"},
            {"content": "not json either"},
            {"content": json.dumps({"name": "evaluate_action", "arguments": {
                "heating_c": 21.0, "cooling_c": 25.0, "reason": "corrected"
            }})},
            {"content": "APPLY"},
        ]

        def respond(messages, tools):
            response = queue.pop(0)
            if response["content"] == "APPLY":
                response["content"] = json.dumps({
                    "name": "apply_setpoints",
                    "arguments": {"token": next(iter(state.action_tokens))},
                })
            return response

        with patch("worker.run.model_request", side_effect=respond):
            self.assertTrue(asyncio.run(agent_decision(state, "first timestep")))
        self.assertEqual(state.errors, [])
        self.assertGreaterEqual(
            len([entry for entry in state.audit if entry["tool"] == "model_retry"]),
            3,
        )

    def test_agent_malformed_response_keeps_safe_action(self):
        import asyncio
        from unittest.mock import patch

        from worker.run import agent_decision

        state = self.agent_state()
        state.pending_action = {"heating_c": 20.0, "cooling_c": 26.0, "reason": "safe"}
        with patch("worker.run.model_request", return_value={"role": "assistant", "content": "text"}):
            self.assertFalse(asyncio.run(agent_decision(state, "heartbeat")))
        self.assertEqual(state.pending_action["heating_c"], 20.0)

    def test_parse_model_response_accepts_json_content(self):
        from worker.run import _parse_model_response
        payload = _parse_model_response(
            {"role": "assistant", "content": '```json\n{"name": "inspect_model", "arguments": {}}\n```'},
            {"inspect_model", "evaluate_action"},
        )
        self.assertEqual(payload, {"name": "inspect_model", "arguments": {}})

    def test_parse_model_response_extracts_setpoints(self):
        from worker.run import _parse_model_response
        payload = _parse_model_response(
            {"role": "assistant", "content": '{"name":"evaluate_action","setpoints":{"heating_c":21.0,"cooling_c":25.0},"reason":"headroom"}'},
            {"inspect_model", "evaluate_action", "apply_setpoints"},
        )
        self.assertEqual(payload, {"name": "evaluate_action", "arguments": {"heating_c": 21.0, "cooling_c": 25.0, "reason": "headroom"}})

    def test_parse_model_response_rejects_unknown_tool(self):
        from worker.run import _parse_model_response
        self.assertIsNone(_parse_model_response({"role": "assistant", "content": '{"name":"unknown"}'}, {"inspect_model"}))
        self.assertIsNone(_parse_model_response({"role": "assistant", "content": ""}, {"inspect_model"}))

    def test_parse_model_response_accepts_omitted_arguments_for_zero_argument_tool(self):
        from worker.run import _parse_model_response

        result = _parse_model_response(
            {"role": "assistant", "content": '{"name":"read_telemetry"}'},
            {"read_telemetry", "evaluate_action"},
            {"read_telemetry"},
        )
        self.assertEqual(result, {"name": "read_telemetry", "arguments": {}})

    def test_parse_model_response_drops_spurious_zero_argument_fields(self):
        from worker.run import _parse_model_response

        result = _parse_model_response(
            {"role": "assistant", "content": '{"name":"inspect_model","arguments":{"run_period":1}}'},
            {"inspect_model", "evaluate_action"},
            {"inspect_model"},
        )
        self.assertEqual(result, {"name": "inspect_model", "arguments": {}})

    def test_parse_model_response_defaults_missing_action_reason(self):
        from worker.run import _parse_model_response

        result = _parse_model_response({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {
                "name": "evaluate_action",
                "arguments": {"heating_c": 16.7, "cooling_c": 29.4},
            }}],
        }, {"evaluate_action"})
        self.assertEqual(result["arguments"]["reason"], "model proposal")

    def test_parse_model_response_rejects_omitted_arguments_for_parameterized_tool(self):
        from worker.run import _parse_model_response

        self.assertIsNone(_parse_model_response(
            {"role": "assistant", "content": '{"name":"evaluate_action"}'},
            {"read_telemetry", "evaluate_action"},
            {"read_telemetry"},
        ))

    def test_model_tool_result_compacts_telemetry(self):
        from worker.run import _model_tool_result

        result = _model_tool_result("read_telemetry", {
            "latest": {
                "hour": 1,
                "occupied": False,
                "current_heating_c": 16.7,
                "current_cooling_c": 29.4,
                "facility_demand_w": 475.0,
                "zones": [{"name": "SPACE1-1", "pmv": -0.8, "co2_ppm": 700, "temperature_c": 20, "humidity_pct": 60}],
            },
            "rolling": [{"irrelevant": "large"}],
            "targets": {"pmv": [-0.5, 0.5]},
        })
        self.assertEqual(result["latest"]["current_heating_c"], 16.7)
        self.assertNotIn("humidity_pct", result["zones"][0])
        self.assertNotIn("rolling", result)

    def test_agent_redirects_compare_runs_to_required_observation(self):
        import asyncio
        import json
        from unittest.mock import patch

        from worker.run import agent_decision

        state = self.agent_state()
        queue = iter([
            {"content": json.dumps({"name": "compare_runs", "arguments": {}})},
            {"content": json.dumps({"name": "inspect_model", "arguments": {}})},
            {"content": json.dumps({"name": "read_telemetry", "arguments": {}})},
            {"content": json.dumps({"name": "read_runtime_errors", "arguments": {}})},
            {"content": json.dumps({"name": "evaluate_action", "arguments": {
                "heating_c": 21.0, "cooling_c": 25.0, "reason": "safe"
            }})},
            {"content": "APPLY"},
        ])

        def respond(messages, tools):
            response = next(queue)
            if response["content"] == "APPLY":
                response["content"] = json.dumps({
                    "name": "apply_setpoints",
                    "arguments": {"token": next(iter(state.action_tokens))},
                })
            return response

        with patch("worker.run.model_request", side_effect=respond):
            self.assertTrue(asyncio.run(agent_decision(state, "first controllable timestep")))
        self.assertFalse(any(entry.get("tool") == "compare_runs" for entry in state.audit))

    def test_agent_handles_json_content_tool_call(self):
        import asyncio
        import json
        from unittest.mock import patch

        from worker.run import agent_decision

        state = self.agent_state()
        queue = [
            json.dumps({"name": "inspect_model", "arguments": {}}),
            json.dumps({"name": "read_telemetry", "arguments": {}}),
            json.dumps({"name": "read_runtime_errors", "arguments": {}}),
            json.dumps({"name": "evaluate_action", "arguments": {"heating_c": 21.0, "cooling_c": 25.0, "reason": "ok"}}),
            None,
        ]

        def respond(messages, _tools):
            assistant_messages = [m for m in messages if m["role"] == "assistant"]
            index = min(len(assistant_messages), len(queue) - 1)
            content = queue[index]
            if content is None:
                token = next(iter(state.action_tokens))
                content = json.dumps({"name": "apply_setpoints", "arguments": {"token": token}})
            return {"role": "assistant", "content": content}

        with patch("worker.run.model_request", side_effect=respond):
            self.assertTrue(asyncio.run(agent_decision(state, "first timestep")))
        self.assertEqual(state.actions[-1]["heating_c"], 21.0)


class SimulationTests(unittest.TestCase):
    def test_failed_context_change_restores_baseline_setpoints(self):
        from worker.run import restore_baseline_after_failure

        state = EcoLoopState(
            telemetry=[{
                "hour": 8,
                "baseline_heating_c": 22.2,
                "baseline_cooling_c": 23.9,
            }],
            pending_action={"heating_c": 16.7, "cooling_c": 29.4, "reason": "stale unoccupied action"},
        )

        self.assertTrue(restore_baseline_after_failure(state, "occupancy changed"))
        self.assertEqual(state.pending_action, {
            "heating_c": 22.2,
            "cooling_c": 23.9,
            "reason": "baseline fallback after occupancy changed",
        })
        self.assertEqual(state.audit[-1]["tool"], "baseline_fallback")

    def test_failed_heartbeat_keeps_last_validated_action(self):
        from worker.run import restore_baseline_after_failure

        action = {"heating_c": 20.0, "cooling_c": 26.0, "reason": "validated"}
        state = EcoLoopState(
            telemetry=[{
                "baseline_heating_c": 16.7,
                "baseline_cooling_c": 29.4,
            }],
            pending_action=action.copy(),
        )

        self.assertFalse(restore_baseline_after_failure(state, "three-hour heartbeat"))
        self.assertEqual(state.pending_action, action)

    def test_optimized_run_with_agent_errors_is_rejected(self):
        from worker.run import validate_run_acceptance

        state = EcoLoopState(
            errors=[{"source": "agent", "message": "failed"}],
            actions=[{"heating_c": 21.0, "cooling_c": 25.0}],
        )
        with self.assertRaisesRegex(RuntimeError, "agent"):
            validate_run_acceptance("optimized", state)

    def test_control_sanitizes_provider_error_before_persisting(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from worker.run import ProviderHTTPError, Simulation

        simulation = Simulation.__new__(Simulation)
        simulation.mode = "optimized"
        simulation.api = SimpleNamespace(
            exchange=SimpleNamespace(
                warmup_flag=lambda _: False,
                kind_of_sim=lambda _: 3,
            ),
        )
        simulation.state = EcoLoopState(
            model={"targets": {}},
            telemetry=[{"hour": 0}],
        )
        simulation.max_hours = 1
        simulation.last_control_hour = -1
        simulation.last_decision_hour = -1
        simulation.last_decision_sample = None

        detail = (
            "Rate limit for organization org-secret: 8000 TPM; "
            "Authorization: Bearer credential-secret; response body"
        )
        error = ProviderHTTPError("groq", 429, None, detail)
        with patch.object(simulation, "initialize_handles"), \
             patch.object(simulation, "_hour", return_value=0), \
             patch("worker.run.decision_trigger", return_value="first timestep"), \
             patch("worker.run.agent_decision", new=lambda *_: None), \
             patch("worker.run.asyncio.run", side_effect=error):
            simulation.control("state")

        self.assertEqual(simulation.state.errors, [{
            "source": "agent",
            "provider": "groq",
            "status": 429,
            "code": "http_error",
            "message": "groq request failed (HTTP 429)",
        }])
        persisted = repr(simulation.state.errors)
        for secret in ("org-secret", "8000", "Authorization", "credential-secret", "response body"):
            self.assertNotIn(secret, persisted)

    def test_optimized_run_without_actions_is_rejected(self):
        from worker.run import validate_run_acceptance

        with self.assertRaisesRegex(RuntimeError, "no validated"):
            validate_run_acceptance("optimized", EcoLoopState())

    def test_baseline_run_ignores_acceptance_gates(self):
        from worker.run import validate_run_acceptance

        validate_run_acceptance("baseline", EcoLoopState(errors=[{"source": "x"}]))

    def test_guarded_callback_stops_and_records_exception(self):
        from types import SimpleNamespace

        from worker.run import Simulation

        stopped = []
        simulation = Simulation.__new__(Simulation)
        simulation.api = SimpleNamespace(runtime=SimpleNamespace(stop_simulation=stopped.append))
        simulation.callback_error = None
        wrapped = simulation._guard_callback(lambda _: (_ for _ in ()).throw(RuntimeError("callback failed")))

        wrapped("state")

        self.assertEqual(str(simulation.callback_error), "callback failed")
        self.assertEqual(stopped, ["state"])

    def test_full_run_finishes_at_168_hours(self):
        from worker.run import Simulation

        simulation = Simulation.__new__(Simulation)
        simulation.max_hours = None
        simulation.state = EcoLoopState(model={"run_days": 7})

        self.assertFalse(simulation._finished(167))
        self.assertTrue(simulation._finished(168))

    def test_current_payload_reports_primary_and_fallback_usage(self):
        from unittest.mock import patch

        from worker.run import Simulation

        simulation = Simulation.__new__(Simulation)
        simulation.mode = "optimized"
        simulation.state = EcoLoopState(
            telemetry=[{
                "occupied": False,
                "facility_energy_j": 1,
                "hvac_energy_j": 0,
                "facility_demand_w": 1,
                "carbon_g_per_kwh": 1,
                "zones": [],
            }],
            audit=[
                {"tool": "model_provider", "result": {"provider": "groq", "fallback": False}},
                {"tool": "model_provider", "result": {"provider": "ollama", "fallback": True}},
            ],
        )
        with patch("worker.run.LLM_PROVIDER", "groq"), \
             patch("worker.run.GROQ_MODEL", "openai/gpt-oss-120b"), \
             patch("worker.run.OLLAMA_MODEL", "qwen3:8b"):
            metadata = simulation.current_payload()["metadata"]

        self.assertEqual(metadata["primary_provider"], "groq")
        self.assertEqual(metadata["primary_model"], "openai/gpt-oss-120b")
        self.assertEqual(metadata["fallback_provider"], "ollama")
        self.assertEqual(metadata["fallback_model"], "qwen3:8b")
        self.assertEqual(metadata["fallback_calls"], 1)

    def test_preflight_reports_required_checks_without_secret(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from worker.run import preflight

        def open_request(request, timeout):
            if "groq.com" in request.full_url:
                return json.dumps({"choices": [{"message": {"tool_calls": [{
                    "id": "call-preflight",
                    "function": {"name": "inspect_model", "arguments": "{}"},
                }]}}]}).encode()
            return json.dumps({"models": [{"name": "qwen3:8b"}]}).encode()

        def groq_ok_but_ollama_missing(request, timeout):
            if "groq.com" in request.full_url:
                return open_request(request, timeout)
            return b'{"models":[]}'

        def invalid_groq_arguments(arguments_marker):
            def open_invalid(request, timeout):
                if "groq.com" not in request.full_url:
                    return json.dumps({"models": [{"name": "qwen3:8b"}]}).encode()
                function = {"name": "inspect_model"}
                if arguments_marker is not None:
                    function["arguments"] = arguments_marker
                return json.dumps({"choices": [{"message": {"tool_calls": [{
                    "id": "call-preflight",
                    "function": function,
                }]}}]}).encode()
            return open_invalid

        with TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            runs = root / "runs"
            web = root / "web"
            models.mkdir()
            (runs / "baseline").mkdir(parents=True)
            web.mkdir()
            for name in (
                "baseline.idf", "control-ready.idf", "optimized.idf",
                "weather.epw", "carbon-intensity.json",
            ):
                (models / name).write_text("valid")
            (models / "optimized-setpoints.csv").write_text(
                "hour,heating_c,cooling_c,reason\n0,20,26,test\n"
            )
            (runs / "baseline/run.json").write_text("{}")
            (web / "demo-run.json").write_text('{"metadata":{"verified":true}}')
            with patch.dict("os.environ", {"GROQ_API_KEY": "hidden"}), \
                 patch("worker.run.LLM_PROVIDER", "groq"), \
                 patch("worker.run.OLLAMA_MODEL", "qwen3:8b"), \
                 patch("worker.run.MODELS", models), \
                 patch("worker.run.RUNS", runs), \
                 patch("worker.run.ROOT", root):
                missing = preflight(opener=groq_ok_but_ollama_missing)
                malformed = preflight(opener=invalid_groq_arguments("not-json"))
                absent = preflight(opener=invalid_groq_arguments(None))
                wrong = preflight(opener=invalid_groq_arguments('{"unexpected":true}'))
                result = preflight(opener=open_request)
        self.assertIn("groq_key", result)
        self.assertIn("groq_reachable", result)
        self.assertIn("ollama_reachable", result)
        self.assertIn("local_assets", result)
        self.assertTrue(result["groq_key"])
        self.assertTrue(result["groq_reachable"])
        self.assertTrue(result["ollama_reachable"])
        self.assertTrue(result["local_assets"])
        self.assertFalse(missing["ready"])
        self.assertFalse(malformed["groq_reachable"])
        self.assertFalse(absent["groq_reachable"])
        self.assertFalse(wrong["groq_reachable"])
        self.assertNotIn("not-json", json.dumps(malformed))
        self.assertTrue(result["ready"])
        self.assertNotIn("hidden", json.dumps(result))

    def test_preflight_local_mode_does_not_require_groq(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from worker.run import preflight

        with TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            runs = root / "runs"
            web = root / "web"
            models.mkdir()
            (runs / "baseline").mkdir(parents=True)
            web.mkdir()
            for name in (
                "baseline.idf", "control-ready.idf", "optimized.idf",
                "weather.epw", "carbon-intensity.json",
            ):
                (models / name).write_text("valid")
            (models / "optimized-setpoints.csv").write_text(
                "hour,heating_c,cooling_c,reason\n0,20,26,test\n"
            )
            (runs / "baseline/run.json").write_text("{}")
            (web / "demo-run.json").write_text('{"metadata":{"verified":true}}')
            with patch.dict("os.environ", {}, clear=True), \
                 patch("worker.run.LLM_PROVIDER", "ollama"), \
                 patch("worker.run.OLLAMA_MODEL", "qwen3:8b"), \
                 patch("worker.run.MODELS", models), \
                 patch("worker.run.RUNS", runs), \
                 patch("worker.run.ROOT", root):
                missing = preflight(opener=lambda *_: b'{"models":[]}')
                result = preflight(opener=lambda *_: b'{"models":[{"name":"qwen3:8b"}]}')
        self.assertFalse(missing["ready"])
        self.assertTrue(result["ready"])
        self.assertFalse(result["groq_key"])
        self.assertFalse(result["groq_reachable"])
        self.assertTrue(result["ollama_reachable"])

    def test_collect_uses_neutral_pmv_for_zone_without_people(self):
        from types import SimpleNamespace

        from worker.run import Simulation

        class Exchange:
            def warmup_flag(self, _):
                return False

            def kind_of_sim(self, _):
                return 3

            def current_sim_time(self, _):
                return 1.0

            def get_variable_value(self, _, handle):
                if handle == -1:
                    raise AssertionError("collect read missing EnergyPlus handle")
                return float(handle)

            def get_meter_value(self, _, _handle):
                return 8.0

        exchange = Exchange()
        simulation = Simulation.__new__(Simulation)
        simulation.api = SimpleNamespace(exchange=exchange, runtime=SimpleNamespace())
        simulation.mode = "baseline"
        simulation.max_hours = None
        simulation.initialized = True
        simulation.carbon = {"hourly": [100.0]}
        simulation.state = EcoLoopState()
        simulation.handles = {
            "zones": {
                "PLENUM-1": {
                    "temperature": 1,
                    "humidity": 2,
                    "co2": 3,
                    "occupancy": -1,
                    "pmv": -1,
                }
            },
            "hvac_energy": 8,
            "demand": 4,
            "outdoor": 5,
            "heating_sp": 6,
            "cooling_sp": 7,
            "baseline_heating": 9,
            "baseline_cooling": 10,
        }

        simulation.collect(object())

        zone = simulation.state.telemetry[0]["zones"][0]
        self.assertEqual(zone["occupants"], 0.0)
        self.assertEqual(zone["pmv"], 0.0)

    @staticmethod
    def trigger_sample(**overrides):
        sample = {
            "occupied": True,
            "baseline_heating_c": 22.2,
            "baseline_cooling_c": 23.9,
            "facility_demand_w": 10_000.0,
            "carbon_g_per_kwh": 150.0,
            "zones": [{"pmv": 0.1, "co2_ppm": 800.0}],
        }
        sample.update(overrides)
        return sample

    def test_decision_trigger_uses_three_hour_heartbeat(self):
        from worker.run import decision_trigger

        sample = self.trigger_sample()
        self.assertIsNone(decision_trigger(sample, sample, 2, 20.0))
        self.assertEqual(decision_trigger(sample, sample, 3, 20.0), "three-hour heartbeat")

    def test_decision_trigger_detects_state_changes(self):
        from worker.run import decision_trigger

        before = self.trigger_sample()
        occupied = {**before, "occupied": not before["occupied"]}
        carbon = {**before, "carbon_g_per_kwh": before["carbon_g_per_kwh"] + 50}
        self.assertEqual(decision_trigger(occupied, before, 1, 20.0), "occupancy changed")
        self.assertEqual(decision_trigger(carbon, before, 1, 20.0), "carbon band changed")
        schedule = {**before, "baseline_heating_c": before["baseline_heating_c"] + 0.1}
        self.assertEqual(decision_trigger(schedule, before, 1, 20.0), "baseline schedule changed")

    def test_decision_trigger_detects_new_constraint_violation(self):
        from worker.run import decision_trigger

        before = self.trigger_sample()
        bad = copy.deepcopy(before)
        bad["zones"][0]["pmv"] = 0.7
        self.assertEqual(decision_trigger(bad, before, 1, 20.0), "comfort target violated")
        high_co2 = copy.deepcopy(before)
        high_co2["zones"][0]["co2_ppm"] = 1500.0
        self.assertEqual(decision_trigger(high_co2, before, 1, 20.0), "comfort target violated")

    def test_decision_trigger_detects_peak_crossing(self):
        from worker.run import decision_trigger

        before = self.trigger_sample(facility_demand_w=11_000.0)
        spike = {**before, "facility_demand_w": 19_500.0}
        self.assertEqual(decision_trigger(spike, before, 1, 18.0), "peak target crossed")

    def test_decision_trigger_returns_none_for_steady_state(self):
        from worker.run import decision_trigger

        sample = self.trigger_sample()
        self.assertIsNone(decision_trigger(sample, sample, 1, 20.0))


if __name__ == "__main__":
    unittest.main()

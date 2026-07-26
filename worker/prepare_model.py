from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
EPLUS = Path(os.environ.get("ENERGYPLUS_DIR", "/EnergyPlus-25.1.0-68a4a7c774-Linux-Ubuntu22.04-x86_64"))
SOURCE_IDF_URL = "https://raw.githubusercontent.com/NREL/EnergyPlus/v25.1.0/testfiles/5ZoneAirCooled.idf"
SOURCE_EPW_URL = "https://energyplus-weather.s3.amazonaws.com/europe_wmo_region_6/GBR/GBR_London.Gatwick.037760_IWEC/GBR_London.Gatwick.037760_IWEC.epw"
SOURCE_IDF = EPLUS / "ExampleFiles" / "5ZoneAirCooled.idf"
SOURCE_EPW = EPLUS / "WeatherData" / "GBR_London.Gatwick.037760_IWEC.epw"

OUTPUTS = (
    ("Zone Mean Air Temperature", "*"),
    ("Zone Air Relative Humidity", "*"),
    ("Zone Air CO2 Concentration", "*"),
    ("Zone People Occupant Count", "*"),
    ("Zone Thermostat Heating Setpoint Temperature", "*"),
    ("Zone Thermostat Cooling Setpoint Temperature", "*"),
    ("Zone Thermal Comfort Fanger Model PMV", "*"),
    ("Schedule Value", "Htg-SetP-Sch"),
    ("Schedule Value", "Clg-SetP-Sch"),
    ("Facility Total Electricity Demand Rate", "Whole Building"),
    ("Site Outdoor Air Drybulb Temperature", "Environment"),
)


def configure_epjson(data: dict, control_ready: bool) -> dict:
    data.setdefault("Building", {})
    timestep = next(iter(data.setdefault("Timestep", {"Timestep 1": {}}).values()))
    timestep["number_of_timesteps_per_hour"] = 4

    period = next(iter(data.setdefault("RunPeriod", {"EcoLoop Week": {}}).values()))
    period.update(
        begin_month=7,
        begin_day_of_month=15,
        end_month=7,
        end_day_of_month=21,
        use_weather_file_holidays_and_special_days="Yes",
        use_weather_file_daylight_saving_period="Yes",
    )

    data["ZoneAirContaminantBalance"] = {
        "EcoLoop CO2": {
            "carbon_dioxide_concentration": "Yes",
            "outdoor_carbon_dioxide_schedule_name": "EcoLoop Outdoor CO2",
        }
    }
    schedules = data.setdefault("Schedule:Constant", {})
    schedules["EcoLoop Outdoor CO2"] = {"schedule_type_limits_name": "Any Number", "hourly_value": 420.0}
    schedules["EcoLoop Work Efficiency"] = {"schedule_type_limits_name": "Any Number", "hourly_value": 0.0}
    schedules["EcoLoop Clothing"] = {"schedule_type_limits_name": "Any Number", "hourly_value": 0.7}
    schedules["EcoLoop Air Velocity"] = {"schedule_type_limits_name": "Any Number", "hourly_value": 0.1}

    if control_ready:
        # Keep untouched compact schedules as references; actuate copies to avoid circular overrides.
        compact = data["Schedule:Compact"]
        compact["HTGSETP_SCH"] = copy.deepcopy(compact["Htg-SetP-Sch"])
        compact["CLGSETP_SCH"] = copy.deepcopy(compact["Clg-SetP-Sch"])
        for obj in data.get("ThermostatSetpoint:SingleHeating", {}).values():
            if obj.get("setpoint_temperature_schedule_name") == "Htg-SetP-Sch":
                obj["setpoint_temperature_schedule_name"] = "HTGSETP_SCH"
        for obj in data.get("ThermostatSetpoint:SingleCooling", {}).values():
            if obj.get("setpoint_temperature_schedule_name") == "Clg-SetP-Sch":
                obj["setpoint_temperature_schedule_name"] = "CLGSETP_SCH"
        for obj in data.get("ThermostatSetpoint:DualSetpoint", {}).values():
            if obj.get("heating_setpoint_temperature_schedule_name") == "Htg-SetP-Sch":
                obj["heating_setpoint_temperature_schedule_name"] = "HTGSETP_SCH"
            if obj.get("cooling_setpoint_temperature_schedule_name") == "Clg-SetP-Sch":
                obj["cooling_setpoint_temperature_schedule_name"] = "CLGSETP_SCH"

    for person in data.get("People", {}).values():
        person["carbon_dioxide_generation_rate"] = 3.82e-8
        person["work_efficiency_schedule_name"] = "EcoLoop Work Efficiency"
        person["clothing_insulation_calculation_method"] = "ClothingInsulationSchedule"
        person["clothing_insulation_schedule_name"] = "EcoLoop Clothing"
        person["air_velocity_schedule_name"] = "EcoLoop Air Velocity"
        person["thermal_comfort_model_1_type"] = "Fanger"

    data["Output:Variable"] = {
        f"EcoLoop Variable {index}": {
            "key_value": key,
            "variable_name": variable,
            "reporting_frequency": "Timestep",
        }
        for index, (variable, key) in enumerate(OUTPUTS, 1)
    }
    data["Output:Meter"] = {
        "EcoLoop Facility Electricity": {
            "key_name": "Electricity:Facility",
            "reporting_frequency": "Timestep",
        },
        "EcoLoop HVAC Electricity": {
            "key_name": "Electricity:HVAC",
            "reporting_frequency": "Timestep",
        },
    }
    data["Output:SQLite"] = {"EcoLoop SQLite": {"option_type": "SimpleAndTabular"}}
    return data


def convert(path: Path) -> Path:
    target = path.with_suffix(".epJSON" if path.suffix.lower() == ".idf" else ".idf")
    target.unlink(missing_ok=True)
    subprocess.run(["energyplus", "--convert-only", path.name], cwd=path.parent, check=True)
    if not target.exists():
        raise FileNotFoundError(f"EnergyPlus did not create {target}")
    return target


def write_idf(epjson: dict, path: Path) -> None:
    json_path = path.with_suffix(".epJSON")
    json_path.write_text(json.dumps(epjson, indent=2), encoding="utf-8")
    generated = convert(json_path)
    generated.replace(path)


def _download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as response:
        path.write_bytes(response.read())


def ensure_source_assets() -> None:
    _download(SOURCE_IDF_URL, SOURCE_IDF)
    _download(SOURCE_EPW_URL, SOURCE_EPW)


def fetch_carbon(path: Path) -> None:
    url = "https://api.carbonintensity.org.uk/intensity/2025-07-15T00:00Z/2025-07-22T00:00Z"
    with urllib.request.urlopen(url, timeout=20) as response:
        payload = json.load(response)
    half_hours = [row["intensity"]["actual"] or row["intensity"]["forecast"] for row in payload["data"]]
    hourly = [round(sum(half_hours[i : i + 2]) / len(half_hours[i : i + 2]), 1) for i in range(0, len(half_hours), 2)]
    result = {
        "source": "UK NESO Carbon Intensity API",
        "url": url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "units": "gCO2/kWh",
        "hourly": hourly[:168],
    }
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def prepare() -> None:
    MODELS.mkdir(exist_ok=True)
    ensure_source_assets()
    shutil.copy2(SOURCE_IDF, MODELS / "source.idf")
    shutil.copy2(SOURCE_EPW, MODELS / "weather.epw")
    source_json = json.loads(convert(MODELS / "source.idf").read_text(encoding="utf-8"))
    baseline = configure_epjson(copy.deepcopy(source_json), control_ready=False)
    controlled = configure_epjson(copy.deepcopy(source_json), control_ready=True)
    write_idf(baseline, MODELS / "baseline.idf")
    write_idf(controlled, MODELS / "control-ready.idf")
    fetch_carbon(MODELS / "carbon-intensity.json")


from worker.tools import compare

REPLAY_START_HOUR = 195 * 24  # July 15 in a non-leap year.


def annualize_schedule(data: list[dict]) -> list[dict]:
    if not data:
        raise ValueError("optimized schedule has no setpoint rows")
    return [data[(hour - REPLAY_START_HOUR) % len(data)] for hour in range(8760)]


def build_replay_model(schedule_path: Path) -> None:
    with open(MODELS / "optimized-setpoints.csv") as f:
        reader = csv.DictReader(f)
        data = list(reader)
    rows = annualize_schedule(data)
    with schedule_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("hour", "heating_c", "cooling_c", "reason"),
            lineterminator="\n",
        )
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"hour": i, "heating_c": row["heating_c"], "cooling_c": row["cooling_c"], "reason": "verified autonomous action"})
    source_json = json.loads(convert(MODELS / "control-ready.idf").read_text(encoding="utf-8"))
    for schedule_type, objects in source_json.items():
        if schedule_type.startswith("Schedule:") and isinstance(objects, dict):
            objects.pop("HTGSETP_SCH", None)
            objects.pop("CLGSETP_SCH", None)
    source_json.setdefault("Schedule:File", {}).update({
        "HTGSETP_SCH": {"schedule_type_limits_name": "Temperature", "file_name": schedule_path.name, "column_number": 2, "rows_to_skip_at_top": 1, "number_of_hours_of_data": 8760, "column_separator": "Comma", "interpolate_to_timestep": "No", "minutes_per_item": 60},
        "CLGSETP_SCH": {"schedule_type_limits_name": "Temperature", "file_name": schedule_path.name, "column_number": 3, "rows_to_skip_at_top": 1, "number_of_hours_of_data": 8760, "column_separator": "Comma", "interpolate_to_timestep": "No", "minutes_per_item": 60},
    })
    write_idf(source_json, MODELS / "optimized.idf")


def finalize_artifacts() -> dict:
    baseline = json.loads((ROOT / "runs" / "baseline" / "run.json").read_text())
    optimized = json.loads((ROOT / "runs" / "optimized" / "run.json").read_text())
    comparison = compare(baseline["summary"], optimized["summary"])
    if not baseline.get("metadata", {}).get("verified") or not optimized.get("metadata", {}).get("verified"):
        raise ValueError("baseline and optimized runs must be verified")
    if not comparison["thermal_comfort_pass"]:
        raise ValueError(
            "optimized run misses the PMV comfort boundary "
            f"(pmv={optimized['summary']['pmv_compliance_pct']:.2f}%, "
            f"delta={comparison['pmv_compliance_delta_pct']:.2f} points)"
        )
    if comparison["energy_change_pct"] >= 0:
        raise ValueError(
            "optimized run has no measured electricity savings "
            f"(delta={comparison['energy_change_pct']:.2f}%)"
        )
    if not optimized.get("actions"):
        raise ValueError("optimized run has no validated autonomous actions")
    merged = {
        "metadata": {"verified": True, "source": "EnergyPlus 25.1", "model": "5ZoneAirCooled", "weather": "London/Gatwick"},
        "baseline": baseline,
        "optimized": optimized,
        "comparison": comparison,
    }
    schedule_path = MODELS / "optimized-setpoints.csv"
    with schedule_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("hour", "heating_c", "cooling_c", "reason"),
            lineterminator="\n",
        )
        writer.writeheader()
        by_hour = {}
        for sample in optimized.get("telemetry", []):
            by_hour.setdefault(sample["hour"], sample)
        for hour, sample in sorted(by_hour.items()):
            writer.writerow({"hour": hour, "heating_c": sample["current_heating_c"], "cooling_c": sample["current_cooling_c"], "reason": "verified autonomous action"})
    build_replay_model(schedule_path)
    (ROOT / "web").mkdir(exist_ok=True)
    (ROOT / "web" / "demo-run.json").write_text(
        json.dumps(merged, separators=(",", ":")),
        encoding="utf-8",
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "finalize"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        finalize_artifacts()


if __name__ == "__main__":
    main()

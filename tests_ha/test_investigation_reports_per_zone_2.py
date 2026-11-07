"""Real Home Assistant: zones set up together split the old shared reports file (aaearon/ClimateAdvisor#2).

Occupant view: after upgrading, each zone's AI Report History shows the reports from that zone's thermostat,
a corrupt old file doesn't stop the zones from starting, and deleting a report only affects the selected zone.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.climate_advisor.const import (
    CONF_AI_API_KEY,
    CONF_AI_ENABLED,
    DOMAIN,
    INVESTIGATION_REPORTS_FILE,
)

_SEED = (
    Path(__file__).resolve().parent.parent
    / "tools"
    / "integration_harness"
    / "config-seed"
    / ".storage"
    / "core.config_entries"
)
_BACKUP = "climate_advisor_investigation_reports_legacy_shared.json"


def _report(hours_ago: int, thermostat: str) -> dict:
    return {
        "timestamp": (dt_util.now() - timedelta(hours=hours_ago)).isoformat(),
        "report_type": "investigation",
        "result": {"input_context": f"=== HVAC ENTITY ===\n  entity_id:        {thermostat}\n", "text": thermostat},
    }


async def _setup_two_zones(hass: HomeAssistant, tmp_path: Path) -> list[MockConfigEntry]:
    hass.config.config_dir = str(tmp_path)
    seed = next(e for e in json.loads(_SEED.read_text(encoding="utf-8"))["data"]["entries"] if e["domain"] == DOMAIN)
    today = dt_util.start_of_local_day()
    hass.states.async_set(
        seed["data"]["weather_entity"],
        "sunny",
        {
            "temperature": 60,
            "forecast": [
                {
                    "datetime": (today + timedelta(days=d)).isoformat(),
                    "temperature": 72,
                    "templow": 55,
                    "condition": "sunny",
                }
                for d in range(3)
            ],
        },
    )
    entries = []
    for thermostat in ("climate.a", "climate.b"):
        hass.states.async_set(
            thermostat,
            "heat",
            {"hvac_modes": ["off", "heat"], "hvac_action": "idle", "current_temperature": 70, "temperature": 70},
        )
        data = {**seed["data"], "climate_entity": thermostat, CONF_AI_ENABLED: True, CONF_AI_API_KEY: "test"}
        entry = MockConfigEntry(
            domain=DOMAIN, version=seed["version"], minor_version=seed.get("minor_version", 1), data=data
        )
        entry.add_to_hass(hass)
        entries.append(entry)
    # Setting up the integration sets up all its entries together, as a Home Assistant start does.
    assert await async_setup_component(hass, DOMAIN, {})
    await hass.async_block_till_done()
    assert [e.state for e in entries] == [ConfigEntryState.LOADED] * 2
    return entries


def _history(hass: HomeAssistant, entry: MockConfigEntry) -> list[str]:
    return [e["result"]["text"] for e in hass.data[DOMAIN][entry.entry_id].get_investigation_report_history()]


async def _unload(hass: HomeAssistant, entries: list[MockConfigEntry]) -> None:
    for entry in entries:
        assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()


async def test_zones_starting_together_each_get_their_own_old_reports(hass: HomeAssistant, tmp_path) -> None:
    legacy = tmp_path / INVESTIGATION_REPORTS_FILE
    legacy.write_text(json.dumps([_report(3, "climate.a"), _report(2, "climate.b"), _report(1, "climate.gone")]))
    original = legacy.read_bytes()

    entries = await _setup_two_zones(hass, tmp_path)

    assert _history(hass, entries[0]) == ["climate.a"]
    assert _history(hass, entries[1]) == ["climate.b"]
    assert not legacy.exists()
    assert (tmp_path / _BACKUP).read_bytes() == original
    await _unload(hass, entries)


async def test_corrupt_shared_file_does_not_stop_the_zones(hass: HomeAssistant, tmp_path) -> None:
    (tmp_path / INVESTIGATION_REPORTS_FILE).write_text("{not json")

    entries = await _setup_two_zones(hass, tmp_path)

    assert _history(hass, entries[0]) == []
    assert (tmp_path / _BACKUP).read_text() == "{not json"
    await _unload(hass, entries)


async def test_rest_delete_is_scoped_to_the_selected_zone(hass: HomeAssistant, tmp_path, hass_client) -> None:
    (tmp_path / INVESTIGATION_REPORTS_FILE).write_text(json.dumps([_report(3, "climate.a"), _report(2, "climate.b")]))
    assert await async_setup_component(hass, "http", {})
    entries = await _setup_two_zones(hass, tmp_path)
    client = await hass_client()
    stamp = hass.data[DOMAIN][entries[0].entry_id].get_investigation_report_history()[0]["timestamp"]
    body = {"report_type": "investigation", "timestamp": stamp}

    refused = await client.post("/api/climate_advisor/delete_report", json=body)
    assert refused.status == 400
    assert _history(hass, entries[0]) == ["climate.a"]

    wrong_zone = await client.post(f"/api/climate_advisor/delete_report?entry_id={entries[1].entry_id}", json=body)
    assert (await wrong_zone.json())["found"] is False
    assert _history(hass, entries[0]) == ["climate.a"]

    deleted = await client.post(f"/api/climate_advisor/delete_report?entry_id={entries[0].entry_id}", json=body)
    assert (await deleted.json())["found"] is True
    assert _history(hass, entries[0]) == []
    assert _history(hass, entries[1]) == ["climate.b"]
    await _unload(hass, entries)

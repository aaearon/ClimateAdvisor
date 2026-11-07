"""Investigation reports belong to one zone (aaearon/ClimateAdvisor#2).

Every zone used to read and write one shared file, each overwriting it with its own in-memory list.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Thread
from unittest.mock import MagicMock

import pytest

if "homeassistant" not in sys.modules:
    from conftest import _install_ha_stubs

    _install_ha_stubs()

from custom_components.climate_advisor import coordinator as coordinator_module  # noqa: E402
from custom_components.climate_advisor.const import DOMAIN, INVESTIGATION_REPORTS_FILE  # noqa: E402


def _hass(config_dir: Path) -> MagicMock:
    async def _run_inline(fn, *args):
        return fn(*args)

    hass = MagicMock()
    hass.config.config_dir = str(config_dir)
    hass.config.path = lambda *parts: str(config_dir.joinpath(*parts))
    hass.async_add_executor_job = _run_inline
    return hass


def _zone(config_dir: Path, entry_id: str, climate_entity: str = "climate.any"):
    """A real coordinator's report code on a bare instance: restart = building a new one and loading."""
    coord = object.__new__(coordinator_module.ClimateAdvisorCoordinator)
    coord.hass = _hass(config_dir)
    coord._entry_id = entry_id
    coord.config = {"climate_entity": climate_entity}
    coord._investigation_report_history = []
    coord._load_investigation_reports()
    return coord


def _texts(coord) -> list[str]:
    return [e["result"]["text"] for e in coord.get_investigation_report_history()]


class TestZonesKeepTheirOwnReports:
    def test_second_zone_save_does_not_lose_the_first_zones_report(self, tmp_path):
        """Occupant impact: a report on one zone is still there after a restart, whatever another zone stored."""
        zone_a, zone_b = _zone(tmp_path, "entry_a"), _zone(tmp_path, "entry_b")

        asyncio.run(zone_a.async_store_investigation_report({"text": "report A"}))
        asyncio.run(zone_b.async_store_investigation_report({"text": "report B"}))

        assert _texts(_zone(tmp_path, "entry_a")) == ["report A"]
        assert _texts(_zone(tmp_path, "entry_b")) == ["report B"]

    def test_reports_are_written_to_the_entry_scoped_file_not_the_shared_one(self, tmp_path):
        zone = _zone(tmp_path, "entry_a")

        asyncio.run(zone.async_store_investigation_report({"text": "report A"}))

        assert (tmp_path / "climate_advisor_investigation_reports_entry_a.json").exists()
        assert not (tmp_path / INVESTIGATION_REPORTS_FILE).exists()

    def test_delete_then_another_zones_save_does_not_bring_it_back_or_touch_others(self, tmp_path):
        """Occupant impact: a deleted report stays deleted and the other zone's history is untouched."""
        zone_a, zone_b = _zone(tmp_path, "entry_a"), _zone(tmp_path, "entry_b")
        asyncio.run(zone_a.async_store_investigation_report({"text": "report A"}))
        asyncio.run(zone_b.async_store_investigation_report({"text": "report B"}))

        stamp = zone_a.get_investigation_report_history()[0]["timestamp"]
        assert zone_a.delete_investigation_report(stamp) is True
        zone_a._save_investigation_reports()
        asyncio.run(zone_b.async_store_investigation_report({"text": "report B2"}))

        assert _texts(_zone(tmp_path, "entry_a")) == []
        assert _texts(_zone(tmp_path, "entry_b")) == ["report B", "report B2"]


# ---------------------------------------------------------------------------
# Migration of the old shared file
# ---------------------------------------------------------------------------

_BACKUP = "climate_advisor_investigation_reports_legacy_shared.json"
_ZONES = {"entry_a": "climate.a", "entry_b": "climate.b", "entry_c": "climate.c"}


def _report(stamp: str, thermostat: str | None, **extra) -> dict:
    context = "" if thermostat is None else f"=== HVAC ENTITY ===\n  entity_id:        {thermostat}\n  hvac_mode: off\n"
    return {
        "timestamp": f"2026-11-0{stamp}T10:00:00+00:00",
        "report_type": "investigation",
        "result": {"input_context": context, "text": stamp, **extra},
    }


def _write_legacy(config_dir: Path, reports) -> bytes:
    data = json.dumps(reports).encode()
    (config_dir / INVESTIGATION_REPORTS_FILE).write_bytes(data)
    return data


def _scoped(config_dir: Path, entry_id: str) -> Path:
    return config_dir / f"climate_advisor_investigation_reports_{entry_id}.json"


def _scoped_texts(config_dir: Path, entry_id: str) -> list[str]:
    path = _scoped(config_dir, entry_id)
    return [e["result"]["text"] for e in json.loads(path.read_text())] if path.exists() else []


class TestReportThermostat:
    def test_reads_the_thermostat_from_the_real_context_builders(self):
        """Pins the parser to what the investigator really stores, so a reworded context section fails here."""
        from custom_components.climate_advisor.ai_skills_context import build_hvac_entity_context
        from custom_components.climate_advisor.diagnostic_snapshot import render_diagnostic_snapshot_text
        from custom_components.climate_advisor.investigation_report_migration import report_thermostat

        coordinator = MagicMock()
        coordinator.config = {"climate_entity": "climate.bedroom"}
        coordinator.data = {}
        coordinator.automation_engine = None
        section = asyncio.run(build_hvac_entity_context(MagicMock(), coordinator))
        footer = render_diagnostic_snapshot_text({"config_summary": {"climate_entity": "climate.bedroom"}})

        assert report_thermostat({"result": {"input_context": section}}) == "climate.bedroom"
        assert report_thermostat({"result": {"input_context": footer}}) == "climate.bedroom"
        assert report_thermostat({"result": {"input_context": section + "\n" + footer}}) == "climate.bedroom"

    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param({}, id="no_result"),
            pytest.param({"result": "text"}, id="result_not_a_dict"),
            pytest.param({"result": {}}, id="no_context"),
            pytest.param({"result": {"input_context": None}}, id="context_not_text"),
            pytest.param({"result": {"input_context": "no thermostat here"}}, id="no_section"),
            pytest.param(
                {
                    "result": {
                        "input_context": "=== HVAC ENTITY ===\n  entity_id:        climate.a\n"
                        "=== HVAC ENTITY ===\n  entity_id:        climate.b\n"
                    }
                },
                id="two_thermostats",
            ),
        ],
    )
    def test_unclear_reports_identify_nothing(self, entry):
        from custom_components.climate_advisor.investigation_report_migration import report_thermostat

        assert report_thermostat(entry) is None


class TestMigrateSharedReports:
    def _migrate(self, config_dir: Path, zones=None) -> None:
        from custom_components.climate_advisor.investigation_report_migration import migrate_shared_reports

        migrate_shared_reports(config_dir, _ZONES if zones is None else zones)

    def test_each_report_goes_only_to_the_zone_whose_thermostat_produced_it(self, tmp_path):
        """Occupant impact: after upgrading, a zone's history shows its own reports and nobody else's."""
        original = _write_legacy(
            tmp_path, [_report("1", "climate.a"), _report("2", "climate.b"), _report("3", "climate.a")]
        )

        self._migrate(tmp_path)

        assert _scoped_texts(tmp_path, "entry_a") == ["1", "3"]
        assert _scoped_texts(tmp_path, "entry_b") == ["2"]
        assert not _scoped(tmp_path, "entry_c").exists()
        assert not (tmp_path / INVESTIGATION_REPORTS_FILE).exists()
        assert (tmp_path / _BACKUP).read_bytes() == original

    @pytest.mark.parametrize(
        "thermostat",
        [pytest.param(None, id="no_context"), pytest.param("climate.removed", id="thermostat_no_zone_has")],
    )
    def test_a_report_that_cannot_be_placed_stays_in_the_backup_and_no_zone(self, tmp_path, thermostat):
        """Never guess: a report shown on the wrong zone is misleading, and it is kept on disk regardless."""
        _write_legacy(tmp_path, [_report("1", thermostat), _report("2", "climate.a")])

        self._migrate(tmp_path)

        assert _scoped_texts(tmp_path, "entry_a") == ["2"]
        assert not _scoped(tmp_path, "entry_b").exists()
        assert "1" in [e["result"]["text"] for e in json.loads((tmp_path / _BACKUP).read_text())]

    def test_two_zones_on_one_thermostat_is_not_guessed(self, tmp_path):
        _write_legacy(tmp_path, [_report("1", "climate.a")])

        self._migrate(tmp_path, {"entry_a": "climate.a", "entry_b": "climate.a"})

        assert not _scoped(tmp_path, "entry_a").exists()
        assert not _scoped(tmp_path, "entry_b").exists()
        assert (tmp_path / _BACKUP).exists()

    def test_the_only_zone_takes_everything(self, tmp_path):
        """Occupant impact: a single-zone install keeps its whole history through the upgrade."""
        _write_legacy(tmp_path, [_report("1", None), _report("2", "climate.old_thermostat")])

        self._migrate(tmp_path, {"entry_a": "climate.a"})

        assert _scoped_texts(tmp_path, "entry_a") == ["1", "2"]
        assert (tmp_path / _BACKUP).exists()

    def test_entries_the_loader_could_not_read_stay_in_the_backup_only(self, tmp_path):
        no_stamp = _report("1", "climate.a")
        del no_stamp["timestamp"]
        odd_stamp = {**_report("2", "climate.a"), "timestamp": 5}
        _write_legacy(tmp_path, [no_stamp, odd_stamp, "not a dict", _report("3", "climate.a")])

        self._migrate(tmp_path, {"entry_a": "climate.a"})

        assert _scoped_texts(tmp_path, "entry_a") == ["3"]
        assert len(json.loads((tmp_path / _BACKUP).read_text())) == 4

    def test_merges_with_reports_a_zone_already_has(self, tmp_path):
        _scoped(tmp_path, "entry_a").write_text(json.dumps([_report("2", "climate.a")]))
        _write_legacy(tmp_path, [_report("1", "climate.a"), _report("2", "climate.a")])

        self._migrate(tmp_path)

        assert _scoped_texts(tmp_path, "entry_a") == ["1", "2"]

    def test_running_again_changes_nothing(self, tmp_path):
        _write_legacy(tmp_path, [_report("1", "climate.a")])
        self._migrate(tmp_path)
        files = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

        self._migrate(tmp_path)

        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == files

    def test_crash_before_the_legacy_file_is_removed_repeats_cleanly(self, tmp_path, monkeypatch):
        original = _write_legacy(tmp_path, [_report("1", "climate.a")])
        real_unlink = Path.unlink

        def _crash(self, *args, **kwargs):
            raise OSError("crash")

        monkeypatch.setattr(Path, "unlink", _crash)
        self._migrate(tmp_path)
        assert (tmp_path / INVESTIGATION_REPORTS_FILE).exists()
        monkeypatch.setattr(Path, "unlink", real_unlink)

        self._migrate(tmp_path)

        assert _scoped_texts(tmp_path, "entry_a") == ["1"]
        assert not (tmp_path / INVESTIGATION_REPORTS_FILE).exists()
        assert (tmp_path / _BACKUP).read_bytes() == original
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted([_BACKUP, _scoped(tmp_path, "entry_a").name])

    def test_a_later_shared_file_never_overwrites_the_backup(self, tmp_path):
        """A downgrade and re-upgrade writes a new shared file; the first backup must survive it."""
        first = _write_legacy(tmp_path, [_report("1", "climate.a")])
        self._migrate(tmp_path)
        second = _write_legacy(tmp_path, [_report("2", "climate.a")])

        self._migrate(tmp_path)

        assert (tmp_path / _BACKUP).read_bytes() == first
        assert (tmp_path / "climate_advisor_investigation_reports_legacy_shared.1.json").read_bytes() == second
        assert _scoped_texts(tmp_path, "entry_a") == ["1", "2"]

    def test_failure_while_replacing_keeps_the_shared_file_for_the_next_restart(self, tmp_path, monkeypatch):
        import custom_components.climate_advisor.investigation_report_migration as migration

        original = _write_legacy(tmp_path, [_report("1", "climate.a"), _report("2", "climate.b")])
        real_replace = migration.os.replace

        def _full_disk(src, dst):
            if str(dst).endswith("_entry_b.json"):
                raise OSError("disk full")
            real_replace(src, dst)

        monkeypatch.setattr(migration.os, "replace", _full_disk)
        self._migrate(tmp_path)

        assert (tmp_path / INVESTIGATION_REPORTS_FILE).read_bytes() == original
        assert (tmp_path / _BACKUP).read_bytes() == original
        assert not list(tmp_path.glob("*.tmp"))

        monkeypatch.setattr(migration.os, "replace", real_replace)
        self._migrate(tmp_path)

        assert _scoped_texts(tmp_path, "entry_a") == ["1"]
        assert _scoped_texts(tmp_path, "entry_b") == ["2"]
        assert not (tmp_path / INVESTIGATION_REPORTS_FILE).exists()

    def _fail_staging_for(self, monkeypatch, entry_id: str) -> None:
        import custom_components.climate_advisor.investigation_report_migration as migration

        real_mkstemp = migration.tempfile.mkstemp

        def _mkstemp(*args, **kwargs):
            if entry_id in kwargs.get("prefix", ""):
                raise OSError("disk full")
            return real_mkstemp(*args, **kwargs)

        monkeypatch.setattr(migration.tempfile, "mkstemp", _mkstemp)

    def test_failed_later_zone_write_leaves_earlier_zone_files_unchanged(self, tmp_path, monkeypatch):
        existing = json.dumps([_report("9", "climate.a")])
        _scoped(tmp_path, "entry_a").write_text(existing)
        original = _write_legacy(tmp_path, [_report("1", "climate.a"), _report("2", "climate.b")])
        self._fail_staging_for(monkeypatch, "entry_b")

        self._migrate(tmp_path)

        assert _scoped(tmp_path, "entry_a").read_text() == existing
        assert not _scoped(tmp_path, "entry_b").exists()
        assert (tmp_path / INVESTIGATION_REPORTS_FILE).read_bytes() == original
        assert not list(tmp_path.glob("*.tmp"))

    def test_unparsable_zone_file_leaves_every_zone_file_unchanged(self, tmp_path):
        existing = json.dumps([_report("9", "climate.a")])
        _scoped(tmp_path, "entry_a").write_text(existing)
        _scoped(tmp_path, "entry_b").write_text("{not json")
        original = _write_legacy(tmp_path, [_report("1", "climate.a"), _report("2", "climate.b")])

        self._migrate(tmp_path)

        assert _scoped(tmp_path, "entry_a").read_text() == existing
        assert _scoped(tmp_path, "entry_b").read_text() == "{not json"
        assert (tmp_path / INVESTIGATION_REPORTS_FILE).read_bytes() == original
        assert not (tmp_path / _BACKUP).exists()
        assert not list(tmp_path.glob("*.tmp"))

    def test_report_deleted_after_a_failed_attempt_stays_deleted_after_the_retry(self, tmp_path, monkeypatch):
        """Occupant impact: whatever a zone showed after a failed migration and was deleted does not come back."""
        recent = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        older = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
        own = {**_report("9", "climate.a"), "timestamp": recent}
        shared_a = {**_report("1", "climate.a"), "timestamp": older}
        _scoped(tmp_path, "entry_a").write_text(json.dumps([own]))
        _write_legacy(tmp_path, [shared_a, _report("2", "climate.b")])
        self._fail_staging_for(monkeypatch, "entry_b")
        self._migrate(tmp_path)
        monkeypatch.undo()

        zone_a = _zone(tmp_path, "entry_a")
        deleted = [e["result"]["text"] for e in zone_a.get_investigation_report_history()]
        assert "9" in deleted
        for entry in zone_a.get_investigation_report_history():
            zone_a.delete_investigation_report(entry["timestamp"])
        zone_a._save_investigation_reports()

        self._migrate(tmp_path)

        assert not set(deleted) & set(_scoped_texts(tmp_path, "entry_a"))
        assert _scoped_texts(tmp_path, "entry_b") == ["2"]
        assert not (tmp_path / INVESTIGATION_REPORTS_FILE).exists()

    def test_save_racing_a_staged_retry_survives(self, tmp_path, monkeypatch):
        """Occupant impact: a report stored while another zone retries the migration is not overwritten."""
        import custom_components.climate_advisor.investigation_report_migration as migration

        _scoped(tmp_path, "entry_a").write_text(json.dumps([]))
        original = _write_legacy(tmp_path, [_report("1", "climate.a")])
        zone_a = _zone(tmp_path, "entry_a")
        zone_a._investigation_report_history.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "report_type": "investigation",
                "result": {"text": "saved during staging"},
            }
        )
        real_stage = migration._stage_zone_file
        save_finished = Event()

        def _save_then_finish() -> None:
            zone_a._save_investigation_reports()
            save_finished.set()

        save_thread = Thread(target=_save_then_finish)

        def _stage_after_racing_save(path, content):
            # The snapshot is already taken. Without the lock the save lands now and is then replaced; with it,
            # the save waits, so this bounded wait only ever delays the migration and can't fail the fixed code.
            save_thread.start()
            save_finished.wait(timeout=0.5)
            return real_stage(path, content)

        monkeypatch.setattr(migration, "_stage_zone_file", _stage_after_racing_save)

        self._migrate(tmp_path)
        save_thread.join(timeout=5)

        assert not save_thread.is_alive(), "save deadlocked on the migration lock"
        assert "saved during staging" in _scoped_texts(tmp_path, "entry_a")
        assert (tmp_path / _BACKUP).read_bytes() == original
        assert not (tmp_path / INVESTIGATION_REPORTS_FILE).exists()

    def test_a_zone_file_that_cannot_be_read_is_not_overwritten(self, tmp_path):
        _scoped(tmp_path, "entry_a").write_text("{not json")
        _write_legacy(tmp_path, [_report("1", "climate.a")])

        self._migrate(tmp_path)

        assert _scoped(tmp_path, "entry_a").read_text() == "{not json"
        assert (tmp_path / INVESTIGATION_REPORTS_FILE).exists()

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param(b"{not json", id="corrupt"),
            pytest.param(b"\xff\xfe\x00", id="not_text"),
            pytest.param(b'{"a": 1}', id="not_a_list"),
        ],
    )
    def test_corrupt_shared_file_is_set_aside_without_failing(self, tmp_path, content):
        (tmp_path / INVESTIGATION_REPORTS_FILE).write_bytes(content)

        self._migrate(tmp_path)

        assert (tmp_path / _BACKUP).read_bytes() == content
        assert not (tmp_path / INVESTIGATION_REPORTS_FILE).exists()
        assert not _scoped(tmp_path, "entry_a").exists()

    def test_missing_shared_file_does_nothing(self, tmp_path):
        self._migrate(tmp_path)

        assert list(tmp_path.iterdir()) == []

    def test_no_known_zones_leaves_the_shared_file_alone(self, tmp_path):
        original = _write_legacy(tmp_path, [_report("1", "climate.a")])

        self._migrate(tmp_path, {})

        assert (tmp_path / INVESTIGATION_REPORTS_FILE).read_bytes() == original
        assert not (tmp_path / _BACKUP).exists()

    def test_eight_zones_starting_at_once_each_end_up_with_their_own_reports(self, tmp_path):
        """Zones set up concurrently and file work runs on executor threads."""
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        zones = {f"entry_{i}": f"climate.z{i}" for i in range(8)}
        reports = [_report(str(i + 1), f"climate.z{i}") for i in range(8)]
        original = _write_legacy(tmp_path, reports)
        barrier = Barrier(8)

        def _start_zone() -> None:
            barrier.wait()
            self._migrate(tmp_path, zones)

        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in [pool.submit(_start_zone) for _ in range(8)]:
                future.result()

        for i in range(8):
            assert _scoped_texts(tmp_path, f"entry_{i}") == [str(i + 1)]
        assert (tmp_path / _BACKUP).read_bytes() == original
        assert sorted(p.name for p in tmp_path.glob("*legacy_shared*")) == [_BACKUP]
        assert not (tmp_path / INVESTIGATION_REPORTS_FILE).exists()

    def test_removing_a_zone_later_leaves_other_zones_and_the_backup_intact(self, tmp_path):
        _write_legacy(tmp_path, [_report("1", "climate.a"), _report("2", "climate.b")])
        self._migrate(tmp_path)

        _scoped(tmp_path, "entry_a").unlink()
        self._migrate(tmp_path, {"entry_b": "climate.b"})

        assert _scoped_texts(tmp_path, "entry_b") == ["2"]
        assert (tmp_path / _BACKUP).exists()


class TestCoordinatorMigrationStep:
    def test_storage_failure_is_logged_not_raised_into_setup(self, tmp_path, monkeypatch, caplog):
        """Occupant impact: a report-storage problem must never stop HVAC automation from starting."""

        def _boom(*args):
            raise RuntimeError("boom")

        monkeypatch.setattr(coordinator_module, "migrate_shared_reports", _boom)
        zone = _zone(tmp_path, "entry_a")

        zone._migrate_legacy_investigation_reports({"entry_a": "climate.a"})

        assert "legacy investigation reports" in caplog.text

    def test_unreadable_config_entries_mean_no_migration_this_time(self, tmp_path):
        zone = _zone(tmp_path, "entry_a")
        zone.hass.config_entries.async_entries.side_effect = RuntimeError("boom")

        assert zone._investigation_report_zones() == {}

    def test_zone_map_comes_from_every_config_entry(self, tmp_path):
        zone = _zone(tmp_path, "entry_a")
        entries = [
            MagicMock(entry_id="entry_a", data={"climate_entity": "climate.a"}),
            MagicMock(entry_id="entry_b", data={"climate_entity": "climate.b"}),
        ]
        zone.hass.config_entries.async_entries.return_value = entries

        assert zone._investigation_report_zones() == {"entry_a": "climate.a", "entry_b": "climate.b"}
        zone.hass.config_entries.async_entries.assert_called_once_with(DOMAIN, include_ignore=False)

    def test_no_entry_id_does_not_migrate(self, tmp_path):
        """The harness case: no real config entry, so the unscoped path is still that coordinator's own file."""
        original = _write_legacy(tmp_path, [_report("1", "climate.a")])
        zone = _zone(tmp_path, "")

        zone._migrate_legacy_investigation_reports({"entry_a": "climate.a"})

        assert (tmp_path / INVESTIGATION_REPORTS_FILE).read_bytes() == original


# ---------------------------------------------------------------------------
# Delete view
# ---------------------------------------------------------------------------


def _delete_request(hass: MagicMock, query: dict, timestamp: str) -> MagicMock:
    async def _json():
        return {"report_type": "investigation", "timestamp": timestamp}

    request = MagicMock()
    request.app = {"hass": hass}
    request.method = "POST"
    request.query = query
    request.json = _json
    return request


class TestDeleteView:
    def _zones(self, tmp_path):
        zone_a, zone_b = _zone(tmp_path, "entry_a"), _zone(tmp_path, "entry_b")
        asyncio.run(zone_a.async_store_investigation_report({"text": "report A"}))
        # Both zones hold a report with the same timestamp: only entry_id may decide which one goes.
        stamp = zone_a.get_investigation_report_history()[0]["timestamp"]
        zone_b._investigation_report_history = [
            {"timestamp": stamp, "report_type": "investigation", "result": {"text": "report B"}}
        ]
        zone_b._save_investigation_reports()
        hass = _hass(tmp_path)
        hass.data = {DOMAIN: {"entry_a": zone_a, "entry_b": zone_b}}
        return hass, zone_a, zone_b, stamp

    def _post(self, hass, query, stamp):
        from custom_components.climate_advisor.api import ClimateAdvisorDeleteReportView

        return asyncio.run(ClimateAdvisorDeleteReportView().post(_delete_request(hass, query, stamp)))

    def test_entry_id_deletes_only_from_that_zone(self, tmp_path):
        hass, _, _, stamp = self._zones(tmp_path)

        response = self._post(hass, {"entry_id": "entry_b"}, stamp)

        assert response.status == 200
        assert _texts(_zone(tmp_path, "entry_a")) == ["report A"]
        assert _texts(_zone(tmp_path, "entry_b")) == []

    def test_no_entry_id_with_several_zones_refuses_and_deletes_nothing(self, tmp_path):
        hass, _, _, stamp = self._zones(tmp_path)

        response = self._post(hass, {}, stamp)

        assert response.status == 400
        assert _texts(_zone(tmp_path, "entry_a")) == ["report A"]
        assert _texts(_zone(tmp_path, "entry_b")) == ["report B"]

    def test_no_entry_id_with_one_zone_still_works(self, tmp_path):
        zone = _zone(tmp_path, "entry_a")
        asyncio.run(zone.async_store_investigation_report({"text": "report A"}))
        hass = _hass(tmp_path)
        hass.data = {DOMAIN: {"entry_a": zone}}

        response = self._post(hass, {}, zone.get_investigation_report_history()[0]["timestamp"])

        assert response.status == 200
        assert _texts(_zone(tmp_path, "entry_a")) == []

"""One-time split of the old shared investigation reports file into per-zone files.

Every zone used to read and write `climate_advisor_investigation_reports.json`, so the file can hold
reports from any zone. `storage_paths.migrate_legacy_storage_file()` is not used here: it hands the whole
file to whichever zone starts first, which would file every other zone's reports under one arbitrary zone
(and lose them along with that zone).

A report records no zone field, but its stored `result.input_context` names the thermostat it was generated
from, and a zone is defined by its thermostat. So each report goes to the zone that has that thermostat. With
only one zone, everything goes to it. A report that can't be placed goes to no zone. The original file is
always kept whole as a backup, so nothing is lost either way.

One attempt per Home Assistant run: the first AI zone to start makes it, and every other zone waits for it before
loading (zones with AI off hold no reports in memory). So no zone has loaded when zone files are written, and none
can overwrite them from a stale in-memory list, which is what a retry during the run would allow. After a failed
attempt, zones run on their own files and the next start migrates and merges.

Remaining window: the zone files can't be replaced as one atomic step and there are no tombstones. If the process
stops or a replace fails while they are being replaced, or the shared file can't be removed afterwards, some zone
files are already live while the shared file remains. A report deleted from one of them before the next start is
merged back by that start's attempt.

Blocking I/O: run in the executor.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
import re
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from .const import INVESTIGATION_REPORTS_FILE
from .storage_paths import resolve_entry_scoped_path

_LOGGER = logging.getLogger(__name__)

LEGACY_BACKUP_STEM = "climate_advisor_investigation_reports_legacy_shared"

_ATTEMPTED_KEY = "climate_advisor_report_migration_attempted"
_ATTEMPT_LOCK = threading.Lock()

# The two places the investigator context names its thermostat: build_hvac_entity_context() in
# ai_skills_context.py and render_diagnostic_snapshot_text() in diagnostic_snapshot.py.
_HVAC_SECTION_RE = re.compile(r"^=== HVAC ENTITY ===\n  entity_id:\s+(\S+)[ \t]*$", re.MULTILINE)
_SNAPSHOT_FOOTER_RE = re.compile(r"^  config: climate_entity=(\S+)", re.MULTILINE)


def report_thermostat(entry: dict[str, Any]) -> str | None:
    """Return the thermostat a stored report was generated from, or None unless it names exactly one."""
    result = entry.get("result")
    context = result.get("input_context") if isinstance(result, dict) else None
    if not isinstance(context, str):
        return None
    found = set(_HVAC_SECTION_RE.findall(context)) | set(_SNAPSHOT_FOOTER_RE.findall(context))
    return found.pop() if len(found) == 1 else None


def _split_by_zone(reports: list[Any], zones: dict[str, str | None]) -> dict[str, list[dict[str, Any]]]:
    """Group the reports each zone can honestly claim. Entries the loader couldn't read are left out."""
    usable = [e for e in reports if isinstance(e, dict) and isinstance(e.get("timestamp"), str) and e["timestamp"]]
    if len(zones) == 1:
        return {next(iter(zones)): usable} if usable else {}

    claimants: dict[str, list[str]] = {}
    for entry_id, thermostat in zones.items():
        if thermostat:
            claimants.setdefault(thermostat, []).append(entry_id)

    split: dict[str, list[dict[str, Any]]] = {}
    for entry in usable:
        owners = claimants.get(report_thermostat(entry) or "", [])
        if len(owners) == 1:
            split.setdefault(owners[0], []).append(entry)
    return split


def _merged_zone_reports(path: Path, reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A zone file's content merged with its share of the old reports. Raises if the file can't be read."""
    merged: dict[str, dict[str, Any]] = {e["timestamp"]: e for e in reports}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            raise ValueError("not a list")
        merged.update(
            {e["timestamp"]: e for e in existing if isinstance(e, dict) and isinstance(e.get("timestamp"), str)}
        )
    return [merged[k] for k in sorted(merged)]


def _stage_zone_file(path: Path, reports: list[dict[str, Any]]) -> str:
    """Write a zone's content to a synced temp file beside its target and return the temp path."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f"{path.stem}_", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        if sys.platform != "win32":
            os.chmod(tmp_path, 0o600)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
    return tmp_path


def _write_backup(config_dir: Path, data: bytes) -> Path:
    """Keep the shared file's exact bytes under a name no later migration overwrites."""
    for n in itertools.count():
        path = config_dir / (f"{LEGACY_BACKUP_STEM}.json" if n == 0 else f"{LEGACY_BACKUP_STEM}.{n}.json")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if path.read_bytes() == data:  # an interrupted earlier run already got this far
                return path
            continue
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                path.unlink()
            raise
        return path
    raise AssertionError("unreachable")


def migrate_shared_reports_once(run_state: dict[str, Any], config_dir: Path, zones: dict[str, str | None]) -> None:
    """Make this run's one migration attempt, or wait for the one in progress. Returns once no attempt is running.

    `run_state` is `hass.data`, so the gate lasts as long as the Home Assistant run and survives zone reloads.
    """
    with _ATTEMPT_LOCK:
        if run_state.get(_ATTEMPTED_KEY):
            return
        run_state[_ATTEMPTED_KEY] = True
        migrate_shared_reports(config_dir, zones)


def migrate_shared_reports(config_dir: Path, zones: dict[str, str | None]) -> None:
    """Split the shared reports file between zones, then replace it with a backup. Harmless if there is none.

    `zones` maps every configured entry_id to its thermostat, including zones not set up yet. In order: parse every
    target zone file, write the backup, stage every zone's merged content in a synced temp file, replace the zone
    files, remove the shared file. A failure before the replacing leaves every zone file untouched, and the next
    start repeats the work (merging by timestamp, so repeating adds no duplicates).
    """
    legacy_path = config_dir / INVESTIGATION_REPORTS_FILE
    if not zones or not legacy_path.exists():
        return
    try:
        data = legacy_path.read_bytes()
    except OSError as err:
        _LOGGER.warning("Could not read shared investigation reports file, will retry next start: %s", err)
        return

    try:
        reports = json.loads(data)
        if not isinstance(reports, list):
            raise ValueError("not a list")
    except ValueError as err:
        _LOGGER.warning("Shared investigation reports file is unreadable (%s), setting it aside", err)
        reports = []

    split = _split_by_zone(reports, zones)
    staged: list[tuple[str, Path]] = []
    try:
        targets = {
            resolve_entry_scoped_path(config_dir, INVESTIGATION_REPORTS_FILE, entry_id): zone_reports
            for entry_id, zone_reports in split.items()
        }
        merged = {path: _merged_zone_reports(path, zone_reports) for path, zone_reports in targets.items()}
        backup_path = _write_backup(config_dir, data)
        for path, content in merged.items():
            staged.append((_stage_zone_file(path, content), path))
        while staged:
            os.replace(staged[0][0], staged[0][1])
            staged.pop(0)
    except (OSError, ValueError) as err:
        _LOGGER.error("Could not migrate shared investigation reports, will retry next start: %s", err)
        return
    finally:
        for tmp_path, _ in staged:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

    placed = sum(len(r) for r in split.values())
    if placed < len(reports):
        _LOGGER.warning(
            "%d of %d investigation reports could not be assigned to a zone; they are kept in %s",
            len(reports) - placed,
            len(reports),
            backup_path.name,
        )
    _LOGGER.info("Moved %d investigation reports to %d zones; backup: %s", placed, len(split), backup_path.name)
    try:
        legacy_path.unlink()
    except OSError as err:
        _LOGGER.warning("Could not remove shared investigation reports file after migrating: %s", err)

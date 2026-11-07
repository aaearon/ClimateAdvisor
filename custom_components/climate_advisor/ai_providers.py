"""Registers the built-in AI providers into ``ai_backend.PROVIDERS``.

``ai_backend`` imports no provider, so whoever needs ``get_provider`` (config flow, setup,
coordinator) first calls this. Importing the provider modules registers their specs; both are
cheap (no SDK import at module level). Idempotent.
"""

from __future__ import annotations

from typing import Any


def ensure_providers_registered() -> None:
    """Import the provider modules so their specs are in ``PROVIDERS`` (sync, idempotent)."""
    from . import chatgpt_backend, claude_api  # noqa: F401, PLC0415


async def async_ensure_providers_registered(hass: Any) -> None:
    """Same, with the first-time module imports kept off the event loop."""
    await hass.async_add_executor_job(ensure_providers_registered)

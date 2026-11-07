"""Provider seam for Climate Advisor's AI subsystem.

Provider-neutral contracts only: this module imports no provider SDK and no
provider module. Each provider registers a ``ProviderSpec`` in ``PROVIDERS``
(the built-in Anthropic spec registers itself when ``claude_api`` is imported).
``ClaudeAPIClient`` is the shared wrapper (circuit breaker, rate limits, status,
budget, history) and talks to a provider only through ``AIBackend``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

CREDENTIAL_API_KEY = "api_key"
CREDENTIAL_ACCOUNT = "account"

# Default ``requester_id`` of a client request: no HA user started it, so no user check applies.
USER_NOT_CHECKED: Any = object()

ERROR_CATEGORIES: tuple[str, ...] = ("auth", "quota", "rate_limit", "transient", "permanent")


class AIBackendError(Exception):
    """Neutral error raised by a backend; ``category`` drives shared handling.

    ``str(exc)`` is the provider's own message. ``timeout`` marks a timeout (for
    log/error wording only) and ``model_not_found`` marks a rejected model id
    (eligible for tier fallback when the provider supports it). ``no_retry`` marks a
    failure the wrapper must neither retry nor count toward the circuit breaker.
    """

    def __init__(
        self,
        message: str,
        category: str = "permanent",
        *,
        timeout: bool = False,
        model_not_found: bool = False,
        no_retry: bool = False,
    ) -> None:
        if category not in ERROR_CATEGORIES:
            raise ValueError(f"Unknown AIBackendError category: {category}")
        super().__init__(message)
        self.category = category
        self.timeout = timeout
        self.model_not_found = model_not_found
        self.no_retry = no_retry


class AIUserNotAllowedError(Exception):
    """The requester may not use the provider the zone is on now; nothing was sent."""

    def __init__(self, message: str = "Only the user who connected the AI account can run investigations") -> None:
        super().__init__(message)


class AIBackend(Protocol):
    """What the shared wrapper needs from a provider transport.

    Request methods take keyword-only, already-resolved options (see
    ``ProviderSpec.resolve_request_options``) and raise ``AIBackendError`` for
    provider failures. ``async_request`` returns a ``ClaudeResponse``.
    ``async_stream`` yields ``{"type": "text"|"thinking", "text": str}`` events
    and finally ``{"type": "stop", "stop_reason", "truncated_empty",
    "input_tokens", "output_tokens", "estimated_cost"}``.
    """

    client: Any  # underlying SDK/transport object, or None when inactive

    @property
    def inactive_reason(self) -> str | None:
        """Why requests cannot be made right now, or None when active."""
        ...

    async def async_request(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str,
    ) -> Any: ...

    def async_stream(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str,
    ) -> AsyncIterator[dict[str, Any]]: ...

    async def async_list_models(self) -> list[str]: ...

    async def async_test_connection(self, hass: Any | None = None) -> tuple[bool, str]: ...

    def update_config(self, config: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class ProviderSpec:
    """Static description of one AI provider and its capabilities."""

    id: str
    label: str
    credential_kind: str  # CREDENTIAL_API_KEY | CREDENTIAL_ACCOUNT
    # (config, client=None) -> AIBackend; may block (called in the executor).
    build_backend: Callable[..., AIBackend]
    usage_url: str | None
    tracks_cost: bool
    supports_tuning: bool  # temperature, max tokens, reasoning effort
    supports_tier_fallback: bool
    plan_restricted: bool  # manual-only and connected-owner-only
    # (config, overrides) -> resolved request options passed to the backend.
    resolve_request_options: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    # (config, hass) -> model ids; must not raise.
    list_models: Callable[[dict[str, Any], Any | None], Awaitable[list[str]]]
    # (config, hass) -> (ok, message)
    validate_credentials: Callable[[dict[str, Any], Any | None], Awaitable[tuple[bool, str]]]
    settings_fields: tuple[str, ...]  # config keys this provider's settings own
    # True only where the shared AI settings rows may keep their Claude/Anthropic wording.
    uses_claude_wording: bool = False
    # The config key holding this provider's API key (credential_kind == api_key only); the options
    # flow, settings rows, diagnostics redaction and coordinator gate read it from here. A backend
    # reads its own key from the config it is given.
    credential_setting: str = "ai_api_key"
    # The config key holding this provider's chosen model; the options flow, settings rows and
    # ``configured_model`` all read it from here. Must be one of ``settings_fields``.
    model_setting: str = "ai_model"
    # Model preselected when none is saved, and the list shown when the live list is unavailable.
    default_model: str = ""
    fallback_models: tuple[str, ...] = ()


PROVIDERS: dict[str, ProviderSpec] = {}


def get_provider(provider_id: str) -> ProviderSpec | None:
    """Return the registered spec, or None for an unknown id (never raises)."""
    try:
        return PROVIDERS.get(provider_id)
    except TypeError:  # unhashable id from a malformed config
        return None

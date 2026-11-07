"""ChatGPT sign-in inside the options flow, against a fake account manager with the real contract's surface."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.climate_advisor import ai_account, config_flow
from custom_components.climate_advisor.ai_backend import PROVIDERS, AIBackendError
from custom_components.climate_advisor.const import AI_ACCOUNT_HASS_KEY

from .test_config_flow import FULL_CONFIG_V12, _make_options_flow

PKG = Path(__file__).resolve().parent.parent / "custom_components" / "climate_advisor"
STRINGS = json.loads((PKG / "strings.json").read_text(encoding="utf-8"))
URL = "https://auth.openai.com/oauth/authorize?state=abc"
PASTED = "http://127.0.0.1:1455/auth/callback?code=c&state=abc"
CHATGPT = {**FULL_CONFIG_V12, "ai_provider": "chatgpt", "ai_chatgpt_model": "gpt-a"}
CHATGPT_NO_MODEL = {**FULL_CONFIG_V12, "ai_provider": "chatgpt"}
MODELS = ["gpt-a", "gpt-b"]

LoginError = ai_account.AccountLoginError
_REAL_READINESS = config_flow._check_account_readiness  # the env fixture replaces the module attribute


def _err(reason):
    return LoginError(reason)


class FakeAccount:
    def __init__(self, state="disconnected", email="me@example.test"):
        self.state, self.email = state, email
        self.complete_errors: list[Exception] = []
        self.begin_error: Exception | None = None
        self.begins = 0
        self.has_pending_login = False
        self.pending_login_id: str | None = None
        self.async_load = AsyncMock()
        self.async_cancel_login = MagicMock()  # sync @callback on the real manager
        self.async_disconnect = AsyncMock(return_value="confirmed")
        self.completed_with: list[tuple[str, str]] = []

    def status(self):
        return {"state": self.state, "email": self.email, "login_pending": self.has_pending_login}

    async def async_begin_login(self):
        if self.begin_error:
            raise self.begin_error
        self.begins += 1
        self.has_pending_login = True
        self.pending_login_id = f"id{self.begins}"
        return f"{URL}{self.begins}"

    async def async_complete_login(self, pasted_url, owner_user_id):
        self.completed_with.append((pasted_url, owner_user_id))
        if self.complete_errors:
            raise self.complete_errors.pop(0)
        self.state, self.has_pending_login = "connected", False
        return self.status()


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(config_flow, "_current_user_id", lambda: "admin-1")
    monkeypatch.setattr(config_flow, "_check_account_readiness", AsyncMock(return_value="ready"))
    from custom_components.climate_advisor.ai_providers import ensure_providers_registered

    ensure_providers_registered()
    models = AsyncMock(return_value=list(MODELS))
    monkeypatch.setitem(PROVIDERS, "chatgpt", dataclasses.replace(PROVIDERS["chatgpt"], list_models=models))

    def make(data=CHATGPT, manager=None):
        flow, captured = _make_options_flow(data)
        flow._login_open = False  # the helper skips __init__
        manager = manager or FakeAccount()
        flow.hass.data = {AI_ACCOUNT_HASS_KEY: manager}
        flow.hass.async_create_task = MagicMock(side_effect=lambda coro: asyncio.ensure_future(coro))
        return flow, captured, manager

    make.models = models
    return make


def _default(result, key):
    """Default of one field's marker (the selector is a stub in unit tests)."""
    marker = next(k for k in result["data_schema"].schema if str(k) == key)
    return marker.default() if callable(marker.default) else marker.default


def run(coro):
    return asyncio.run(coro)


def _menu_options(result):
    return list(result["menu_options"])


class TestMenu:
    def test_save_on_account_provider_routes_to_account_menu(self, env):
        flow, captured, _ = env()
        run(flow.async_step_ai_provider({"ai_provider": "chatgpt"}))
        result = run(flow.async_step_ai_settings({"ai_enabled": True}))
        assert result["step_id"] == "ai_account"
        assert _menu_options(result) == ["ai_account_signin", "init"]
        assert captured["data"]["ai_provider"] == "chatgpt"

    def test_unproved_save_does_not_say_nothing_is_connected(self, env):
        flow, _, _ = env(manager=FakeAccount(state="save_uncertain"))
        result = asyncio.run(flow.async_step_ai_account())
        assert result["menu_options"] == ["ai_account_signin", "init"]
        summary = result["description_placeholders"]["account_summary"]
        assert "could not be confirmed" in summary
        assert "No ChatGPT account" not in summary

    def test_reauth_required_offers_sign_in_again(self, env):
        flow, _, _ = env(manager=FakeAccount(state="reauth_required"))
        result = run(flow.async_step_ai_account())
        assert _menu_options(result) == ["ai_account_signin", "init"]
        assert "expired" in result["description_placeholders"]["account_summary"]

    def test_connected_shows_account_and_disconnect(self, env):
        flow, _, _ = env(manager=FakeAccount(state="connected"))
        result = run(flow.async_step_ai_account())
        assert _menu_options(result) == ["ai_account_disconnect", "init"]
        assert "me@example.test" in result["description_placeholders"]["account_summary"]

    def test_claude_provider_flow_unchanged(self, env):
        flow, captured, manager = env(data={**FULL_CONFIG_V12, "ai_api_key": "sk-keep"})
        run(flow.async_step_ai_provider({"ai_provider": "anthropic"}))
        result = run(flow.async_step_ai_settings({"ai_enabled": True, "ai_temperature": 0.3, "ai_max_tokens": 1024}))
        assert result["step_id"] == "init"
        assert captured["data"]["ai_api_key"] == "sk-keep"
        manager.async_load.assert_not_called()


class TestSignIn:
    def test_menu_goes_straight_to_the_link_with_no_gate(self, env):
        flow, _, manager = env()
        menu = run(flow.async_step_ai_account())
        assert "ai_account_signin" in menu["menu_options"]
        signin = run(flow.async_step_ai_account_signin())
        assert signin["step_id"] == "ai_account_signin" and signin["errors"] == {}
        assert signin["description_placeholders"]["login_url"] == f"{URL}1"
        assert manager.begins == 1
        assert not hasattr(flow, "async_step_ai_account_privacy")

    def test_full_flow_success_without_commit_or_reload(self, env):
        flow, captured, manager = env()
        signin = run(flow.async_step_ai_account_signin())
        assert signin["step_id"] == "ai_account_signin"
        assert signin["description_placeholders"]["login_url"] == f"{URL}1"
        assert {str(k) for k in signin["data_schema"].schema} == {"ai_account_redirect_url"}

        done = run(flow.async_step_ai_account_signin({"ai_account_redirect_url": f"  {PASTED}  "}))
        assert done["step_id"] == "ai_account_done"
        assert manager.completed_with == [(PASTED, "admin-1")]
        assert flow._readiness == "ready"
        assert "ready" in done["description_placeholders"]["readiness"]

        flow.hass.config_entries.async_update_entry.assert_not_called()
        flow.hass.config_entries.async_reload.assert_not_called()
        assert "data" not in captured

        back = run(flow.async_step_ai_account_done({}))
        assert back["type"] == "menu" and back["step_id"] == "init"

    def test_temporary_readiness_error_does_not_fail_sign_in(self, env, monkeypatch):
        monkeypatch.setattr(config_flow, "_check_account_readiness", AsyncMock(return_value="not_ready"))
        flow, _, manager = env()
        run(flow.async_step_ai_account_signin())
        done = run(flow.async_step_ai_account_signin({"ai_account_redirect_url": PASTED}))
        assert done["step_id"] == "ai_account_done"
        assert manager.state == "connected"
        assert "not ready" in done["description_placeholders"]["readiness"]

    def test_mistyped_paste_reshows_form_with_same_link(self, env):
        flow, _, manager = env()
        run(flow.async_step_ai_account_signin())
        manager.complete_errors = [_err("bad_url")]
        form = run(flow.async_step_ai_account_signin({"ai_account_redirect_url": "oops"}))
        assert form["step_id"] == "ai_account_signin"
        assert form["errors"] == {"ai_account_redirect_url": "bad_url"}
        assert form["description_placeholders"]["login_url"] == f"{URL}1"
        assert manager.begins == 1
        done = run(flow.async_step_ai_account_signin({"ai_account_redirect_url": PASTED}))
        assert done["step_id"] == "ai_account_done"

    @pytest.mark.parametrize(
        "reason",
        [
            "login_expired",
            "no_pending_login",
            "consent_denied",
            "exchange_failed",
            "login_checks_failed",
            "missing_scope",
            "bad_identity",
            "save_failed",
        ],
    )
    def test_spent_login_gets_a_fresh_link(self, env, reason):
        flow, _, manager = env()
        run(flow.async_step_ai_account_signin())
        manager.complete_errors = [_err(reason)]
        form = run(flow.async_step_ai_account_signin({"ai_account_redirect_url": PASTED}))
        assert form["errors"] == {"ai_account_redirect_url": reason}
        assert form["description_placeholders"]["login_url"] == f"{URL}2"

    @pytest.mark.parametrize("reason", ["bad_url", "state_mismatch"])
    def test_attempt_kept_after_a_wrong_paste_keeps_the_same_link(self, env, reason):
        flow, _, manager = env()
        run(flow.async_step_ai_account_signin())
        manager.complete_errors = [_err(reason)]
        form = run(flow.async_step_ai_account_signin({"ai_account_redirect_url": PASTED}))
        assert form["description_placeholders"]["login_url"] == f"{URL}1"
        assert manager.begins == 1

    @pytest.mark.parametrize("reason", config_flow.ACCOUNT_LOGIN_ERROR_REASONS)
    def test_every_reason_maps_to_an_existing_string(self, env, reason):
        flow, _, manager = env()
        run(flow.async_step_ai_account_signin())
        manager.complete_errors = [_err(reason)]
        form = run(flow.async_step_ai_account_signin({"ai_account_redirect_url": PASTED}))
        key = form["errors"]["ai_account_redirect_url"]
        assert key == reason
        assert STRINGS["options"]["error"][key].strip()

    def test_unknown_reason_and_backend_error_show_generic_text(self, env):
        flow, _, manager = env()
        run(flow.async_step_ai_account_signin())
        novel = _err("save_failed")
        novel.reason = "brand_new"  # a reason the flow has no string for
        manager.complete_errors = [novel, AIBackendError("disk", "transient")]
        for _ in range(2):
            form = run(flow.async_step_ai_account_signin({"ai_account_redirect_url": PASTED}))
            assert form["errors"] == {"ai_account_redirect_url": "login_failed"}
        assert "login_failed" in STRINGS["options"]["error"]

    def test_begin_failure_shows_the_error_and_submitting_tries_again(self, env):
        flow, _, manager = env()
        manager.begin_error = _err("already_connected")
        form = run(flow.async_step_ai_account_signin())
        assert form["step_id"] == "ai_account_signin"
        assert form["errors"] == {"base": "already_connected"}
        assert form["description_placeholders"]["login_url"] == ""
        manager.begin_error = None
        retried = run(flow.async_step_ai_account_signin({"ai_account_redirect_url": ""}))
        assert retried["errors"] == {} and retried["description_placeholders"]["login_url"] == f"{URL}1"
        assert manager.completed_with == []

    def test_no_user_id_refuses_with_owner_required_and_never_guesses(self, env, monkeypatch):
        monkeypatch.setattr(config_flow, "_current_user_id", lambda: None)
        flow, _, manager = env()
        form = run(flow.async_step_ai_account_signin())
        assert form["errors"] == {"base": "owner_required"}
        assert manager.begins == 0
        manager.has_pending_login = True
        flow._login_url = URL
        flow._login_open = True
        form = run(flow.async_step_ai_account_signin({"ai_account_redirect_url": PASTED}))
        assert form["errors"] == {"ai_account_redirect_url": "owner_required"}
        assert manager.completed_with == []

    def test_abandoned_flow_cancels_pending_login(self, env):
        async def go():
            flow, _, manager = env()
            await flow.async_step_ai_account_signin()
            flow.async_remove()
            await asyncio.sleep(0)
            return manager

        manager = run(go())
        manager.async_cancel_login.assert_called_once_with("id1")

    def test_finished_flow_does_not_cancel(self, env):
        async def go():
            flow, _, manager = env()
            await flow.async_step_ai_account_signin()
            await flow.async_step_ai_account_signin({"ai_account_redirect_url": PASTED})
            flow.async_remove()
            await asyncio.sleep(0)
            return manager

        run(go()).async_cancel_login.assert_not_called()


class TestReadinessProbe:
    def test_probes_the_registered_account_provider_not_a_hardcoded_id(self, monkeypatch):
        backend = MagicMock(async_fetch_models=AsyncMock())
        build = MagicMock(return_value=backend)
        from custom_components.climate_advisor import ai_providers

        monkeypatch.setattr(ai_providers, "async_ensure_providers_registered", AsyncMock())
        other = dataclasses.replace(
            PROVIDERS["anthropic"], id="other-account", credential_kind="account", build_backend=build
        )
        monkeypatch.setattr(config_flow, "_check_account_readiness", _REAL_READINESS)
        monkeypatch.delitem(PROVIDERS, "chatgpt")
        monkeypatch.setitem(PROVIDERS, "other-account", other)
        manager = object()
        assert run(_REAL_READINESS(MagicMock(), manager)) == "ready"
        build.assert_called_once_with({}, manager)
        backend.async_fetch_models.assert_awaited_once()


class TestModelAfterSignIn:
    def _sign_in(self, flow):
        run(flow.async_step_ai_account_signin())
        return run(flow.async_step_ai_account_signin({"ai_account_redirect_url": PASTED}))

    def test_no_model_goes_to_a_model_choice_without_committing_first(self, env):
        flow, captured, manager = env(data=CHATGPT_NO_MODEL)
        step = self._sign_in(flow)
        assert step["step_id"] == "ai_account_model"
        assert manager.state == "connected"
        assert {str(k) for k in step["data_schema"].schema} == {"ai_chatgpt_model"}
        assert _default(step, "ai_chatgpt_model") == "gpt-a"
        flow.hass.config_entries.async_update_entry.assert_not_called()
        flow.hass.config_entries.async_reload.assert_not_called()

    def test_choice_is_saved_after_sign_in_then_done(self, env):
        flow, captured, _ = env(data=CHATGPT_NO_MODEL)
        self._sign_in(flow)
        done = run(flow.async_step_ai_account_model({"ai_chatgpt_model": "gpt-b"}))
        assert done["step_id"] == "ai_account_done"
        assert captured["data"]["ai_chatgpt_model"] == "gpt-b"
        assert captured["data"]["ai_provider"] == "chatgpt"
        flow.hass.config_entries.async_reload.assert_not_called()
        assert "AI Settings" not in done["description_placeholders"]["readiness"]

    def test_model_list_failure_keeps_sign_in_and_points_to_ai_settings(self, env):
        flow, captured, manager = env(data=CHATGPT_NO_MODEL)
        env.models.side_effect = RuntimeError("boom")
        done = self._sign_in(flow)
        assert done["step_id"] == "ai_account_done"
        assert manager.state == "connected"
        assert "pick a model in AI Settings" in done["description_placeholders"]["readiness"]
        assert "data" not in captured
        manager.async_disconnect.assert_not_called()

    def test_no_model_never_reports_ready(self, env):
        flow, _, _ = env(data=CHATGPT_NO_MODEL)
        env.models.side_effect = RuntimeError("boom")
        done = self._sign_in(flow)
        assert "are ready" not in done["description_placeholders"]["readiness"]

    def test_empty_model_list_is_a_temporary_failure_too(self, env):
        flow, _, manager = env(data=CHATGPT_NO_MODEL)
        env.models.return_value = []
        done = self._sign_in(flow)
        assert done["step_id"] == "ai_account_done"
        assert "pick a model in AI Settings" in done["description_placeholders"]["readiness"]

    def test_zone_with_a_model_stays_one_step(self, env):
        flow, _, _ = env(data=CHATGPT)
        assert self._sign_in(flow)["step_id"] == "ai_account_done"
        env.models.assert_not_called()

    def test_second_zone_form_has_a_model_that_is_always_saved(self, env):
        flow, captured, _ = env(data={**FULL_CONFIG_V12}, manager=FakeAccount(state="connected"))
        run(flow.async_step_ai_provider({"ai_provider": "chatgpt"}))
        form = run(flow.async_step_ai_settings())
        assert "ai_chatgpt_model" in {str(k) for k in form["data_schema"].schema}
        marker = next(k for k in form["data_schema"].schema if str(k) == "ai_chatgpt_model")
        assert isinstance(marker, config_flow.vol.Required)
        assert _default(form, "ai_chatgpt_model") == "gpt-a"
        run(flow.async_step_ai_settings({"ai_enabled": True, "ai_chatgpt_model": _default(form, "ai_chatgpt_model")}))
        assert captured["data"]["ai_chatgpt_model"] == "gpt-a"


class TestOverlappingFlows:
    def test_replaced_flow_cancels_only_its_own_attempt(self, env):
        manager = FakeAccount()
        first, _, _ = env(manager=manager)
        second, _, _ = env(manager=manager)
        run(first.async_step_ai_account_signin())
        run(second.async_step_ai_account_signin())
        assert (first._login_id, second._login_id) == ("id1", "id2")
        first.async_remove()
        manager.async_cancel_login.assert_called_once_with("id1")
        second.async_remove()
        assert manager.async_cancel_login.call_args_list[-1].args == ("id2",)

    def test_stale_paste_gets_state_mismatch_and_does_not_restart_the_login(self, env):
        manager = FakeAccount()
        first, _, _ = env(manager=manager)
        run(first.async_step_ai_account_signin())
        manager.complete_errors = [_err("state_mismatch")]
        form = run(first.async_step_ai_account_signin({"ai_account_redirect_url": PASTED}))
        assert form["errors"] == {"ai_account_redirect_url": "state_mismatch"}
        assert manager.begins == 1
        assert first._login_id == "id1"


class TestDisconnect:
    def test_disconnect_confirmed(self, env):
        flow, _, manager = env(manager=FakeAccount(state="connected"))
        form = run(flow.async_step_ai_account_disconnect())
        assert form["step_id"] == "ai_account_disconnect"
        done = run(flow.async_step_ai_account_disconnect({"ai_account_consent": True}))
        manager.async_disconnect.assert_awaited_once()
        assert done["step_id"] == "ai_account_disconnected"
        assert done["description_placeholders"]["revocation"] == "Access was revoked."

    def test_unconfirmed_revocation_is_reported(self, env):
        flow, _, manager = env(manager=FakeAccount(state="connected"))
        manager.async_disconnect = AsyncMock(return_value="unconfirmed")
        done = run(flow.async_step_ai_account_disconnect({"ai_account_consent": True}))
        assert "ChatGPT Settings" in done["description_placeholders"]["revocation"]

    def test_declining_goes_back_without_disconnecting(self, env):
        flow, _, manager = env(manager=FakeAccount(state="connected"))
        back = run(flow.async_step_ai_account_disconnect({"ai_account_consent": False}))
        assert back["step_id"] == "ai_account"
        manager.async_disconnect.assert_not_called()


class TestStrings:
    steps = STRINGS["options"]["step"]

    def test_new_steps_have_strings_and_field_labels(self):
        for step in (
            "ai_account_signin",
            "ai_account_model",
            "ai_account_done",
            "ai_account_disconnect",
        ):
            assert self.steps[step]["title"] and self.steps[step]["description"]
        assert set(self.steps["ai_account"]["menu_options"]) == {"ai_account_signin", "ai_account_disconnect", "init"}
        for step, key in (
            ("ai_account_signin", "ai_account_redirect_url"),
            ("ai_account_model", "ai_chatgpt_model"),
            ("ai_account_disconnect", "ai_account_consent"),
        ):
            assert self.steps[step]["data"][key] and self.steps[step]["data_description"][key]

    def test_provider_step_strings_and_menu_label(self):
        step = self.steps["ai_provider"]
        assert step["title"] and step["description"]
        assert step["data"]["ai_provider"] and step["data_description"]["ai_provider"]
        assert len(step["data"]["ai_provider"].split()) <= 5
        assert self.steps["init"]["menu_options"]["ai_provider"] == "AI Settings"
        assert "ai_settings" not in self.steps["init"]["menu_options"]
        assert "ai_provider" not in self.steps["ai_settings"]["data"]
        assert "ai_provider" not in self.steps["ai_settings"]["data_description"]

    def test_required_texts(self):
        assert "ai_account_privacy" not in self.steps
        assert "consent_required" not in STRINGS["options"]["error"]
        signin = self.steps["ai_account_signin"]["description"]
        assert "Requests send this home's climate data to OpenAI under the connected ChatGPT plan." in signin
        assert "{login_url}" in signin and "will fail to load" in signin and "expected" in signin
        assert "every zone" in signin
        done = self.steps["ai_account_done"]
        assert done["title"] == "Using ChatGPT plan" and "{readiness}" in done["description"]

    def test_repair_text_points_to_the_options_flow_not_a_bundle(self):
        text = STRINGS["issues"]["ai_account_reauth_required"]["description"]
        assert "AI Settings" in text and "Sign in with ChatGPT" in text
        for gone in ("bundle", "scp", "tool", "panel"):
            assert gone not in text

    def test_no_bundle_text_anywhere_in_strings(self):
        raw = (PKG / "strings.json").read_text(encoding="utf-8").lower()
        for gone in ("login bundle", "chatgpt_login", "scp ", "login tool", "sign-in tool"):
            assert gone not in raw, gone

    def test_placeholders_used_are_supplied(self):
        supplied = {
            "ai_account": {"account_summary"},
            "ai_account_signin": {"login_url", "redirect_host"},
            "ai_account_done": {"readiness"},
            "ai_account_model": {"provider_label"},
            "ai_account_disconnected": {"revocation"},
        }
        for step, names in supplied.items():
            used = set(re.findall(r"\{(\w+)\}", self.steps[step]["description"]))
            assert used == names, step

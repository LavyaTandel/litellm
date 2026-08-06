"""Tests for PTU config on the model deployment (v1 model-settings design)."""

import json
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LiteLLM_ProxyModelTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.model_management_endpoints import (
    _raise_if_ptu_cost_attribution_disabled,
    _validate_ptu_model_info,
    add_new_model,
    update_db_model,
)
from litellm.proxy.spend_tracking.ptu_feature_flag import PTU_COST_ATTRIBUTION_ENV_VAR
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo, updateDeployment


def test_model_info_accepts_valid_ptu_fields():
    info = ModelInfo(id="x", team_id="t", ptu_count=5, cost_per_ptu_per_hour=2.0)
    assert info.ptu_count == 5
    assert info.cost_per_ptu_per_hour == 2.0


def test_model_info_rejects_non_positive_count():
    with pytest.raises(ValueError):
        ModelInfo(id="x", team_id="t", ptu_count=0, cost_per_ptu_per_hour=2.0)


def test_model_info_rejects_negative_rate():
    with pytest.raises(ValueError):
        ModelInfo(id="x", team_id="t", ptu_count=5, cost_per_ptu_per_hour=-1.0)


def test_model_info_allows_partial_delta_for_patch():
    # A PATCH delta may carry only one field; bounds-only validation must not reject it.
    info = ModelInfo(id="x", ptu_count=5)
    assert info.ptu_count == 5
    assert info.cost_per_ptu_per_hour is None


def test_validate_helper_no_ptu_is_noop():
    _validate_ptu_model_info({"team_id": "t"})


def test_validate_helper_requires_both_fields():
    with pytest.raises(HTTPException) as exc:
        _validate_ptu_model_info({"team_id": "t", "ptu_count": 5})
    assert exc.value.status_code == 400
    assert "set together" in exc.value.detail


def test_validate_helper_requires_team_id():
    with pytest.raises(HTTPException) as exc:
        _validate_ptu_model_info({"ptu_count": 5, "cost_per_ptu_per_hour": 2.0})
    assert exc.value.status_code == 400
    assert "team_id" in exc.value.detail


def test_validate_helper_passes_full_config():
    _validate_ptu_model_info({"team_id": "t", "ptu_count": 5, "cost_per_ptu_per_hour": 2.0})


def test_model_info_rejects_effective_to_before_from():
    import datetime

    with pytest.raises(ValueError):
        ModelInfo(
            id="x",
            team_id="t",
            ptu_count=5,
            cost_per_ptu_per_hour=2.0,
            ptu_effective_from=datetime.datetime(2026, 7, 30, tzinfo=datetime.timezone.utc),
            ptu_effective_to=datetime.datetime(2026, 7, 29, tzinfo=datetime.timezone.utc),
        )


def test_model_info_accepts_valid_effective_window():
    import datetime

    info = ModelInfo(
        id="x",
        team_id="t",
        ptu_count=5,
        cost_per_ptu_per_hour=2.0,
        ptu_effective_from=datetime.datetime(2026, 7, 30, tzinfo=datetime.timezone.utc),
        ptu_effective_to=datetime.datetime(2026, 8, 30, tzinfo=datetime.timezone.utc),
    )
    assert info.ptu_effective_from is not None


def test_model_info_compares_mixed_naive_and_aware_timestamps():
    import datetime

    info = ModelInfo(
        id="x",
        team_id="t",
        ptu_count=5,
        cost_per_ptu_per_hour=2.0,
        ptu_effective_from=datetime.datetime(2026, 7, 30, 23, 0),
        ptu_effective_to=datetime.datetime(2026, 7, 31, 0, 0, tzinfo=datetime.timezone.utc),
    )
    assert info.ptu_effective_to is not None

    with pytest.raises(ValueError):
        ModelInfo(
            id="x",
            team_id="t",
            ptu_count=5,
            cost_per_ptu_per_hour=2.0,
            ptu_effective_from=datetime.datetime(2026, 7, 31, 2, 0),
            ptu_effective_to=datetime.datetime(2026, 7, 31, 0, 0, tzinfo=datetime.timezone.utc),
        )


def test_validate_helper_rejects_effective_to_before_from():
    with pytest.raises(HTTPException) as exc:
        _validate_ptu_model_info(
            {
                "team_id": "t",
                "ptu_count": 5,
                "cost_per_ptu_per_hour": 2.0,
                "ptu_effective_from": "2026-07-30T00:00:00Z",
                "ptu_effective_to": "2026-07-29T00:00:00Z",
            }
        )
    assert exc.value.status_code == 400
    assert "ptu_effective_to" in exc.value.detail


def test_validate_helper_accepts_valid_window_on_merged_info():
    _validate_ptu_model_info(
        {
            "team_id": "t",
            "ptu_count": 5,
            "cost_per_ptu_per_hour": 2.0,
            "ptu_effective_from": "2026-07-30T00:00:00Z",
            "ptu_effective_to": "2026-08-30T00:00:00Z",
        }
    )


def test_validate_helper_rejects_inverted_window_without_count_or_rate():
    """A patch that touches only one end of the window merges to a model_info with no count
    or rate. Returning early on that shape let an inverted window reach the row, and the next
    load then failed to parse it and dropped the deployment out of the router."""
    with pytest.raises(HTTPException) as exc:
        _validate_ptu_model_info(
            {
                "team_id": "t",
                "ptu_effective_from": "2026-08-02T00:00:00Z",
                "ptu_effective_to": "2026-08-01T00:00:00Z",
            }
        )
    assert exc.value.status_code == 400
    assert "ptu_effective_to" in exc.value.detail


def test_validate_helper_rejects_equal_window_bounds_without_count_or_rate():
    with pytest.raises(HTTPException) as exc:
        _validate_ptu_model_info(
            {
                "ptu_effective_from": "2026-08-01T00:00:00Z",
                "ptu_effective_to": "2026-08-01T00:00:00Z",
            }
        )
    assert exc.value.status_code == 400


def test_validate_helper_accepts_ordered_window_without_count_or_rate():
    """Window-only edits stay legal; only the ordering is enforced, and no team_id is
    demanded while the deployment carries no priced PTU config."""
    _validate_ptu_model_info(
        {
            "ptu_effective_from": "2026-08-01T00:00:00Z",
            "ptu_effective_to": "2026-08-02T00:00:00Z",
        }
    )


def test_validate_helper_accepts_a_single_open_ended_bound():
    _validate_ptu_model_info({"ptu_effective_from": "2026-08-01T00:00:00Z"})
    _validate_ptu_model_info({"ptu_effective_to": "2026-08-02T00:00:00Z"})


class TestPtuCostAttributionGate:
    """PTU config is only writable once an operator sets LITELLM_ENABLE_PTU_COST_ATTRIBUTION.

    The fields are rejected rather than dropped: a silent accept-and-drop would let a
    caller believe a flat cost was configured while the rollup that prices it is not
    even scheduled.
    """

    @pytest.fixture(autouse=True)
    def _flag_off(self, monkeypatch):
        monkeypatch.delenv(PTU_COST_ATTRIBUTION_ENV_VAR, raising=False)

    @pytest.fixture
    def flag_on(self, monkeypatch):
        monkeypatch.setenv(PTU_COST_ATTRIBUTION_ENV_VAR, "true")

    @pytest.mark.parametrize(
        "model_info",
        [
            {"team_id": "t", "ptu_count": 5, "cost_per_ptu_per_hour": 2.0},
            {"ptu_count": 5},
            {"cost_per_ptu_per_hour": 2.0},
            {"ptu_effective_from": "2026-08-01T00:00:00Z"},
            {"ptu_effective_to": "2026-08-02T00:00:00Z"},
        ],
    )
    def test_rejects_any_ptu_field_while_disabled(self, model_info):
        with pytest.raises(HTTPException) as exc:
            _raise_if_ptu_cost_attribution_disabled(model_info)
        assert exc.value.status_code == 400
        assert PTU_COST_ATTRIBUTION_ENV_VAR in exc.value.detail

    def test_names_every_offending_field(self):
        with pytest.raises(HTTPException) as exc:
            _raise_if_ptu_cost_attribution_disabled({"ptu_count": 5, "cost_per_ptu_per_hour": 2.0})
        assert "ptu_count" in exc.value.detail
        assert "cost_per_ptu_per_hour" in exc.value.detail

    def test_allows_a_request_without_ptu_fields_while_disabled(self):
        _raise_if_ptu_cost_attribution_disabled({"team_id": "t", "access_groups": ["a"]})

    def test_allows_every_ptu_field_once_enabled(self, flag_on):
        _raise_if_ptu_cost_attribution_disabled(
            {
                "team_id": "t",
                "ptu_count": 5,
                "cost_per_ptu_per_hour": 2.0,
                "ptu_effective_from": "2026-08-01T00:00:00Z",
                "ptu_effective_to": "2026-08-02T00:00:00Z",
            }
        )


def _deployment_without_ptu() -> Deployment:
    return Deployment(
        model_name="gpt-4o",
        litellm_params=LiteLLM_Params(model="openai/gpt-4o"),
        model_info=ModelInfo(id="dep-0", team_id="t"),
    )


def _deployment_with_stored_ptu() -> Deployment:
    return Deployment(
        model_name="gpt-4o",
        litellm_params=LiteLLM_Params(model="openai/gpt-4o"),
        model_info=ModelInfo(id="dep-0", team_id="t", ptu_count=15, cost_per_ptu_per_hour=2.0),
    )


class TestUpdateDbModelPtuGate:
    @pytest.fixture(autouse=True)
    def _flag_off(self, monkeypatch):
        monkeypatch.delenv(PTU_COST_ATTRIBUTION_ENV_VAR, raising=False)

    def test_patch_carrying_ptu_config_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            update_db_model(
                db_model=_deployment_without_ptu(),
                updated_patch=updateDeployment(model_info=ModelInfo(id="dep-0", team_id="t", ptu_count=15)),
            )
        assert exc.value.status_code == 400

    def test_patch_that_touches_nothing_ptu_still_succeeds(self):
        result = update_db_model(
            db_model=_deployment_without_ptu(),
            updated_patch=updateDeployment(model_info=ModelInfo(id="dep-0", access_groups=["a"])),
        )
        assert json.loads(result["model_info"])["access_groups"] == ["a"]

    def test_unrelated_patch_of_a_model_that_stores_ptu_config_is_not_blocked(self):
        """A deployment configured during an earlier opt-in stays editable: the gate reads the
        incoming patch, not the merged deployment, so the stored config is left in place."""
        result = update_db_model(
            db_model=_deployment_with_stored_ptu(),
            updated_patch=updateDeployment(model_name="gpt-4o-renamed"),
        )
        assert result["model_name"] == "gpt-4o-renamed"

    def test_explicit_nulls_do_not_erase_stored_ptu_config_while_disabled(self):
        """A client round-tripping a model_info blob sends the PTU keys as nulls. While the
        feature is disabled those nulls must not reach the clear loop: disabling pauses PTU,
        it does not silently discard a billing configuration the operator set up earlier."""
        result = update_db_model(
            db_model=_deployment_with_stored_ptu(),
            updated_patch=updateDeployment(
                model_info=ModelInfo(id="dep-0", ptu_count=None, cost_per_ptu_per_hour=None)
            ),
        )
        stored = json.loads(result["model_info"])
        assert stored["ptu_count"] == 15
        assert stored["cost_per_ptu_per_hour"] == 2.0

    def test_explicit_nulls_still_clear_once_enabled(self, monkeypatch):
        """Clearing remains available to an operator who opted in, which is how PTU config is
        removed from a deployment."""
        monkeypatch.setenv(PTU_COST_ATTRIBUTION_ENV_VAR, "true")
        result = update_db_model(
            db_model=_deployment_with_stored_ptu(),
            updated_patch=updateDeployment(
                model_info=ModelInfo(id="dep-0", ptu_count=None, cost_per_ptu_per_hour=None)
            ),
        )
        stored = json.loads(result["model_info"])
        assert "ptu_count" not in stored
        assert "cost_per_ptu_per_hour" not in stored

    def test_patch_carrying_ptu_config_is_accepted_once_enabled(self, monkeypatch):
        monkeypatch.setenv(PTU_COST_ATTRIBUTION_ENV_VAR, "true")
        result = update_db_model(
            db_model=_deployment_without_ptu(),
            updated_patch=updateDeployment(
                model_info=ModelInfo(id="dep-0", team_id="t", ptu_count=15, cost_per_ptu_per_hour=2.0)
            ),
        )
        stored = json.loads(result["model_info"])
        assert stored["ptu_count"] == 15
        assert stored["cost_per_ptu_per_hour"] == 2.0


class TestAddNewModelPtuGate:
    @pytest.fixture(autouse=True)
    def _flag_off(self, monkeypatch):
        monkeypatch.delenv(PTU_COST_ATTRIBUTION_ENV_VAR, raising=False)

    @staticmethod
    def _patched_proxy(model_id: str):
        """Patch everything /model/new touches except the PTU gate, and hand back the DB writers."""
        db_row = LiteLLM_ProxyModelTable(
            model_id=model_id,
            model_name="ptu-model",
            litellm_params={"model": "openai/gpt-4.1-nano"},
            model_info={"id": model_id},
            created_by="test-admin",
            updated_by="test-admin",
        )
        add_model_to_db = AsyncMock(return_value=db_row)
        add_team_model_to_db = AsyncMock(return_value=db_row)

        mock_proxy_config = MagicMock()
        mock_proxy_config.add_deployment = AsyncMock(return_value=None)

        mock_router = MagicMock()
        mock_router.get_model_ids.return_value = [model_id]

        proxy_server = "litellm.proxy.proxy_server"
        endpoints = "litellm.proxy.management_endpoints.model_management_endpoints"
        return (add_model_to_db, add_team_model_to_db), [
            patch(f"{proxy_server}.prisma_client", MagicMock()),
            patch(f"{proxy_server}.store_model_in_db", True),
            patch(f"{proxy_server}.proxy_config", mock_proxy_config),
            patch(f"{proxy_server}.proxy_logging_obj", MagicMock()),
            patch(f"{proxy_server}.general_settings", {}),
            patch(f"{proxy_server}.premium_user", True),
            patch(f"{proxy_server}.llm_router", mock_router),
            patch(
                f"{endpoints}.ModelManagementAuthChecks.can_user_make_model_call",
                AsyncMock(return_value=True),
            ),
            patch(f"{endpoints}._add_model_to_db", add_model_to_db),
            patch(f"{endpoints}._add_team_model_to_db", add_team_model_to_db),
        ]

    @staticmethod
    def _ptu_deployment(model_id: str) -> Deployment:
        return Deployment(
            model_name="ptu-model",
            litellm_params=LiteLLM_Params(model="openai/gpt-4.1-nano", api_key="fake-key"),
            model_info=ModelInfo(id=model_id, team_id="team-1", ptu_count=15, cost_per_ptu_per_hour=2.0),
        )

    @pytest.mark.asyncio
    async def test_model_new_rejects_ptu_config_while_disabled(self):
        (add_model_to_db, add_team_model_to_db), patches = self._patched_proxy("ptu-gate-model")
        admin = UserAPIKeyAuth(user_id="test-admin", user_role=LitellmUserRoles.PROXY_ADMIN)

        with ExitStack() as stack:
            for active_patch in patches:
                stack.enter_context(active_patch)
            with pytest.raises(Exception) as exc:
                await add_new_model(model_params=self._ptu_deployment("ptu-gate-model"), user_api_key_dict=admin)

        assert PTU_COST_ATTRIBUTION_ENV_VAR in str(exc.value)
        add_model_to_db.assert_not_called()
        add_team_model_to_db.assert_not_called()

    @pytest.mark.asyncio
    async def test_model_new_accepts_a_deployment_without_ptu_config_while_disabled(self):
        _, patches = self._patched_proxy("plain-model")
        admin = UserAPIKeyAuth(user_id="test-admin", user_role=LitellmUserRoles.PROXY_ADMIN)

        with ExitStack() as stack:
            for active_patch in patches:
                stack.enter_context(active_patch)
            result = await add_new_model(
                model_params=Deployment(
                    model_name="ptu-model",
                    litellm_params=LiteLLM_Params(model="openai/gpt-4.1-nano", api_key="fake-key"),
                    model_info=ModelInfo(id="plain-model"),
                ),
                user_api_key_dict=admin,
            )

        assert result.model_id == "plain-model"

    @pytest.mark.asyncio
    async def test_model_new_accepts_ptu_config_once_enabled(self, monkeypatch):
        monkeypatch.setenv(PTU_COST_ATTRIBUTION_ENV_VAR, "true")
        (_, add_team_model_to_db), patches = self._patched_proxy("ptu-gate-model")
        admin = UserAPIKeyAuth(user_id="test-admin", user_role=LitellmUserRoles.PROXY_ADMIN)

        with ExitStack() as stack:
            for active_patch in patches:
                stack.enter_context(active_patch)
            result = await add_new_model(model_params=self._ptu_deployment("ptu-gate-model"), user_api_key_dict=admin)

        assert result.model_id == "ptu-gate-model"
        add_team_model_to_db.assert_called_once()

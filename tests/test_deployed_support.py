"""The deployable agent folder, checked without a cloud.

`examples/deployed_support` is what `scripts/deploy_agent_engine.sh` puts on Agent Platform,
and the ways it can be wrong are quiet ones. The wrong plugin order throws away every routing
decision and the agent still answers. A missing cache config makes the pruning save almost
nothing and the agent still answers. A requirements file that drifts from pyproject.toml
builds an image that imports and then fails on the first embedding call.

So these are the checks a deploy cannot make for itself, run for free on every push.
`scripts/verify_agent_engine.py` makes the ones that need a real deployment.
"""

from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path

import pytest
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin

from adk_smart_router import SmartRouterPlugin
from adk_smart_router.config import RouterConfig
from adk_smart_router.provenance import PROVENANCE_KEY

REPO_ROOT = Path(__file__).resolve().parent.parent
FOLDER = REPO_ROOT / "examples" / "deployed_support"
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy_agent_engine.sh"
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_agent_engine.py"


@pytest.fixture
def app(monkeypatch):
    """The deployed `App`, built the way the container builds it.

    The module reads the environment at import and refuses to come up without a project, so
    the variables go in first and the module is imported fresh each time.
    """
    monkeypatch.setenv("SMART_ROUTER_BQ_PROJECT", "a-project")
    monkeypatch.setenv("SMART_ROUTER_BQ_DATASET", "smart_router")
    monkeypatch.setenv("SMART_ROUTER_GCS_BUCKET", "a-bucket")
    module = importlib.import_module("deployed_support.agent")
    return importlib.reload(module).app


def test_the_module_exports_app_and_not_only_root_agent(app):
    """ADK's loader checks for `app` before it checks for `root_agent`, and a folder that
    exports only the second loses its plugins on the way to production with no warning."""
    from google.adk.apps import App

    assert isinstance(app, App)


def test_the_router_is_registered_before_the_analytics_plugin(app):
    """The other order persists the event before the router has stamped provenance onto it,
    so every routing decision is dropped and the warehouse says the router never ran
    (DESIGN.md 6.3). `smart-router doctor` checks this; so does this, without a cloud."""
    kinds = [type(p) for p in app.plugins]
    assert SmartRouterPlugin in kinds
    assert BigQueryAgentAnalyticsPlugin in kinds
    assert kinds.index(SmartRouterPlugin) < kinds.index(BigQueryAgentAnalyticsPlugin)


def test_the_analytics_plugin_allows_the_provenance_key(app):
    """Without the allowlist entry BQAA drops the router's metadata, which looks exactly
    like the wrong plugin order and has a different cause."""
    analytics = next(p for p in app.plugins if isinstance(p, BigQueryAgentAnalyticsPlugin))
    # `.config`, which is where cli.py's doctor check reads it from too. The constructor
    # keyword and the attribute are not the same name.
    assert PROVENANCE_KEY in (analytics.config.custom_metadata_allowlist or [])


def test_context_caching_is_configured(app):
    """Pruning the declared list saves almost nothing without explicit caching, because the
    declarations removed would otherwise bill at the uncached rate. 2048 is the 2.5 Flash
    floor (DESIGN.md 6.3)."""
    assert app.context_cache_config is not None
    assert app.context_cache_config.min_tokens == 2048


def test_it_deploys_the_reference_agent_and_so_loads_the_published_artifact(app):
    """Artifact selection keys on the agent's name, not the app's. The deployed agent has to
    stay `support` or it will look for a model nobody trained."""
    assert app.root_agent.name == "support"
    assert len(app.root_agent.tools) == 100


def test_requirements_match_the_package_dependencies():
    """The image installs this file, not pyproject.toml. While the repository is private the
    router itself is staged as source rather than installed, so its dependencies do not come
    along on their own and this list is the only thing that brings them."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    declared = pyproject["project"]["dependencies"]
    lines = [
        line.strip()
        for line in (FOLDER / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines == declared


def test_the_app_name_is_the_folder_basename_everywhere():
    """`adk deploy` derives the deployed resource's name from the folder, and both scripts
    have to agree with it: one to find an existing deployment to update, the other to find
    the deployment to drive."""
    assert FOLDER.name == "deployed_support"
    assert 'APP_NAME="deployed_support"' in DEPLOY_SCRIPT.read_text()
    assert 'APP_NAME = "deployed_support"' in VERIFY_SCRIPT.read_text()


def test_the_deploy_script_names_no_project():
    """The same rule the rest of the scripts follow. A default project is a deploy that bills
    somebody else, and the failure reads as a permissions error rather than a missing export.
    The .env the script writes is generated at run time into a temp directory."""
    text = DEPLOY_SCRIPT.read_text()
    assert "SMART_ROUTER_BQ_PROJECT" in text
    assert re.search(r"require_env[^\n]*SMART_ROUTER_BQ_PROJECT", text)
    assert 'PROJECT_ID="${SMART_ROUTER_BQ_PROJECT:-}"' in text


def test_the_deploy_script_does_not_override_the_two_settings_that_look_like_locations():
    """Two settings read like "where the deployment runs" and are not.

    `environment` is matched against the environment an artifact was published under.
    `vertex_location` is where the embedding model is served. The first real deploy of this
    script set the first to "agent-platform" and the second to the deployment's own region,
    and both readings are wrong in the same way: the value describes something other than the
    runtime. The agent answered every question and routed nothing, twice, because the router
    fails open on both.

    So neither may differ from `RouterConfig`. Settings that are deliberately different, like
    route_fraction, are different because a smoke test that routes nothing proves nothing;
    these two are only ever different by mistake.
    """
    text = DEPLOY_SCRIPT.read_text()
    defaults = RouterConfig()
    for var, shell, value in (
        ("SMART_ROUTER_ENVIRONMENT", "ENVIRONMENT", defaults.environment),
        ("SMART_ROUTER_VERTEX_LOCATION", "VERTEX_LOCATION", defaults.vertex_location),
    ):
        assert f'{shell}="${{{var}:-{value}}}"' in text
        assert f"{var}=${{{shell}}}" in text


def test_the_deploy_script_refuses_to_act_without_confirm():
    """It creates a resource that bills until it is deleted."""
    text = DEPLOY_SCRIPT.read_text()
    assert "--confirm" in text
    assert '[[ "$CONFIRMED" != true ]]' in text
    assert "--delete" in text


def test_the_training_corpus_is_not_staged_into_the_image():
    """7 MB of synthetic conversations that a serving container never opens. ADK applies no
    ignore patterns to extra packages, so the trimming has to be explicit."""
    text = DEPLOY_SCRIPT.read_text()
    # Named files and one named directory, rather than the whole folder. The assertion is on
    # the copies, because that is the shape a later `cp -r "$REFERENCE_SRC"` would break.
    assert 'cp "$REFERENCE_SRC"/*.py "$dst/"' in text
    assert 'cp -r "$REFERENCE_SRC/tools" "$dst/tools"' in text
    assert 'cp -r "$REFERENCE_SRC"' not in text

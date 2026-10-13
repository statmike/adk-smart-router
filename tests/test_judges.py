"""Tests for judges.py and its integration into SmartRouterPlugin.

Covers:
1. `ToolSpec` and `request_edits.tool_specs()` across both `parameters_json_schema` and
   `parameters.properties` plus full Python docstrings (`func.__doc__`).
2. `PrototypeJudge`: catalog-centroid centering, schema-graph prerequisite propagation
   (producers, inspectors, high-in-degree schema hubs, and escalation fallbacks),
   per-agent catalog caching, and fail-open behavior on embedding errors.
3. `LiveLlmJudge`: structured JSON tool selection, rank-to-probability conversion, and
   timeout/error fail-open.
4. `combine_predictions` and `SmartRouterPlugin` integration:
   - Cold-start routing when `ModelStore.load` returns `None` (no GCS artifact yet).
   - Cold-start fallback when the GCS artifact has aged past `max_model_age_days`.
   - Hybrid `("xgboost", "prototype")`: pruning uncalled classless tools while rescuing
     newly deployed classless tools that semantically match the prompt.
   - Custom `ToolJudge` protocol plug-in (standing in for future low-latency `TabFM` or
     `JEV` endpoints).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest
from google.adk.agents.run_config import StreamingMode
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.genai import types

from adk_smart_router import plugin as plugin_mod
from adk_smart_router.config import RouterConfig, load_config, min_cache_tokens
from adk_smart_router.judges import (
    LiveLlmJudge,
    PrototypeJudge,
    ToolJudge,
)
from adk_smart_router.model_store import ArtifactMetadata, RouterModel
from adk_smart_router.plugin import SmartRouterPlugin
from adk_smart_router.predictor import Prediction
from adk_smart_router.provenance import PROVENANCE_KEY
from adk_smart_router.request_edits import ToolSpec, tool_specs


class FakeSession:
    def __init__(self, app_name: str, session_id: str) -> None:
        self.app_name = app_name
        self.id = session_id


class FakeRunConfig:
    def __init__(self, streaming_mode: StreamingMode = StreamingMode.NONE) -> None:
        self.streaming_mode = streaming_mode


class FakePluginManager:
    def __init__(self, plugins: list[Any]) -> None:
        self.plugins = plugins


class FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.before_tool_callback = None
        self.canonical_before_tool_callbacks: list[Any] = []
        self.parent_agent = None
        self.sub_agents: list[Any] = []


class FakeInvocationContext:
    def __init__(self, agent: FakeAgent) -> None:
        self.agent = agent
        self.plugin_manager = FakePluginManager([])
        self.session_service = object()
        self.context_cache_config = "set"
        self._custom_metadata: dict[str, Any] = {}


class FakeContext:
    def __init__(
        self,
        *,
        agent_name: str = "root",
        app_name: str = "support",
        user_id: str = "u1",
        session_id: str = "s1",
    ) -> None:
        self.agent_name = agent_name
        self.user_id = user_id
        self.session = FakeSession(app_name, session_id)
        self.run_config = FakeRunConfig(StreamingMode.NONE)
        self.state: dict[str, Any] = {}
        self.invocation_id = "i1"
        self._ictx = FakeInvocationContext(FakeAgent(agent_name))
        self.custom_metadata: dict[str, Any] = self._ictx._custom_metadata  # noqa: SLF001

    def get_invocation_context(self) -> FakeInvocationContext:
        return self._ictx


class FakePredictor:
    def __init__(self, preds: tuple[Prediction, ...]) -> None:
        self.preds = preds

    def predict(self, vector: Any) -> tuple[Prediction, ...]:
        return self.preds


class FakeStore:
    def __init__(self, model: RouterModel | None) -> None:
        self.model = model

    def load(self, agent_name: str) -> RouterModel | None:
        return self.model

    def latest_version(self, agent_name: str) -> str | None:
        return self.model.metadata.version if self.model is not None else None


def declared_names(request: LlmRequest) -> list[str]:
    found: list[str] = []
    for tool in request.config.tools or []:
        for declaration in tool.function_declarations or []:
            if declaration.name:
                found.append(declaration.name)
    return found


class SemanticEmbedder:
    """Deterministic keyword-vector embedder so semantic similarity is exact and fast."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls: list[str] = []
        self._vocab = (
            "return",
            "refund",
            "order",
            "sku",
            "product",
            "price",
            "invoice",
            "billing",
            "subscription",
            "shipping",
            "carrier",
            "warranty",
            "loyalty",
            "points",
            "freight",
            "customs",
            "escalate",
            "human",
        )

    async def embed(self, text: str) -> list[float] | None:
        vec, _ = await self.embed_with_reason(text)
        return vec

    async def embed_with_reason(self, text: str) -> tuple[list[float] | None, str]:
        self.calls.append(text)
        if self.fail_on and (self.fail_on in text or self.fail_on.replace("_", " ") in text):
            return None, "error"
        lower = text.lower()
        vec = [float(lower.count(word)) for word in self._vocab]
        # Small baseline component so every vector has non-zero norm.
        vec.append(0.1)
        norm = float(np.linalg.norm(vec))
        return [v / norm for v in vec], ""

    async def warmup(self) -> bool:
        return True

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": 0, "misses": len(self.calls), "timeouts": 0, "failures": 0, "skipped": 0}


def _make_catalog_specs() -> dict[str, ToolSpec]:
    """24-tool support catalog with producers, consumer tools, and classless tools."""
    specs: dict[str, ToolSpec] = {
        "search_products": ToolSpec(
            name="search_products",
            description="Search the product catalogue by keyword.\nReturns: matching sku, title.",
            parameters=(("query", "Search terms"),),
        ),
        "list_orders": ToolSpec(
            name="list_orders",
            description="List recent orders for a customer.\nReturns: order_id, date, total.",
            parameters=(("customer_id", "Customer identifier"),),
        ),
        "get_order_status": ToolSpec(
            name="get_order_status",
            description="Fetch full order details and line items.",
            parameters=(("order_id", "Order identifier"),),
        ),
        "start_return": ToolSpec(
            name="start_return",
            description="Open a return RMA for an item on an order and issue a return label.",
            parameters=(("order_id", "Order identifier"), ("sku", "Product SKU")),
        ),
        "get_price": ToolSpec(
            name="get_price",
            description="Look up unit price and discount for a product SKU.",
            parameters=(("sku", "Product SKU"), ("order_id", "Order identifier")),
        ),
        "get_invoice": ToolSpec(
            name="get_invoice",
            description="Retrieve billing invoice PDF and tax breakdown for an order.",
            parameters=(("order_id", "Order identifier"),),
        ),
        "list_subscriptions": ToolSpec(
            name="list_subscriptions",
            description="List recurring subscriptions for a customer.\nReturns: subscription_id.",
            parameters=(("customer_id", "Customer identifier"),),
        ),
        "cancel_subscription": ToolSpec(
            name="cancel_subscription",
            description="Cancel an active recurring subscription.",
            parameters=(("subscription_id", "Subscription identifier"), ("sku", "Product SKU")),
        ),
        "escalate_to_human": ToolSpec(
            name="escalate_to_human",
            description="Escalate the conversation to a human support specialist.",
            parameters=(("reason", "Escalation summary"),),
        ),
        "track_freight_customs": ToolSpec(
            name="track_freight_customs",
            description="Check international freight customs clearance and duties.",
            parameters=(("order_id", "Order identifier"), ("sku", "Product SKU")),
        ),
    }
    for i in range(14):
        name = f"filler_tool_{i:02d}"
        specs[name] = ToolSpec(
            name=name,
            description=f"Filler tool {i} for warehouse maintenance.",
            parameters=(("order_id", "Order identifier"), ("sku", "Product SKU")),
        )
    return specs


def _make_rich_request(specs: dict[str, ToolSpec], prompt: str) -> LlmRequest:
    declarations = []
    tools_dict = {}
    for name, spec in specs.items():
        declarations.append(
            types.FunctionDeclaration(
                name=name,
                description=spec.description.splitlines()[0] if spec.description else "",
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        k: {"type": "string", "description": d} for k, d in spec.parameters
                    },
                },
            )
        )
        tool = BaseTool(name=name, description=spec.description)

        def _dummy() -> None:
            pass

        _dummy.__doc__ = spec.description
        tool.func = _dummy  # type: ignore[attr-defined]
        tools_dict[name] = tool

    return LlmRequest(
        model=None,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=types.GenerateContentConfig(tools=[types.Tool(function_declarations=declarations)]),
        tools_dict=tools_dict,
    )


def test_config_loads_judges_and_prototype_weight_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SMART_ROUTER_JUDGES", "xgboost, prototype")
    monkeypatch.setenv("SMART_ROUTER_PROTOTYPE_WEIGHT", "0.22")
    monkeypatch.setenv("SMART_ROUTER_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("SMART_ROUTER_EMBEDDING_DIMS", "768")
    monkeypatch.setenv("SMART_ROUTER_LLM_JUDGE_MODEL", "gemini-3.5-flash-lite")
    cfg = load_config()
    assert cfg.judges == ("xgboost", "prototype")
    assert cfg.prototype_weight == pytest.approx(0.22)
    assert cfg.embedding_model == "gemini-embedding-001"
    assert cfg.embedding_dims == 768
    assert cfg.llm_judge_model == "gemini-3.5-flash-lite"
    assert min_cache_tokens("gemini-3.7-flash") == 4096
    assert min_cache_tokens("gemini-flash-latest") == 4096


def test_tool_specs_reads_json_schema_and_full_python_docstring() -> None:
    specs = _make_catalog_specs()
    req = _make_rich_request(specs, "I need to return an item from my order")
    extracted = tool_specs(req)
    assert set(extracted.keys()) == set(specs.keys())
    assert "Returns: matching sku" in extracted["search_products"].description
    assert extracted["start_return"].param_names == frozenset({"order_id", "sku"})


async def test_prototype_judge_boosts_prerequisite_producers_and_caches_catalog() -> None:
    cfg = RouterConfig(judges=("prototype",), cold_start_max_tools_kept=8)
    embedder = SemanticEmbedder()
    judge = PrototypeJudge(cfg, embedder)
    specs = _make_catalog_specs()

    prompt = "I want to return a broken item and get a refund"
    q_vec = await embedder.embed(prompt)
    calls_before = len(embedder.calls)

    preds = await judge.predict(prompt, q_vec, specs, agent_name="support_agent")
    calls_after_first = len(embedder.calls)
    assert calls_after_first - calls_before == len(specs)

    top_names = [p.label for p in preds[:6]]
    assert "start_return" in top_names
    # Prerequisite producers for `order_id` and `sku` (`list_orders`, `search_products`)
    # are automatically promoted via the schema prerequisite matrix `W` even though the
    # prompt never mentioned "search" or "list".
    assert "list_orders" in top_names or "search_products" in top_names

    # Second call for the same agent reuses the compiled CatalogIndex with 0 extra embeds.
    preds_second = await judge.predict(prompt, q_vec, specs, agent_name="support_agent")
    assert len(embedder.calls) == calls_after_first
    assert [p.label for p in preds_second] == [p.label for p in preds]


async def test_prototype_judge_fails_open_when_tool_embedding_fails() -> None:
    cfg = RouterConfig(judges=("prototype",))
    embedder = SemanticEmbedder(fail_on="start_return")
    judge = PrototypeJudge(cfg, embedder)
    specs = _make_catalog_specs()
    q_vec = await embedder.embed("return my order")
    preds = await judge.predict("return my order", q_vec, specs, agent_name="support_agent")
    assert preds == ()


async def test_plugin_routes_in_cold_start_when_no_gcs_artifact_exists() -> None:
    specs = _make_catalog_specs()
    embedder = SemanticEmbedder()
    plugin = SmartRouterPlugin(
        RouterConfig(
            mode="active",
            route_fraction=1.0,
            holdout_fraction=0.0,
            judges=("prototype",),
            cold_start_max_tools_kept=8,
            miss_cost_tokens=0,
        ),
        model_store=FakeStore(None),  # No artifact in GCS!
        embedder=embedder,  # type: ignore[arg-type]
    )
    ctx = FakeContext(session_id="cold-1")
    req = _make_rich_request(specs, "I need to return an item from my order")

    await plugin.before_model_callback(callback_context=ctx, llm_request=req)  # type: ignore[arg-type]
    kept = declared_names(req)
    assert len(kept) < len(specs)
    assert "start_return" in kept
    record = ctx.custom_metadata[PROVENANCE_KEY][0]
    assert record["model_version"] == "cold_start:prototype"
    assert record["decision_source"] == "fresh"


async def test_plugin_falls_back_to_prototype_when_artifact_ages_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = _make_catalog_specs()
    embedder = SemanticEmbedder()
    stale_date = (datetime.now(UTC) - timedelta(days=30)).strftime("%Y-%m-%d")
    meta = ArtifactMetadata(
        version="20260101T000000Z",
        sha256="a" * 64,
        size_bytes=10,
        trained_at=stale_date,
        rows=100,
        classes=tuple(specs.keys()),
        backend="vertex",
        environment="prod",
        embedding_model="gemini-embedding-2",
        embedding_dims=768,
        gate_metrics={"recall": 0.99},
    )
    stale_model = RouterModel(
        agent_name="root", metadata=meta, booster_path="/nonexistent/model.ubj"
    )
    monkeypatch.setattr(
        plugin_mod, "Predictor", lambda m: FakePredictor((Prediction("get_invoice", 0.99),))
    )

    plugin = SmartRouterPlugin(
        RouterConfig(
            mode="active",
            route_fraction=1.0,
            holdout_fraction=0.0,
            judges=("xgboost", "prototype"),
            max_model_age_days=14,
            cold_start_max_tools_kept=8,
            miss_cost_tokens=0,
        ),
        model_store=FakeStore(stale_model),
        embedder=embedder,  # type: ignore[arg-type]
    )
    ctx = FakeContext(session_id="stale-fallback-1")
    req = _make_rich_request(specs, "I need to return an item from my order")

    await plugin.before_model_callback(callback_context=ctx, llm_request=req)  # type: ignore[arg-type]
    kept = declared_names(req)
    assert len(kept) < len(specs)
    assert "start_return" in kept
    assert ctx.custom_metadata[PROVENANCE_KEY][0]["model_version"] == "cold_start:prototype"


async def test_hybrid_xgboost_and_prototype_prunes_uncalled_and_rescues_new_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = _make_catalog_specs()
    embedder = SemanticEmbedder()
    # Train XGBoost on all tools EXCEPT `track_freight_customs` and the 14 filler tools
    # (simulating 14 uncalled tools + 1 newly deployed `track_freight_customs` tool).
    trained_classes = (
        "search_products",
        "list_orders",
        "get_order_status",
        "start_return",
        "get_price",
        "get_invoice",
        "list_subscriptions",
        "cancel_subscription",
        "escalate_to_human",
    )
    fresh_date = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
    menus = (frozenset({"list_orders", "get_order_status", "start_return", "escalate_to_human"}),)
    meta = ArtifactMetadata(
        version="v1-hybrid",
        sha256="b" * 64,
        size_bytes=10,
        trained_at=fresh_date,
        rows=500,
        classes=trained_classes,
        backend="vertex",
        environment="prod",
        embedding_model="gemini-embedding-2",
        embedding_dims=768,
        gate_metrics={"recall": 0.99},
        menus=menus,
    )
    model = RouterModel(agent_name="root", metadata=meta, booster_path="/nonexistent/model.ubj")
    monkeypatch.setattr(
        plugin_mod,
        "Predictor",
        lambda m: FakePredictor(
            (
                Prediction("get_order_status", 0.60),
                Prediction("list_orders", 0.35),
                Prediction("start_return", 0.05),
            )
        ),
    )

    plugin = SmartRouterPlugin(
        RouterConfig(
            mode="active",
            route_fraction=1.0,
            holdout_fraction=0.0,
            judges=("xgboost", "prototype"),
            miss_cost_tokens=0,
        ),
        model_store=FakeStore(model),
        embedder=embedder,  # type: ignore[arg-type]
    )

    # 1. Ordinary return prompt: the 14 uncalled `filler_tool_*` AND `track_freight_customs`
    #    are pruned instead of riding on every prompt via `unknown`.
    ctx1 = FakeContext(session_id="hyb-1")
    req1 = _make_rich_request(specs, "I need to return an item from my order")
    await plugin.before_model_callback(callback_context=ctx1, llm_request=req1)  # type: ignore[arg-type]
    kept1 = set(declared_names(req1))
    assert kept1 == {"list_orders", "get_order_status", "start_return", "escalate_to_human"}

    # 2. Prompt specifically asking for the newly deployed `track_freight_customs` tool:
    #    `PrototypeJudge` ranks `track_freight_customs` #1 and rescues it onto the kept menu!
    ctx2 = FakeContext(session_id="hyb-2")
    req2 = _make_rich_request(
        specs, "Check international freight customs clearance and customs duties on my order"
    )
    await plugin.before_model_callback(callback_context=ctx2, llm_request=req2)  # type: ignore[arg-type]
    kept2 = set(declared_names(req2))
    assert "track_freight_customs" in kept2
    assert not any(name.startswith("filler_tool_") for name in kept2)


async def test_live_llm_judge_parses_json_and_handles_failures() -> None:
    captured_calls: list[dict[str, Any]] = []

    class FakeModels:
        def __init__(self, payload: str, *, should_raise: bool = False) -> None:
            self.payload = payload
            self.should_raise = should_raise

        def generate_content(self, **kwargs: Any) -> Any:
            captured_calls.append(kwargs)
            if self.should_raise:
                raise RuntimeError("API unavailable")

            class Resp:
                text = self.payload

            return Resp()

    class FakeGenaiClient:
        def __init__(self, payload: str, *, should_raise: bool = False) -> None:
            self.models = FakeModels(payload, should_raise=should_raise)

    specs = _make_catalog_specs()
    cfg = RouterConfig(judges=("llm_live",), cold_start_max_tools_kept=5)

    judge = LiveLlmJudge(
        cfg,
        client_factory=lambda: FakeGenaiClient(  # type: ignore[return-value]
            '["list_orders", "get_order_status", "start_return", "nonexistent_tool"]'
        ),
    )
    long_prompt = "return my item " + ("x" * 5000)
    preds = await judge.predict(long_prompt, None, specs)
    assert [p.label for p in preds[:3]] == ["list_orders", "get_order_status", "start_return"]
    assert sum(p.probability for p in preds) == pytest.approx(1.0)
    assert len(captured_calls) == 1
    assert len(captured_calls[0]["contents"]) == 4000
    assert "Catalog:" in captured_calls[0]["config"].system_instruction

    failing_judge = LiveLlmJudge(
        cfg,
        client_factory=lambda: FakeGenaiClient("", should_raise=True),  # type: ignore[return-value]
    )
    assert await failing_judge.predict("return my item", None, specs) == ()


async def test_prototype_judge_bounds_menu_cache_and_docstring_churn() -> None:
    cfg = RouterConfig(judges=("prototype",), cold_start_max_tools_kept=8)
    embedder = SemanticEmbedder()
    judge = PrototypeJudge(cfg, embedder)
    specs = _make_catalog_specs()
    q_vec = await embedder.embed("return my order")

    for turn in range(8):
        mutated = dict(specs)
        mutated["start_return"] = ToolSpec(
            name="start_return",
            description=f"Open a return RMA (dynamic turn {turn}).",
            parameters=(("order_id", "Order identifier"), ("sku", "Product SKU")),
        )
        await judge.predict("return my order", q_vec, mutated, agent_name="churn_agent")
        judge.menus_for("churn_agent", size=8, count=4)

    # Menu cache is strictly bounded to 1 entry per agent_name, and description-only
    # rebuilds stop after _MAX_INDEX_REBUILDS_PER_AGENT (4).
    assert len(judge._menus) == 1  # noqa: SLF001
    assert judge._rebuilds["churn_agent"] == 4  # noqa: SLF001


async def test_custom_tool_judge_protocol_plugs_in_future_endpoint() -> None:
    """Verify a future low-latency TabFM or JEV endpoint can plug into SmartRouterPlugin."""

    class MockTabFMJudge:
        @property
        def name(self) -> str:
            return "tabfm_endpoint"

        @property
        def weight(self) -> float:
            return 1.0

        async def predict(
            self,
            prompt: str,
            vector: Any,
            specs: dict[str, ToolSpec],
            *,
            agent_name: str = "",
        ) -> list[Prediction]:
            return [
                Prediction("get_invoice", 0.55),
                Prediction("list_orders", 0.40),
                Prediction("escalate_to_human", 0.05),
            ]

    custom = MockTabFMJudge()
    assert isinstance(custom, ToolJudge)

    specs = _make_catalog_specs()
    plugin = SmartRouterPlugin(
        RouterConfig(
            mode="active",
            route_fraction=1.0,
            holdout_fraction=0.0,
            judges=("tabfm_endpoint",),
            cold_start_max_tools_kept=4,
            miss_cost_tokens=0,
        ),
        model_store=FakeStore(None),
        embedder=SemanticEmbedder(),  # type: ignore[arg-type]
        custom_judges=(custom,),
    )
    ctx = FakeContext(session_id="tabfm-1")
    req = _make_rich_request(specs, "send me my billing invoice")
    await plugin.before_model_callback(callback_context=ctx, llm_request=req)  # type: ignore[arg-type]
    kept = set(declared_names(req))
    assert kept == {"get_invoice", "list_orders"}
    assert ctx.custom_metadata[PROVENANCE_KEY][0]["model_version"] == "cold_start:tabfm_endpoint"


def test_top_level_package_exports_tool_judge() -> None:
    import adk_smart_router

    assert "ToolJudge" in adk_smart_router.__all__
    assert adk_smart_router.ToolJudge is ToolJudge


def test_config_loads_cold_start_menu_count_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SMART_ROUTER_COLD_START_MENU_COUNT", "24")
    cfg = load_config()
    assert cfg.cold_start_menu_count == 24


async def test_build_catalog_menus_and_plugin_cold_start_canonical_menu_snapping() -> None:
    from adk_smart_router.judges import build_catalog_index, build_catalog_menus

    specs = _make_catalog_specs()
    embedder = SemanticEmbedder()
    names = tuple(sorted(specs.keys()))
    vecs = [await embedder.embed(specs[n].document()) for n in names]
    valid_vecs = [v for v in vecs if v is not None]
    index = build_catalog_index(names, valid_vecs, specs)

    menus1 = build_catalog_menus(index, size=8, count=6)
    menus2 = build_catalog_menus(index, size=8, count=6)
    assert menus1 == menus2
    assert 1 <= len(menus1) <= 6
    assert all(len(m) <= 8 for m in menus1)
    assert build_catalog_menus(index, size=len(names), count=6) == ()

    plugin = SmartRouterPlugin(
        RouterConfig(
            mode="active",
            route_fraction=1.0,
            holdout_fraction=0.0,
            judges=("prototype",),
            cold_start_max_tools_kept=8,
            cold_start_menu_count=6,
            miss_cost_tokens=0,
        ),
        model_store=FakeStore(None),
        embedder=SemanticEmbedder(),  # type: ignore[arg-type]
    )
    ctx = FakeContext(session_id="cold-menu-1")
    req = _make_rich_request(specs, "I need to return an item from my order")
    await plugin.before_model_callback(callback_context=ctx, llm_request=req)  # type: ignore[arg-type]
    kept = frozenset(declared_names(req))
    assert kept in menus1
    assert "start_return" in kept
    assert ctx.custom_metadata[PROVENANCE_KEY][0]["model_version"] == "cold_start:prototype"

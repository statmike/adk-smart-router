#!/usr/bin/env python
"""V2-1 — score the published router as a set predictor, on the model we already have.

`check_router.py` asks whether the artifact predicts the way the training run said it
did, in the training run's own terms: one prompt, one tool, is it in the top seven. This
asks the different question that DESIGN.md 19 is built on. The router declares a **set**,
**once**, for a **session**. So: given a session's first message, does the menu it ships
contain the tools that session went on to want, and how many declaration tokens did it
manage to leave behind?

No retrain. That is the point of doing this first. The same artifact, the same menus, the
same `pruner.decide` the plugin calls, scored under DESIGN.md 19.4 instead of 10. What
comes out separates two things the old metric could not tell apart: how much of the 8.7%
was the frame, and how much was the model.

Five arms, and the gaps between them are the answer:

    full list       no pruning. Recall 1.0 by construction, nothing saved.
    always prune    every menu `pruner.decide` produces, sent. The decline rule turned off,
                    which is the ceiling on saving and the floor on recall.
    router @N       the shipped rule: the same menus, sent only when DESIGN.md 19.5's
                    arithmetic says the prune pays at a miss priced at N billed tokens.
    best menu       the published menu that best covers what the session truly wanted.
                    An oracle: the ceiling this menu pool allows.
    true set        the true tools plus whatever is protected. A second oracle, with no
                    menu pool at all.

full list to router is what is being delivered. Router to best menu is what better
prediction could buy, and always prune is how much of that gap the decline rule is giving
back to hold recall. Best menu to true set is what the menu pool costs.

`--sweep` runs one router arm per miss price instead of one, because the price is the knob
and a single point hides the frontier it sits on.

Sessions are read unrouted, `corpus-` by default, because a routed session's tool set is
censored by the prune it received (DESIGN.md 11.4) and scoring a predictor on its own
output flatters it.

Cost is two BigQuery reads and 25 token counts. No embedding calls: the vectors are the
ones BigQuery already computed. Authentication is ADC only.

Usage:
    uv run python scripts/score_sets.py
    uv run python scripts/score_sets.py --session-prefix ab-20260910b- --eval-only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from google.cloud import bigquery

# The package is not installed into the venv; pytest reaches it through `pythonpath` in
# pyproject.toml and scripts have to do the same by hand.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from deployment import DEFAULT_DATASET, bucket_arg, project_arg  # noqa: E402
from measure_menus import declaration_tokens  # noqa: E402

from adk_smart_router.config import FRAMEWORK_TOOLS, RouterConfig, load_config  # noqa: E402
from adk_smart_router.embedding import Embedder  # noqa: E402
from adk_smart_router.judges import (  # noqa: E402
    CatalogIndex,
    build_catalog_index,
    build_catalog_menus,
    combine_predictions,
    score_catalog,
)
from adk_smart_router.model_store import ModelStore  # noqa: E402
from adk_smart_router.predictor import Prediction, Predictor  # noqa: E402
from adk_smart_router.pruner import (  # noqa: E402
    PredictionLike,
    contained_mass,
    decide,
    worth_pruning,
)
from adk_smart_router.request_edits import ToolSpec  # noqa: E402

DEFAULT_AGENT = "support"

# The training procedure's split (DESIGN.md 9.4), repeated rather than parameterised for
# the same reason check_router.py repeats it: a silent divergence is the bug worth catching.
EVAL_PCT = 25

# Vertex bills a cached input token at a tenth of an uncached one, which is the whole of
# why a removed declaration is worth 0.1 of itself and not 1.0 (DESIGN.md 18.16).
CACHED_RATE = 0.10

# The F-beta of DESIGN.md 19.4, weighting recall over precision. Beta comes out of 19.1:
# an extra tool costs about 6.5 billed tokens a turn and a missing one about 2278, a ratio
# near 75, and F-beta weights recall beta-squared times precision. 8 squared is 64.
BETA = 8

# One row per session: the first user message, the tools the session went on to want, and
# what it actually cost. The tool set is DESIGN.md 19.2 — attempted, not merely successful,
# because a tool the model tried and got an error from still had to be declared for the
# attempt to happen.
SESSIONS_SQL = """
WITH censored AS (
  -- Sessions the router pruned. Their tool set is not what the session wanted, it is what
  -- the session was left able to want, so scoring a menu against it grades the router on
  -- agreeing with itself. `train_router.sql` drops these from the training corpus for the
  -- same reason (DESIGN.md 11.4); this is the same filter on the scoring side, and the two
  -- have to stay identical. 'fresh' and 'reused' are the sources that mean a prune really
  -- happened; they are `provenance.ROUTED_SOURCES`, and `tests/test_decision_sources.py`
  -- pins this copy to it. There was a third, 'grown', deleted with the pruner's growth path;
  -- it never occurred in the warehouse, so dropping it censors exactly the same sessions as
  -- before. Do not re-add it from an old schema.
  --
  -- On a generated corpus this removes nothing, because nothing was routed. It earns its
  -- keep the first time the tuner is pointed at live traffic, where all that is left
  -- uncensored is the holdout arm and the sessions the decline rule passed through.
  SELECT DISTINCT e.session_id
  FROM `{project}.{dataset}.agent_events` AS e,
    UNNEST(JSON_QUERY_ARRAY(e.attributes, '$.custom_metadata.smart_router')) AS record
  WHERE e.agent = @agent
    AND JSON_VALUE(record, '$.tool_name') IS NULL
    AND JSON_VALUE(record, '$.decision_source') IN ('fresh', 'reused')
),
opening AS (
  -- The id of each session's first message, not the message itself: BigQuery cannot
  -- ARRAY_AGG a STRUCT that contains an ARRAY, and `v` is one.
  SELECT
    session_id,
    ARRAY_AGG(invocation_id ORDER BY prompt_ts LIMIT 1)[OFFSET(0)] AS invocation_id
  FROM `{project}.{dataset}.router_features_{agent}`
  WHERE ARRAY_LENGTH(v) = @dims
    AND prompt_text IS NOT NULL
    AND STARTS_WITH(session_id, @prefix)
    AND session_id NOT IN (SELECT session_id FROM censored)
  GROUP BY session_id
),
first_message AS (
  SELECT r.session_id, r.prompt_text, r.v
  FROM `{project}.{dataset}.router_features_{agent}` r
  JOIN opening o USING (invocation_id)
),
attempted AS (
  SELECT session_id, tool_name
  FROM `{project}.{dataset}.v_tool_completed`
  WHERE agent = @agent AND status = 'OK'
    AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NULL
  UNION DISTINCT
  SELECT session_id, tool_name
  FROM `{project}.{dataset}.v_tool_error`
  WHERE agent = @agent
),
spend AS (
  SELECT
    session_id,
    COUNT(*) AS turns,
    SUM(usage_prompt_tokens - LEAST(COALESCE(usage_cached_tokens, 0), usage_prompt_tokens)
        + LEAST(COALESCE(usage_cached_tokens, 0), usage_prompt_tokens) * {rate}) AS billed
  FROM `{project}.{dataset}.v_llm_response`
  WHERE agent = @agent AND usage_prompt_tokens IS NOT NULL
  GROUP BY session_id
)
SELECT
  f.session_id,
  f.prompt_text,
  f.v,
  IFNULL((SELECT ARRAY_AGG(DISTINCT a.tool_name ORDER BY a.tool_name)
          FROM attempted a
          WHERE a.session_id = f.session_id AND a.tool_name != 'transfer_to_agent'), [])
    AS tool_set,
  IFNULL(s.turns, 0) AS turns,
  IFNULL(s.billed, 0.0) AS billed
FROM first_message f
LEFT JOIN spend s USING (session_id)
{eval_filter}
"""


@dataclass(frozen=True)
class Session:
    """One session: what it opened with, what it turned out to need, what it cost."""

    session_id: str
    vector: list[float]
    tools: frozenset[str]
    turns: int
    billed: float


@dataclass
class Arm:
    """Running totals for one way of choosing a menu."""

    name: str
    sessions: int = 0
    declined: int = 0
    recall_sum: float = 0.0
    recall_n: int = 0
    fbeta_sum: float = 0.0  # same denominator as recall: both are undefined on an empty set
    efficiency_sum: float = 0.0
    efficiency_n: int = 0
    jaccard_sum: float = 0.0
    exact: int = 0
    complete: int = 0  # sessions whose menu held every tool the session wanted
    removed_billed: float = 0.0  # 0.1 * tokens removed * turns, summed

    # The same two, over pruned sessions only. A declined session scores a free 1.0 on
    # both, so at a 36% routed share the blended figures are mostly measuring the
    # decline rule rather than the model.
    pruned_recall_sum: float = 0.0
    pruned_recall_n: int = 0
    pruned_complete: int = 0
    pruned: int = 0

    def mean(self, total: float, n: int) -> float:
        return total / n if n else 0.0


# --- Data -----------------------------------------------------------------------


def fetch_sessions(
    project: str, dataset: str, agent: str, dims: int, prefix: str, eval_only: bool
) -> list[Session]:
    sql = SESSIONS_SQL.format(
        project=project,
        dataset=dataset,
        agent=agent,
        rate=CACHED_RATE,
        eval_filter=(
            f"WHERE ABS(MOD(FARM_FINGERPRINT(f.session_id), 100)) < {EVAL_PCT}" if eval_only else ""
        ),
    )
    job = bigquery.Client(project=project).query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("agent", "STRING", agent),
                bigquery.ScalarQueryParameter("dims", "INT64", dims),
                bigquery.ScalarQueryParameter("prefix", "STRING", prefix),
            ]
        ),
    )
    return [
        Session(
            session_id=row["session_id"],
            vector=list(row["v"]),
            tools=frozenset(row["tool_set"]),
            turns=int(row["turns"]),
            billed=float(row["billed"]),
        )
        for row in job.result()
    ]


# --- The four ways of choosing a menu -------------------------------------------
#
# The first two are the shipped code, called the way the plugin calls it. The other two are
# oracles, which is why they are allowed to look at `session.tools`.


Menu = frozenset[str]


def menu_always(
    ranked: Sequence[PredictionLike],
    live: Menu,
    known: Menu,
    cfg: RouterConfig,
    pool: Sequence[Menu],
) -> Menu | None:
    """The menu the pruner picks for this session, before anything asks whether to send it.

    `prior=None` is not a simplification. The prefix lock (DESIGN.md 18.22) means a session
    that does not prune on turn one never prunes at all, so turn one with no prior is the
    only decision the router ever gets to make.
    """
    if not ranked:
        return None
    return decide(ranked, live, known, None, cfg, pool).kept


def menu_router(
    ranked: Sequence[PredictionLike],
    menu: Menu | None,
    live: Menu,
    tokens: dict[str, int],
    cfg: RouterConfig,
) -> Menu | None:
    """What the plugin really declares: `menu_always`, kept only when the prune pays.

    The rule itself is `pruner.worth_pruning`, called here rather than restated, so this arm
    cannot drift away from the one the router runs. The only difference is where the token
    counts come from. Live, the plugin measures the whole prompt through the tokenizer; here
    they are the per-tool declaration counts `measure_menus` already collected, which is the
    same subtraction over a corpus that has no prompts to measure.
    """
    if menu is None:
        return None
    saved = tok(live - menu, tokens)
    return menu if worth_pruning(saved, contained_mass(ranked, menu), cfg) else None


def menu_best(session: Session, pool: Sequence[Menu], live: Menu, protected: Menu) -> Menu | None:
    """The published menu that covers most of what the session truly wanted.

    Ties fall to the smaller menu and then to the alphabetically first, the same total order
    `menus.rank` settles on, so the arm is deterministic and does not quietly reward width.
    """
    if not pool:
        return None
    best = min(pool, key=lambda menu: (-len(menu & session.tools), len(menu), tuple(sorted(menu))))
    return (best & live) | (protected & live)


def menu_true(session: Session, live: Menu, protected: Menu) -> Menu:
    """Exactly what the session needed, plus what is never prunable. No pool, no model."""
    return (session.tools & live) | (protected & live)


async def build_reference_catalog_index(cfg: RouterConfig, agent: str = "support") -> CatalogIndex:
    """Extract `ToolSpec` metadata from the target agent and compile its `CatalogIndex`."""
    import inspect
    import warnings

    if agent == "xlam":
        from xlam_agent.agent import root_agent

        if not root_agent.tools:
            raise SystemExit(
                "examples/xlam_agent/tools.py is missing. "
                "Run `uv run python scripts/build_xlam_agent.py --fetch` first."
            )
    else:
        from reference_agent.agent import root_agent

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tools = await root_agent.canonical_tools()
        specs: dict[str, ToolSpec] = {}
        for tool in tools:
            decl = tool._get_declaration()  # noqa: SLF001
            if decl is None or not decl.name or decl.name in FRAMEWORK_TOOLS:
                continue
            func = getattr(tool, "func", None)
            raw_doc = getattr(func, "__doc__", None) or decl.description or ""
            desc = inspect.cleandoc(str(raw_doc)).strip()
            params: list[tuple[str, str]] = []
            if getattr(decl, "parameters_json_schema", None):
                props = decl.parameters_json_schema.get("properties", {})
                params = [(str(k), str(v.get("description", "") or "")) for k, v in props.items()]
            elif getattr(decl, "parameters", None) and decl.parameters.properties:
                params = [
                    (str(k), str(v.description or ""))
                    for k, v in decl.parameters.properties.items()
                ]
            specs[decl.name] = ToolSpec(name=decl.name, description=desc, parameters=tuple(params))

    embedder = Embedder(replace(cfg, embed_timeout_ms=5000))
    await embedder.warmup()
    names = tuple(sorted(specs.keys()))
    sem = asyncio.Semaphore(10)

    async def _embed_one(doc: str) -> list[float] | None:
        async with sem:
            for _ in range(3):
                vec = await embedder.embed(doc)
                if vec is not None:
                    return vec
            return None

    vectors = await asyncio.gather(*(_embed_one(specs[n].document()) for n in names))
    valid_vectors = [v for v in vectors if v is not None]
    if len(valid_vectors) != len(names):
        raise RuntimeError(f"Embedded {len(valid_vectors)} of {len(names)} tools")
    return build_catalog_index(names, valid_vectors, specs)


# --- Scoring --------------------------------------------------------------------


def score(
    arm: Arm, session: Session, menu: Menu | None, live: Menu, tokens: dict[str, int]
) -> None:
    """Fold one session into an arm, under the DESIGN.md 19.4 definitions."""
    arm.sessions += 1
    declined = menu is None
    if menu is None:
        arm.declined += 1
        menu = live  # a declined session ships the full list

    wanted = session.tools & live
    removed = live - menu
    removable = live - wanted
    recall = len(menu & wanted) / len(wanted) if wanted else None
    complete = wanted <= menu

    if recall is not None:
        arm.recall_sum += recall
        arm.recall_n += 1
        arm.fbeta_sum += fbeta(recall, len(menu & wanted) / len(menu) if menu else 0.0)
    arm.complete += complete
    arm.exact += (menu & live) == wanted

    if not declined:
        arm.pruned += 1
        arm.pruned_complete += complete
        if recall is not None:
            arm.pruned_recall_sum += recall
            arm.pruned_recall_n += 1

    union = menu | wanted
    arm.jaccard_sum += len(menu & wanted) / len(union) if union else 1.0

    removable_tokens = tok(removable, tokens)
    removed_tokens = tok(removed, tokens)
    if removable_tokens:
        arm.efficiency_sum += removed_tokens / removable_tokens
        arm.efficiency_n += 1

    arm.removed_billed += CACHED_RATE * removed_tokens * session.turns


def tok(tools: Menu, tokens: dict[str, int]) -> int:
    return sum(tokens.get(tool, 0) for tool in tools)


def fbeta(recall: float, precision: float, beta: int = BETA) -> float:
    """Recall and precision in one number, with recall worth `beta` squared times as much.

    At beta 8 this is recall with a small penalty for width, which is what DESIGN.md 19.1's
    75:1 cost ratio says the objective should be. It is here for comparison between arms
    rather than for gating: on a corpus where a session wants 1.1 tools and a menu carries
    seven, precision is near 0.15 for everybody and the ordering is recall's.
    """
    if recall <= 0.0 or precision <= 0.0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def report(arms: list[Arm], sessions: list[Session], toolless: int) -> None:
    billed = sum(s.billed for s in sessions)
    print(f"\n{len(sessions)} sessions, {toolless} of which called no tool at all.")
    print(f"{billed:,.0f} billed-equivalent tokens across them.\n")

    head = f"  {'arm':<22}{'routed':>8}{'recall':>9}{f'F{BETA}':>8}{'tok eff':>9}{'jaccard':>9}"
    print(head + f"{'complete':>10}{'exact':>8}{'pred G1':>10}")
    for arm in arms:
        routed = (arm.sessions - arm.declined) / arm.sessions if arm.sessions else 0.0
        print(
            f"  {arm.name:<22}{routed:>8.1%}"
            f"{arm.mean(arm.recall_sum, arm.recall_n):>9.4f}"
            f"{arm.mean(arm.fbeta_sum, arm.recall_n):>8.4f}"
            f"{arm.mean(arm.efficiency_sum, arm.efficiency_n):>9.4f}"
            f"{arm.mean(arm.jaccard_sum, arm.sessions):>9.4f}"
            f"{arm.complete / arm.sessions if arm.sessions else 0:>10.1%}"
            f"{arm.exact / arm.sessions if arm.sessions else 0:>8.1%}"
            f"{-arm.removed_billed / billed if billed else 0:>10.1%}"
        )

    print("\n  the same two, over pruned sessions only")
    for arm in arms:
        if not arm.pruned or arm.pruned == arm.sessions:
            continue
        print(
            f"  {arm.name:<22}{arm.pruned:>8}"
            f"{arm.mean(arm.pruned_recall_sum, arm.pruned_recall_n):>9.4f}"
            f"{'':>8}{'':>9}{'':>9}"
            f"{arm.pruned_complete / arm.pruned:>10.1%}"
        )

    print(
        "\n  routed    sessions the router pruned at all. The rest ship the full list.\n"
        "  recall    share of the tools a session wanted that its menu contained.\n"
        "            A declined session scores a free 1.0, hence the second table.\n"
        f"  F{BETA}        recall and precision together, recall worth {BETA**2} times as much.\n"
        "  tok eff   share of removable declaration tokens actually removed.\n"
        "  complete  sessions whose menu held every tool they wanted.\n"
        "  pred G1   DESIGN.md 19.5 criterion 3. Steady state only: turn-one residency\n"
        "            is a separate open question and is not in this number."
    )


def run_tier_comparison(
    sessions: list[Session],
    predictor: Predictor,
    known: Menu,
    pool: Sequence[Menu],
    live: Menu,
    protected: Menu,
    tokens: dict[str, int],
    cfg: RouterConfig,
    cost: int,
    turns: float,
    unseen_holdout: int,
    agent: str = "support",
) -> None:
    """Compare Baseline (full list), Cold-Start (prototype), Trained (xgboost), and Hybrid."""
    print("       compiling PrototypeJudge catalog index via gemini-embedding-2...")
    cat_index = asyncio.run(build_reference_catalog_index(cfg, agent=agent))
    cold_menus = build_catalog_menus(cat_index, size=15, count=24)
    priced_cfg = replace(cfg, miss_cost_tokens=cost, expected_session_turns=turns)
    cold_11_cfg = replace(priced_cfg, cold_start_max_tools_kept=11)
    cold_15_cfg = replace(priced_cfg, cold_start_max_tools_kept=15)

    arms = [
        Arm("full list (no router)"),
        Arm("cold-start (proto k=11)"),
        Arm("cold-start (proto k=15)"),
        Arm("cold-start (proto 24 menus)"),
        Arm(f"trained (xgb @{cost})"),
        Arm(f"hybrid (xgb+proto @{cost})"),
        Arm("best menu (oracle)"),
        Arm("true set (oracle)"),
    ]

    proto_cache: list[tuple[list[Prediction], list[Prediction], list[Prediction]]] = []
    for session in sessions:
        xgb_preds = list(predictor.predict(session.vector))
        proto_11 = score_catalog(session.vector, cat_index, max_tools_kept=11)
        proto_15 = score_catalog(session.vector, cat_index, max_tools_kept=15)
        proto_cache.append((xgb_preds, proto_11, proto_15))

        # 0: full list
        score(arms[0], session, live, live, tokens)

        # 1: cold-start k=11 (no trained model, no menus)
        c11_preds, c11_known, c11_cfg = combine_predictions(
            xgb_preds=None,
            xgb_known=frozenset(),
            aux_results=((1.0, proto_11),),
            live=live,
            cfg=cold_11_cfg,
        )
        m_c11 = menu_always(c11_preds, live, c11_known, c11_cfg, ())
        score(arms[1], session, menu_router(c11_preds, m_c11, live, tokens, c11_cfg), live, tokens)

        # 2: cold-start k=15 (shipped default bespoke cold-start cap)
        c15_preds, c15_known, c15_cfg = combine_predictions(
            xgb_preds=None,
            xgb_known=frozenset(),
            aux_results=((1.0, proto_15),),
            live=live,
            cfg=cold_15_cfg,
        )
        m_c15 = menu_always(c15_preds, live, c15_known, c15_cfg, ())
        score(arms[2], session, menu_router(c15_preds, m_c15, live, tokens, c15_cfg), live, tokens)

        # 3: cold-start 24 canonical menus (zero training data, shared prefix pool)
        m_c24 = menu_always(c15_preds, live, c15_known, c15_cfg, cold_menus)
        score(arms[3], session, menu_router(c15_preds, m_c24, live, tokens, c15_cfg), live, tokens)

        # 4: trained xgboost v1 alone
        m_xgb = menu_always(xgb_preds, live, known, priced_cfg, pool)
        score(
            arms[4],
            session,
            menu_router(xgb_preds, m_xgb, live, tokens, priced_cfg),
            live,
            tokens,
        )

        # 5: hybrid xgboost + prototype
        hyb_preds, hyb_known, hyb_cfg = combine_predictions(
            xgb_preds=xgb_preds,
            xgb_known=known,
            aux_results=((priced_cfg.prototype_weight, proto_15),),
            live=live,
            cfg=priced_cfg,
        )
        m_hyb = menu_always(hyb_preds, live, hyb_known, hyb_cfg, pool)
        score(arms[5], session, menu_router(hyb_preds, m_hyb, live, tokens, hyb_cfg), live, tokens)

        # 6 & 7: oracles
        score(arms[6], session, menu_best(session, pool, live, protected), live, tokens)
        score(arms[7], session, menu_true(session, live, protected), live, tokens)

    print(
        "\n=== TABLE 1: Standard 100-Tool Surface (Baseline vs Cold-Start vs Trained vs Hybrid) ==="
    )
    report(arms, sessions, sum(1 for s in sessions if not s.tools))

    if unseen_holdout <= 0:
        return

    called_sorted = sorted(known)
    held_out = frozenset(called_sorted[10 : 10 + unseen_holdout])
    reduced_known = known - held_out
    reduced_pool = tuple(frozenset(m - held_out) for m in pool if (m - held_out))
    unseen_all = (live - reduced_known) - FRAMEWORK_TOOLS

    stress_arms = [
        Arm("xgb (pass unseen)"),
        Arm("xgb (prune unseen)"),
        Arm("cold-start (proto k=15)"),
        Arm("hybrid (xgb+proto)"),
    ]
    prune_unseen_cfg = replace(priced_cfg, prunable_by_default=unseen_all)

    for session, (xgb_preds, _, proto_15) in zip(sessions, proto_cache, strict=True):
        red_xgb = [p for p in xgb_preds if p.label not in held_out]
        tot_r = sum(p.probability for p in red_xgb) or 1.0
        red_xgb = [Prediction(p.label, p.probability / tot_r) for p in red_xgb]

        # Arm 0: XGBoost default (passes all unseen tools through via `unknown`)
        m0 = menu_always(red_xgb, live, reduced_known, priced_cfg, reduced_pool)
        score(
            stress_arms[0],
            session,
            menu_router(red_xgb, m0, live, tokens, priced_cfg),
            live,
            tokens,
        )

        # Arm 1: XGBoost with prunable_by_default on unseen tools (blind to held-out tools)
        m1 = menu_always(red_xgb, live, reduced_known, prune_unseen_cfg, reduced_pool)
        score(
            stress_arms[1],
            session,
            menu_router(red_xgb, m1, live, tokens, prune_unseen_cfg),
            live,
            tokens,
        )

        # Arm 2: Cold-Start Prototype k=15
        c15_preds, c15_known, c15_cfg = combine_predictions(
            xgb_preds=None,
            xgb_known=frozenset(),
            aux_results=((1.0, proto_15),),
            live=live,
            cfg=cold_15_cfg,
        )
        m2 = menu_always(c15_preds, live, c15_known, c15_cfg, ())
        score(
            stress_arms[2],
            session,
            menu_router(c15_preds, m2, live, tokens, c15_cfg),
            live,
            tokens,
        )

        # Arm 3: Hybrid XGBoost + Prototype
        hyb_preds, hyb_known, hyb_cfg = combine_predictions(
            xgb_preds=red_xgb,
            xgb_known=reduced_known,
            aux_results=((priced_cfg.prototype_weight, proto_15),),
            live=live,
            cfg=priced_cfg,
        )
        m3 = menu_always(hyb_preds, live, hyb_known, hyb_cfg, reduced_pool)
        score(
            stress_arms[3],
            session,
            menu_router(hyb_preds, m3, live, tokens, hyb_cfg),
            live,
            tokens,
        )

    print(
        f"\n=== TABLE 2: Newly Deployed / Unseen Tool Stress Test "
        f"({len(held_out)} Called Tools Held Out from XGBoost, {len(unseen_all)} Total Unseen) ==="
    )
    report(stress_arms, sessions, sum(1 for s in sessions if not s.tools))


# --- Entry point ----------------------------------------------------------------


def main() -> int:
    args = parse_args()
    cfg = load_config(
        bq_project=args.project,
        bq_dataset=args.dataset,
        gcs_bucket=args.bucket,
        pin_version=args.pin_version,
    )

    model = ModelStore(cfg).load(args.agent)
    if model is None:
        print("ModelStore refused to load an artifact; see the warning above.", file=sys.stderr)
        return 1
    pool = model.menus
    print(f"loaded {args.agent} version {model.metadata.version}, {len(model.classes)} classes")
    print(f"       {len(pool)} menus in the published pool")

    tokens = asyncio.run(declaration_tokens(agent=args.agent))
    live = frozenset(tokens)
    protected = (FRAMEWORK_TOOLS | cfg.never_prune) & live
    print(f"       {len(live)} live tools, {sum(tokens.values())} declaration tokens")

    sessions = fetch_sessions(
        args.project,
        args.dataset,
        args.agent,
        cfg.embedding_dims,
        args.session_prefix,
        args.eval_only,
    )
    if not sessions:
        print(f"\nno sessions under prefix {args.session_prefix!r}.", file=sys.stderr)
        return 1

    predictor = Predictor(model)
    known = frozenset(model.classes)
    turns = sum(s.turns for s in sessions) / len(sessions)
    print(f"       {len(sessions)} sessions, {turns:.2f} model calls each on average")

    if args.compare_tiers:
        cost = args.miss_cost if args.miss_cost is not None else cfg.miss_cost_tokens
        run_tier_comparison(
            sessions=sessions,
            predictor=predictor,
            known=known,
            pool=pool,
            live=live,
            protected=protected,
            tokens=tokens,
            cfg=cfg,
            cost=cost,
            turns=turns,
            unseen_holdout=args.unseen_holdout,
            agent=args.agent,
        )
        return 0

    # One router arm per price, because the price is the knob and a single point hides the
    # frontier it sits on. Recall buys saving and saving buys recall back.
    #
    # The corpus's own measured session length replaces the config's prior, because here it
    # is not a prior: these sessions have already finished and their turns are counted.
    costs = args.sweep or [args.miss_cost or cfg.miss_cost_tokens]
    print(f"       a miss priced at {costs} billed tokens (DESIGN.md 19.1)")
    priced_cfgs = [
        replace(cfg, miss_cost_tokens=cost, expected_session_turns=turns) for cost in costs
    ]
    fixed = [Arm("full list"), Arm("always prune")]
    priced = [Arm(f"router @{cost}") for cost in costs]
    oracles = [Arm("best menu"), Arm("true set")]

    for session in sessions:
        ranked = predictor.predict(session.vector)
        menu = menu_always(ranked, live, known, cfg, pool)
        score(fixed[0], session, live, live, tokens)
        score(fixed[1], session, menu, live, tokens)
        for arm, priced_cfg in zip(priced, priced_cfgs, strict=True):
            score(arm, session, menu_router(ranked, menu, live, tokens, priced_cfg), live, tokens)
        score(oracles[0], session, menu_best(session, pool, live, protected), live, tokens)
        score(oracles[1], session, menu_true(session, live, protected), live, tokens)

    report([*fixed, *priced, *oracles], sessions, sum(1 for s in sessions if not s.tools))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--bucket", **bucket_arg())
    parser.add_argument(
        "--pin-version",
        default=None,
        help="optional artifact version to pin (e.g. 20260906T203310Z vs 20260906T203310Z-nomenus)",
    )
    parser.add_argument(
        "--session-prefix",
        default="corpus",
        help="only score sessions whose id starts with this. Must be UNROUTED traffic.",
    )
    parser.add_argument(
        "--miss-cost",
        type=int,
        default=None,
        help="billed tokens a missing tool costs. Defaults to the shipped miss_cost_tokens.",
    )
    parser.add_argument(
        "--sweep",
        type=int,
        nargs="+",
        default=None,
        metavar="TOKENS",
        help="score one router arm per miss price, to see the recall/saving frontier",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="restrict to the training split's held-out sessions (DESIGN.md 9.4)",
    )
    parser.add_argument(
        "--compare-tiers",
        action="store_true",
        help="compare full list vs cold-start (PrototypeJudge) vs trained (XGBoost) vs hybrid",
    )
    parser.add_argument(
        "--unseen-holdout",
        type=int,
        default=15,
        help="number of called tools to hold out in --compare-tiers Table 2 (0 to skip)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

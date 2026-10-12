#!/usr/bin/env python
"""The whole-system ledger for an A/B run, because G1 divides two costs out of view.

G1 is billed input tokens per turn, and `report_ab.py` explains why each word is there.
Its per-turn denominator rests on one assumption, stated in that file: "the router
changes nothing about how long a conversation runs." That is an assumption and not a
measurement, and if it is wrong the gate hides the damage. An agent that loses a tool it
needed may ask a clarifying question, or try a neighbouring tool and try again, and both
add a turn. Per-turn cost can fall while the session gets dearer.

There is a second channel G1 cannot see, running the other way. A tool response is an
input on every later turn of the session, so a wrong call is not a free mistake: its
payload joins the prompt and is re-sent for the rest of the conversation. A menu that
leaves a wrong tool out saves the whole response, which is a saving the declaration
arithmetic never counts.

So this script prices a session rather than a turn, in five parts:

    billed input      uncached + 0.10 x cached, the same unit G1 uses
    output            completion tokens, which the gate leaves out entirely
    turns             reported, with a significance test, rather than divided away
    tool calls        how many, per session
    tool responses    how many characters they returned, the second channel above

It reproduces G1 as one of its rows, deliberately. If the per-turn number here does not
match `report_ab.py` on the same prefix, one of the two is wrong and this is the cheaper
place to find out.

**The result on the reference agent, and its limit.** On `ab-v2-100tools`, whole-system
cost per session falls 22.5% against G1's 23.4%, and the gap is the turn count: the
routed arm ran 4.670 turns a session against 4.637, which is not a significant difference
(p = 0.74) but is not zero either. Output is 1.4% of the bill and moves with it. The tool
response channel measures nothing, and cannot: this agent's tools return a median of 95
characters, so there is no payload to save. An agent whose tools return documents or
database rows would see that channel dominate, which is the argument for measuring it per
adopter rather than quoting our number. `--response-chars` prices that counterfactual.

Cost is two BigQuery reads. Authentication is ADC only.

Usage:
    uv run python scripts/report_ledger.py --prefix ab-v2-100tools
    uv run python scripts/report_ledger.py --prefix ab-v2-100tools --response-chars 5000
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

from deployment import DEFAULT_DATASET, project_arg  # noqa: E402
from google.cloud import bigquery

# Same defaults as everywhere else in the project (DESIGN.md 3.1).
DEFAULT_AGENT = "support"

# What a cached input token costs relative to an uncached one, on Gemini 2.5 Flash. The
# same constant `report_ab.py` defines the gate in, and it is a gate constant rather than
# a display preference (DESIGN.md 10).
CACHED_TOKEN_WEIGHT = 0.10

# Decision sources that mean the router acted. Everything else is a recorded decision not
# to, and a session containing one is neither arm. Kept identical to `report_ab.py`.
ROUTED_SOURCES = ("fresh", "reused")

# Characters per token for tool payloads, used only by the `--response-chars`
# counterfactual. JSON of the shape these tools return runs denser than prose; 4.0 is the
# conventional rough figure and the counterfactual is an order-of-magnitude argument, not
# a bill.
CHARS_PER_TOKEN = 4.0

# Above this median, the tool-response channel is a measurement rather than a
# counterfactual, and the script reads the arms' own payloads instead of an assumed size.
# 500 characters sits well above the ordinary reference agent, whose median is 95, and
# well below the heavy variant of ROADMAP's E2, whose median is around 1,500.
MEASURED_RESPONSE_CHARS = 500


# --- The arithmetic, which needs no warehouse -------------------------------------


@dataclass(frozen=True)
class Arm:
    """One arm's totals. Sums come from SQL; every rate is derived here."""

    name: str
    sessions: int
    turns: int
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    calls: int
    result_chars: int
    mean_turns: float
    sd_turns: float
    mean_chars: float
    sd_chars: float

    @property
    def billed_input(self) -> float:
        uncached = max(self.prompt_tokens - self.cached_tokens, 0)
        return uncached + CACHED_TOKEN_WEIGHT * self.cached_tokens

    @property
    def billed_per_turn(self) -> float:
        """G1's unit, reproduced here so the two scripts can be checked against each other."""
        return self.billed_input / self.turns if self.turns else 0.0

    @property
    def input_per_session(self) -> float:
        return self.billed_input / self.sessions if self.sessions else 0.0

    @property
    def output_per_session(self) -> float:
        return self.completion_tokens / self.sessions if self.sessions else 0.0

    @property
    def whole_per_session(self) -> float:
        """Everything the session bills, input and output. The number this script exists for."""
        return self.input_per_session + self.output_per_session

    @property
    def calls_per_session(self) -> float:
        return self.calls / self.sessions if self.sessions else 0.0

    @property
    def result_chars_per_session(self) -> float:
        return self.result_chars / self.sessions if self.sessions else 0.0


def change(routed: float, holdout: float) -> float | None:
    """Routed against holdout, as a fraction. None when there is nothing to divide by."""
    return (routed - holdout) / holdout if holdout else None


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def welch(
    routed: tuple[float, float, int], holdout: tuple[float, float, int]
) -> tuple[float, float] | None:
    """Welch's t on a per-session quantity, given (mean, sd, sessions) for each arm.

    Returned as (t, two-sided p), with a normal approximation for the p-value. That is
    honest at these sample sizes: both arms carry a few hundred sessions, where the t
    distribution and the normal agree to the third decimal.

    It takes triples rather than two Arms because two different per-session quantities need
    it. Turns, so that "the router does not lengthen conversations" is a claim with a number
    under it rather than the assumption `report_ab.py` has to make to divide by turns. And
    response characters, so that a shift in the tool-response channel can be told apart from
    the noise of a few hundred sessions before anybody writes it down.
    """
    routed_mean, routed_sd, routed_n = routed
    holdout_mean, holdout_sd, holdout_n = holdout
    if routed_n < 2 or holdout_n < 2:
        return None
    variance = routed_sd**2 / routed_n + holdout_sd**2 / holdout_n
    if variance <= 0.0:
        return None
    t = (routed_mean - holdout_mean) / math.sqrt(variance)
    return t, 2.0 * (1.0 - normal_cdf(abs(t)))


def turns_test(routed: Arm, holdout: Arm) -> tuple[float, float] | None:
    return welch(
        (routed.mean_turns, routed.sd_turns, routed.sessions),
        (holdout.mean_turns, holdout.sd_turns, holdout.sessions),
    )


def chars_test(routed: Arm, holdout: Arm) -> tuple[float, float] | None:
    return welch(
        (routed.mean_chars, routed.sd_chars, routed.sessions),
        (holdout.mean_chars, holdout.sd_chars, holdout.sessions),
    )


def response_leverage(routed: Arm, holdout: Arm, chars: float) -> float | None:
    """What the tool-response channel would be worth if responses were `chars` long.

    The counterfactual an adopter needs and our corpus cannot supply. The router's only
    reach into this channel is the calls it prevents, so the quantity is the difference in
    calls per session, priced at the response size and re-sent for the rest of the session.
    A call on turn `k` of an `n`-turn session is carried by the `n - k` turns after it;
    averaged over a uniform arrival, that is half the session.

    Returns the change it would make to whole-system cost per session, as a fraction.
    Positive means the router's own call difference is costing tokens at that response
    size, negative means it is saving them.
    """
    if chars <= 0 or holdout.whole_per_session <= 0:
        return None
    extra_calls = routed.calls_per_session - holdout.calls_per_session
    carried_turns = holdout.mean_turns / 2.0
    tokens = extra_calls * (chars / CHARS_PER_TOKEN) * carried_turns
    # Priced at the cached rate: a tool response lands in the prefix and is cached on the
    # turns that follow it, which is the same discount the declarations get.
    return tokens * CACHED_TOKEN_WEIGHT / holdout.whole_per_session


def measured_response_share(routed: Arm, holdout: Arm) -> float | None:
    """What the arms' own response-character difference is worth, as a share of the bill.

    The same arithmetic as `response_leverage` with the run's measured payload size in
    place of an assumed one, and with the difference taken in characters rather than in
    calls. Characters, because once payloads vary in size the router can move this channel
    without moving the call count, by keeping a terse tool and dropping a wordy one.

    This one is a decomposition, not an addition. The characters are already inside the
    billed input rows above; this says how much of the difference between the arms they
    account for.
    """
    if holdout.whole_per_session <= 0:
        return None
    extra_chars = routed.result_chars_per_session - holdout.result_chars_per_session
    carried_turns = holdout.mean_turns / 2.0
    tokens = (extra_chars / CHARS_PER_TOKEN) * carried_turns
    return tokens * CACHED_TOKEN_WEIGHT / holdout.whole_per_session


# --- The warehouse ----------------------------------------------------------------

# One row per arm. Sessions are labelled from the router's own provenance, the same way
# `report_ab.py` labels them, so the two scripts cannot disagree about who was in which
# arm. Turns and tool calls are counted separately and joined per session rather than in
# one pass, because a session with no tool call still has turns and a fan-out join would
# multiply its tokens by its calls.
_LEDGER_SQL = """
WITH decisions AS (
  SELECT
    e.session_id,
    JSON_VALUE(record, '$.decision_source') AS source
  FROM `{project}.{dataset}.agent_events` AS e,
    UNNEST(JSON_QUERY_ARRAY(e.attributes, '$.custom_metadata.smart_router')) AS record
  WHERE e.agent = @agent
    AND STARTS_WITH(e.session_id, @prefix)
    AND JSON_VALUE(record, '$.tool_name') IS NULL
),
arms AS (
  SELECT
    session_id,
    CASE
      WHEN LOGICAL_OR(source = 'holdout') THEN 'holdout'
      WHEN LOGICAL_OR(source IN UNNEST(@routed_sources)) THEN 'routed'
      ELSE 'neither'
    END AS arm
  FROM decisions
  GROUP BY session_id
),
turns AS (
  SELECT
    session_id,
    COUNT(*) AS turns,
    SUM(COALESCE(usage_prompt_tokens, 0)) AS prompt_tokens,
    SUM(COALESCE(usage_cached_tokens, 0)) AS cached_tokens,
    SUM(COALESCE(usage_completion_tokens, 0)) AS completion_tokens
  FROM `{project}.{dataset}.v_llm_response`
  WHERE agent = @agent
    AND STARTS_WITH(session_id, @prefix)
    AND usage_prompt_tokens IS NOT NULL
  GROUP BY session_id
),
calls AS (
  -- `transfer_to_agent` is excluded for the reason dashboard.sql gives: it is ADK's own
  -- routing call, not one of the agent's tools, and the router never sees it.
  SELECT
    session_id,
    COUNT(*) AS calls,
    SUM(LENGTH(IFNULL(TO_JSON_STRING(JSON_QUERY(content, '$.result')), ''))) AS result_chars
  FROM `{project}.{dataset}.agent_events`
  WHERE agent = @agent
    AND STARTS_WITH(session_id, @prefix)
    AND event_type = 'TOOL_COMPLETED'
    AND JSON_VALUE(content, '$.tool') != 'transfer_to_agent'
  GROUP BY session_id
),
per_session AS (
  SELECT
    arms.arm,
    turns.turns,
    turns.prompt_tokens,
    turns.cached_tokens,
    turns.completion_tokens,
    IFNULL(calls.calls, 0) AS calls,
    IFNULL(calls.result_chars, 0) AS result_chars
  FROM arms
  JOIN turns USING (session_id)
  LEFT JOIN calls USING (session_id)
)
SELECT
  arm AS name,
  COUNT(*) AS sessions,
  SUM(turns) AS turns,
  SUM(prompt_tokens) AS prompt_tokens,
  SUM(cached_tokens) AS cached_tokens,
  SUM(completion_tokens) AS completion_tokens,
  SUM(calls) AS calls,
  SUM(result_chars) AS result_chars,
  AVG(turns) AS mean_turns,
  IFNULL(STDDEV(turns), 0.0) AS sd_turns,
  AVG(result_chars) AS mean_chars,
  IFNULL(STDDEV(result_chars), 0.0) AS sd_chars
FROM per_session
GROUP BY name
ORDER BY name
"""

# The response-size distribution, which decides whether the second channel is worth
# anything on this agent. A mean alone would hide a long tail, and a long tail is exactly
# the case where preventing one wrong call pays for the whole feature.
_RESPONSES_SQL = """
SELECT
  COUNT(*) AS calls,
  IFNULL(AVG(LENGTH(IFNULL(TO_JSON_STRING(JSON_QUERY(content, '$.result')), ''))), 0.0) AS mean,
  IFNULL(APPROX_QUANTILES(
    LENGTH(IFNULL(TO_JSON_STRING(JSON_QUERY(content, '$.result')), '')), 100
  )[OFFSET(50)], 0) AS p50,
  IFNULL(APPROX_QUANTILES(
    LENGTH(IFNULL(TO_JSON_STRING(JSON_QUERY(content, '$.result')), '')), 100
  )[OFFSET(95)], 0) AS p95,
  IFNULL(MAX(LENGTH(IFNULL(TO_JSON_STRING(JSON_QUERY(content, '$.result')), ''))), 0) AS max
FROM `{project}.{dataset}.agent_events`
WHERE agent = @agent
  AND STARTS_WITH(session_id, @prefix)
  AND event_type = 'TOOL_COMPLETED'
  AND JSON_VALUE(content, '$.tool') != 'transfer_to_agent'
"""


def _params(agent: str, prefix: str, routed: bool) -> list[bigquery.ScalarQueryParameter]:
    params: list[bigquery.ScalarQueryParameter] = [
        bigquery.ScalarQueryParameter("agent", "STRING", agent),
        bigquery.ScalarQueryParameter("prefix", "STRING", prefix),
    ]
    if routed:
        params.append(
            bigquery.ArrayQueryParameter("routed_sources", "STRING", list(ROUTED_SOURCES))  # type: ignore[arg-type]
        )
    return params


def fetch_arms(client: bigquery.Client, dataset: str, agent: str, prefix: str) -> list[Arm]:
    sql = _LEDGER_SQL.format(project=client.project, dataset=dataset)
    config = bigquery.QueryJobConfig(query_parameters=_params(agent, prefix, routed=True))
    return [
        Arm(
            name=str(row["name"]),
            sessions=int(row["sessions"]),
            turns=int(row["turns"]),
            prompt_tokens=int(row["prompt_tokens"]),
            cached_tokens=int(row["cached_tokens"]),
            completion_tokens=int(row["completion_tokens"]),
            calls=int(row["calls"]),
            result_chars=int(row["result_chars"]),
            mean_turns=float(row["mean_turns"]),
            sd_turns=float(row["sd_turns"]),
            mean_chars=float(row["mean_chars"]),
            sd_chars=float(row["sd_chars"]),
        )
        for row in client.query(sql, job_config=config).result()
    ]


def fetch_responses(
    client: bigquery.Client, dataset: str, agent: str, prefix: str
) -> dict[str, float]:
    sql = _RESPONSES_SQL.format(project=client.project, dataset=dataset)
    config = bigquery.QueryJobConfig(query_parameters=_params(agent, prefix, routed=False))
    row = next(iter(client.query(sql, job_config=config).result()))
    return {key: float(row[key]) for key in ("calls", "mean", "p50", "p95", "max")}


# --- What it prints ---------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "     —" if value is None else f"{value:+6.1%}"


def _line(label: str, routed: float, holdout: float, unit: str = "", places: int = 1) -> str:
    """`places` exists because turns per session differ in the third decimal.

    Printing 4.6 against 4.7 makes a difference of 0.033 look like one of 0.1, which is
    the whole quantity in dispute on the row it matters most on.
    """
    return (
        f"  {label:<28}{holdout:>12,.{places}f}{routed:>12,.{places}f}"
        f"{_pct(change(routed, holdout)):>10}  {unit}"
    )


def report(arms: list[Arm], responses: dict[str, float], response_chars: float) -> int:
    by_name = {arm.name: arm for arm in arms}
    routed, holdout = by_name.get("routed"), by_name.get("holdout")
    if routed is None or holdout is None:
        print("Both arms are needed and at least one is missing. Arms found: ", end="")
        print(", ".join(sorted(by_name)) or "none")
        return 1

    print(f"\nwhole-system ledger — {routed.sessions:,} routed, {holdout.sessions:,} holdout\n")
    print(f"  {'':<28}{'holdout':>12}{'routed':>12}{'change':>10}")
    print(_line("billed input / turn", routed.billed_per_turn, holdout.billed_per_turn, "G1"))
    print(_line("billed input / session", routed.input_per_session, holdout.input_per_session))
    print(_line("output / session", routed.output_per_session, holdout.output_per_session))
    print(
        _line(
            "whole system / session",
            routed.whole_per_session,
            holdout.whole_per_session,
            "the honest number",
        )
    )
    print()
    print(_line("turns / session", routed.mean_turns, holdout.mean_turns, places=3))
    print(
        _line(
            "tool calls / session",
            routed.calls_per_session,
            holdout.calls_per_session,
            places=3,
        )
    )
    print(
        _line(
            "response chars / session",
            routed.result_chars_per_session,
            holdout.result_chars_per_session,
        )
    )

    print("\n  does the router lengthen conversations?")
    test = turns_test(routed, holdout)
    if test is None:
        print("    not enough sessions to say.")
    else:
        t, p = test
        answer = "no measurable difference" if p >= 0.05 else "yes, and G1 hides it"
        print(f"    Welch t = {t:+.2f}, two-sided p = {p:.3f} — {answer}.")
        gap = change(routed.whole_per_session, holdout.whole_per_session)
        g1 = change(routed.billed_per_turn, holdout.billed_per_turn)
        if gap is not None and g1 is not None:
            print(
                f"    G1 reads {g1:+.1%} and the whole system {gap:+.1%}. The difference is "
                f"what per-turn\n    normalising divides out, and it is {abs(gap - g1):.1%}."
            )

    print("\n  the tool-response channel")
    if responses["calls"] == 0:
        print("    no tool calls in this run, so there is nothing to price.")
    else:
        print(
            f"    {responses['calls']:,.0f} calls returned a median of {responses['p50']:,.0f} "
            f"characters, p95 {responses['p95']:,.0f}, max {responses['max']:,.0f}."
        )
        if responses["p50"] >= MEASURED_RESPONSE_CHARS:
            share = measured_response_share(routed, holdout)
            moved = routed.result_chars_per_session - holdout.result_chars_per_session
            if share is None:
                print("    one arm billed nothing, so there is nothing to measure against.")
            else:
                print(
                    f"    Payloads here are large enough to read rather than price. The routed "
                    f"arm\n    carried {moved:+,.0f} characters of tool response a session, worth "
                    f"{share:+.2%} of the\n    whole-system bill. That is a share of the numbers "
                    f"above, not an addition to them."
                )
            # Response size per session is a heavier-tailed quantity than turns: one call
            # into a document-returning tool can outweigh a hundred ordinary ones. A gap of
            # several percent between the arms is well inside what that tail produces by
            # chance, so the difference gets a test before anybody reads a mechanism into it.
            chars = chars_test(routed, holdout)
            if chars is not None:
                t, p = chars
                verdict = (
                    "inside the noise of a run this size"
                    if p >= 0.05
                    else "a real shift in which tools get called"
                )
                print(f"    Welch t = {t:+.2f}, two-sided p = {p:.3f} — {verdict}.")
            print()
            return 0
        leverage = response_leverage(routed, holdout, response_chars)
        if leverage is None:
            print("    nothing to price at that response size.")
        else:
            extra = routed.calls_per_session - holdout.calls_per_session
            print(
                f"    At {response_chars:,.0f} characters a response, the {extra:+.3f} calls a "
                f"session the router\n    makes would be worth {leverage:+.2%} of the "
                f"whole-system bill. Small response payloads\n    are why this channel is "
                f"invisible here; an adopter whose tools return documents\n    should re-run "
                f"this with their own number."
            )
    print()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefix", required=True, help="session id prefix of the run")
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--response-chars",
        type=float,
        default=5_000.0,
        help="price the response channel as if tools returned this many characters",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = bigquery.Client(project=args.project)
    arms = fetch_arms(client, args.dataset, args.agent, args.prefix)
    responses = fetch_responses(client, args.dataset, args.agent, args.prefix)
    return report(arms, responses, args.response_chars)


if __name__ == "__main__":
    sys.exit(main())

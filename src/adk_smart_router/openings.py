"""Which tool each session called first, read from the agent's own analytics log.

`reachability` decides two things from schemas and asks a person for the rest. This is the
third source, and the only one that cannot be wrong about the user: if a session's first tool
call was `get_order(order_id=...)`, then before any tool had returned anything the model had
an order number, and the only place it can have come from is the person typing. A marking file
says the same thing as a belief. Traffic says it as a demonstration.

First call of the **session**, not of the first turn. A conversation can run three turns of
pleasantries before it calls anything, and until some tool has returned a value, every argument
in the request still came out of the conversation. That is a wider net than turn one and it
demonstrates exactly the same thing.

The soft edge is a failed call earlier in the session, because an error message can carry a
value back into the context and the next call could pick it up. So a session whose earliest
call of any kind failed is dropped rather than counted. It costs a few demonstrations and buys
not having to reason about it.

Cache hits are excluded in the view's spelling of the marker (DESIGN.md 9.1). A hit is the
router answering from its own memo rather than the tool running, and counting one as a first
call would let the router vouch for its own prediction.

This is the one module in the package that talks to BigQuery, and it does it behind a lazy
import so that installing the plugin does not drag a query client into an agent's serving path.
Nothing here runs at request time; it is the `smart-router check-arguments` command's data.
"""

from __future__ import annotations

from typing import Any

__all__ = ["FIRST_CALL_SQL", "first_calls"]

# The slice of google-cloud-bigquery this touches is one method. Aliasing it the way
# `model_store` aliases the GCS types keeps the client out of this module's signatures, so a
# test can pass a fake and nothing here imports BigQuery at import time.
BigQueryClient = Any

# `{project}` and `{dataset}` are formatted in because BigQuery has no parameter for a table
# name. `@agent` and `@since` are real parameters, which is what keeps a caller's strings out
# of the SQL. Same shape as scripts/publish_router.py and src/adk_smart_router/sql/preflight.sql.
FIRST_CALL_SQL = """
WITH calls AS (
  -- SOURCE: v_tool_completed (a VIEW). The cache-hit marker is read from `tool_result`,
  -- the view's spelling (DESIGN 9.1). The base-table spelling would match nothing here.
  SELECT session_id, timestamp, tool_name, TRUE AS ok
  FROM `{project}.{dataset}.v_tool_completed`
  WHERE agent = @agent
    AND timestamp >= TIMESTAMP(@since)
    AND status = 'OK'
    -- ADK control plane, never a routed tool (DESIGN 7.2).
    AND tool_name != 'transfer_to_agent'
    AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NULL
  UNION ALL
  -- Failures are here to disqualify the session, not to be counted. An error can put a value
  -- into the context, and then the next call is no longer a demonstration of anything.
  SELECT session_id, timestamp, tool_name, FALSE AS ok
  FROM `{project}.{dataset}.v_tool_error`
  WHERE agent = @agent
    AND timestamp >= TIMESTAMP(@since)
    AND tool_name != 'transfer_to_agent'
),
opened AS (
  SELECT
    session_id,
    ARRAY_AGG(STRUCT(tool_name, ok) ORDER BY timestamp LIMIT 1)[OFFSET(0)] AS first_call
  FROM calls
  GROUP BY session_id
)
SELECT first_call.tool_name AS tool_name, COUNT(*) AS sessions
FROM opened
WHERE first_call.ok
GROUP BY tool_name
ORDER BY sessions DESC, tool_name
"""


def first_calls(
    project: str,
    dataset: str,
    agent: str,
    since: str,
    client: BigQueryClient = None,
) -> dict[str, int]:
    """Tool name to the number of sessions whose first tool call it was.

    `since` is anything BigQuery's `TIMESTAMP()` accepts, so `2026-01-01` is enough. Pass
    `client` to reuse a `bigquery.Client`; the default builds one from Application Default
    Credentials, which is the only way this project authenticates.
    """
    # Imported here so that `import adk_smart_router` costs an agent nothing at serving time.
    from google.cloud import bigquery

    reader = client or bigquery.Client(project=project)
    job = reader.query(
        FIRST_CALL_SQL.format(project=project, dataset=dataset),
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("agent", "STRING", agent),
                bigquery.ScalarQueryParameter("since", "STRING", since),
            ]
        ),
    )
    return {row["tool_name"]: int(row["sessions"]) for row in job.result()}

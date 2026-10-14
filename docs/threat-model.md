# Threat model

The router reads untrusted user text and uses it to decide which tools an LLM is shown. That is
enough to deserve a written threat model, and this is it.

The short version: an attacker who fully controls the first user message can choose which subset
of your tools that session sees, and can do nothing else. An attacker who can write the artifact
bucket can make that choice for every session of that agent instead of one, and still nothing
else. Everything below is why both of those are true, which parts are enforced in code, and the
handful of things an operator has to get right.

## The property everything rests on

**Pruning is subset-only.** The router can remove a tool from what the model is shown. It cannot
add one, rename one, change a schema, or invent a tool the host never declared. In
`pruner.decide` the returned set is

```python
kept = (chosen & live) | unknown | (protected & live)
```

where `live` is the host's own declared set and `unknown` is a subset of it. Every term is
intersected with `live`, so `kept ⊆ live` by construction rather than by discipline.
`tests/test_pruner.py` pins it: a randomised walk over predictions, menus, priors and settings
asserts `decision.kept <= live` on every decision it produces.

That single property is what bounds the whole attack surface. Whatever a message talks the
classifier into, the resulting menu is a subset of the capability the agent already had.

## Steering the classifier

An attacker who writes the first message controls the embedding, and therefore controls which
menu the session gets. There are only two things worth trying.

**Force a tool in.** Write a message that keeps a tool the attacker wants to reach. This gains
nothing. The tool was declared by the host, so the model could already call it, and the router
declining to prune it just returns the agent to how it behaves with the plugin uninstalled. The
router is not what stands between a user and a tool.

**Force a tool out.** Write a message that lands the session on a menu missing the tool the
session actually needs. This works, and it is the real capability an attacker has. The menu is
chosen once and locked for the session, so the effect lasts the whole conversation. The impact is
denial of function in one session, and it is the attacker's own session: menus are chosen per
session from that session's own first message, and nothing about one session's choice reaches
another. An attacker degrading their own conversation is not an interesting attack, which is why
this is bounded rather than mitigated.

The same effect happens by accident, with no attacker, whenever a first message is off
distribution. That is the ordinary miss rate, it is measured, and it is priced by
`miss_cost_tokens` before the prune happens.

**The exception is `never_prune`.** If a tool must be reachable in every session no matter what
the user says, no prediction may be allowed to remove it. An escalation path, a refusal tool, an
audit or logging tool, a safety handoff. Put those in `never_prune` and the classifier's opinion
about them stops mattering. This is the one place where a prompt-injection concern turns into an
operator action, and it is the first item in the checklist at the bottom.

## Cold-start, hybrid, and external judges

When `PrototypeJudge` (`"prototype"`), `LiveLlmJudge` (`"llm_live"`), or an external `ToolJudge`
endpoint is enabled, two additional inputs enter the scoring path: the live tool catalog's
descriptions/parameter schemas (`ToolSpec`) and, in hybrid mode, the set of unseen/classless tools
(`live - xgb_known`).

**Every judge is bounded by the same subset property.** Whether predictions come from `xgboost`,
`PrototypeJudge`, `LiveLlmJudge`, or a custom `ToolJudge`, they all emit `Sequence[Prediction]` and
pass through `pruner.decide`. A judge that hallucinates a nonexistent tool name or assigns probability
`1.0` to every tool cannot add a tool outside `live` or drop a tool in `never_prune`.

**Hybrid mode (`judges=("xgboost", "prototype")`) prunes unactivated unseen tools.** In pure
`xgboost` mode (`judges=("xgboost",)`), any tool absent from `metadata.json`'s `classes` is kept in
`unknown` on every menu until the next retrain. In hybrid mode, `combine_predictions` uses
`PrototypeJudge` to score unseen tools semantically: up to `_RESCUE_MAX_TOOLS` (3) unseen tools that
rank in the top `_RESCUE_MAX_RANK` (6) are rescued onto the kept menu, while unactivated unseen
tools are marked known and pruned. Do not rely on "untrained means unpruned" to keep a newly added
safety tool alive in hybrid mode—list safety-critical tools explicitly in `never_prune`.

**Third-party MCP tool descriptions and `PrototypeJudge` / `LiveLlmJudge` bounds.**
- **Dynamic docstring churn:** If a remote MCP server injects per-turn dynamic text into tool
  descriptions, `PrototypeJudge` bounds its canonical menu cache to a single entry per `agent_name`
  and caps description-only index rebuilds at `_MAX_INDEX_REBUILDS_PER_AGENT` (4) so dynamic
  docstrings cannot leak memory or trigger unbounded Gemini Enterprise Embedding API calls.
- **Instruction/data separation in `LiveLlmJudge`:** When `llm_live` is enabled, the catalog and
  routing instructions are passed via `GenerateContentConfig(system_instruction=...)`, tool
  descriptions are truncated to `_LLM_JUDGE_MAX_DESC_CHARS` (240 chars), the untrusted user prompt
  is passed as `contents` capped at `_LLM_JUDGE_MAX_PROMPT_CHARS` (4,000 chars), and returned JSON
  tool names are intersected with `prunable_specs`.

## The tool result cache

This is the sharper half of the plugin, and it is treated that way in the code.
`tool_cache.py`'s module docstring calls itself the security surface of the router.

**Identity is always in the key.** `cache_key` takes `user_id` unconditionally. There is no
opt-out, no global scope, and an empty principal fails closed rather than collapsing every caller
into one bucket. Most real tools are principal-scoped and take arguments that never mention the
principal, so `get_balance()` returns a different answer for different people and a key without
identity would serve one of them to the other. The hash is built with length-prefixed components
so no combination of argument values can forge another principal's key.

**Nothing is cached unless you name it.** `cacheable_tools` is empty by default. Read-only-ness
cannot be inferred from an ADK tool, and memoizing `create_ticket` silently drops the second
ticket.

**A tool that writes to `tool_context.actions` is blocklisted the moment it is caught.** A cache
hit reproduces the return value and none of the side effects, so a tool that transfers control,
sets state, or ends an invocation is unsafe to memoize even if a human declared it cacheable. The
allowlist is checked first and the actions check second, on purpose: the second one is there to
catch a wrong human assertion, not to replace it.

**An agent that gates its own tools is never cached.** This is the one that matters most. A
plugin-level cache hit short-circuits before the agent's own `before_tool_callback` chain runs,
and that chain is where authorization usually lives. Serving a cached result past a gate is an
authorization bypass, so `startup_checks` scans the agent tree, records any agent that declares
its own `before_tool_callback`, and `cache_hooks` refuses to serve for that agent. It is a
behaviour change made from a diagnostic scan, and it is the only one in the plugin. Nothing else
`startup_checks` finds changes what the router does.

## Data handling and retention

When `SmartRouterPlugin` is paired with ADK's `BigQueryAgentAnalyticsPlugin` or configured with
`cacheable_tools`, two classes of runtime data are handled:

- **BigQuery telemetry and user prompts (`agent_events`):** `BigQueryAgentAnalyticsPlugin` streams
  conversation events (including user prompts, tool call arguments, and `custom_metadata.smart_router`
  routing provenance) into your BigQuery `agent_events` table. The `smart-router readiness`,
  `smart-router status`, and `smart-router check-arguments` CLI commands and training scripts query
  this table (for example, `check-arguments` scans first-turn `USER_MESSAGE_RECEIVED` text to audit
  whether user messages supply held identifiers). Review your organization's data retention, IAM
  access controls, and `BigQueryLoggerConfig` content-logging settings before logging prompts to
  BigQuery.
- **In-memory tool result caching (`ToolResultCache`):** When tools are listed in `cacheable_tools`,
  their return payloads are stored in process memory keyed by a SHA-256 digest of
  `(app_name, user_id, tool_name, canonical_args)` until their configured TTL expires or a mutating
  tool invalidates the principal's cache entries. Cached tool outputs are never written to disk or
  exported to BigQuery (`event.custom_metadata["smart_router"]["cache_serves"]` records only the
  tool name and serve count, never the tool arguments or return payload). Only mark tools as
  cacheable if their outputs are deterministic and safe to reuse for the same `user_id` principal.

## Who can write the bucket

The digest proves the booster is the file the metadata named. It does not prove the metadata is
yours. Both files are written by the same publisher into the same directory, so anyone who can
write `gs://<bucket>/routers/` can replace both and the loader will verify a hostile artifact
against a hostile digest and be satisfied. That is the artifact trust boundary, and it is the
[Cloud Storage](https://cloud.google.com/storage) bucket.

**What a bucket writer gets.** The same subset property bounds them as bounds a user, but wider.
A published artifact chooses which subset of the declared tools a session is shown. It cannot add
a tool, rename one, change a schema, reach a tool's arguments, alter the cache configuration or
make any network call of its own: the booster is UBJSON parsed by XGBoost, which is data rather
than code, and nothing in the artifact is ever executed. So the worst outcome is denial of
function. The difference from an attacker who writes one message is reach and persistence. A
message degrades one session. An artifact degrades every session of that agent, in every replica,
until someone rolls it back.

**And they get every agent, not just one.** The layout is one bucket with `routers/{agent}/` under
it, so bucket write is not scoped per agent. If two teams publish for two agents into one bucket,
each can publish for the other. The same is true across environments: `environment` is checked at
load, but the field is written by the publisher, so it catches a mistake and does not stop a
deliberate act. Separate buckets are the boundary; the field is not.

**What is enforced in code.** `model_store.py` is written as a supply-chain boundary and the order
of its checks is the point. Agent and version names have to match `[A-Za-z0-9][A-Za-z0-9._-]*`
before they become local paths, so a blob named to walk out of `local_cache_dir` is refused rather
than written. `menus` and `classes` are validated as strictly as anything else, because those
strings reach the tool declarations. The menu count and menu size are capped, since an artifact
naming hundreds of menus would undo the thing menus are for. The booster's size is read from GCS
and checked against both the cap and the declared size before any bytes move, so an object that
lies about how big it is cannot be downloaded into a container's memory-backed `/tmp` and rejected
afterwards. Only then is the file fetched, digested, and moved into place under its final name.

**Why there is no signature.** Signing the artifact would defend against exactly one attacker: one
who has write on the bucket and not on anything else in the project. In the deployments this is
built for, that attacker is rare, because the identity that can write the bucket is usually one
that can also change the agent's code, its tools, or its deployment, and all three are strictly
better attacks than swapping a classifier. Against that narrow slice a signature costs a signing
key, a verification key in every serving process, rotation, and a new way for a good artifact to
fail to load. The router degrades to unrouted on any refusal, so the failure is cheap, but the key
management is not, and it would be protecting against denial of function that the holdout arm
already surfaces as collapsing recall.

The honest condition for revisiting this: if anyone runs a bucket shared across trust domains,
where the publisher for one agent is not trusted by another, signing stops being ceremony. The
cheaper answer for that case is a bucket per domain.

**What an operator should do instead.** These are the three that matter, and they are IAM rather
than cryptography.

- Publishing is one identity. Give the retraining job's service account object write on
  `routers/`, and give nobody else write, including humans. Serving identities get
  `roles/storage.objectViewer` and nothing more.
- Project-level roles are the leak. `roles/editor` and `roles/storage.admin` on the project write
  this bucket without appearing in its own IAM policy. `scripts/provision.sh --check` reports the
  bucket's bindings and says so.
- Keep object versioning on and data-access audit logs on. Versioning is what makes a bad artifact
  a rollback by generation rather than an incident, and the audit log is the only thing that
  answers who wrote it. `provision.sh` enforces versioning on every run; audit logging is a project
  setting and is yours.

## Poisoning the training data

The classifier trains on the agent's own logged traffic, so a user who can generate a lot of
traffic can shift the menu pool over time. What this buys is bounded by the same subset property:
poisoned training data changes which subsets get proposed, never what is in the declared set. The
worst outcome is a menu pool fitted to traffic nobody real sends, which shows up as falling recall
against the permanent holdout arm, which is what the holdout arm is for.

Two structural limits make it harder than it sounds. Labels come from tools that were actually
called successfully, not from what the message asked for, so a message has to make the agent do
something before it can teach the router anything. And publishing is gated: an artifact whose
recall falls more than 0.02 below the live one is refused, and so is one that does not clear the
recall and predicted-saving bars. [training.md](training.md) has the gate.

## What the router is not

**It is not an authorization boundary, and it must never become one.** Removing a tool from a
declaration is not revoking it. If the model can be talked into calling a tool it should not call,
the fix is a `before_tool_callback` that refuses, not a menu that hides it. Two reasons this
matters in practice. Whether a session gets pruned at all is an economic decision, not a security
one, so any protection built on pruning is on or off for reasons that have nothing to do with the
threat. And `mode` is a runtime kill switch, so a control built here is one environment variable
from being disabled.

**It does not decline on strange input.** It is tempting to read "the router passes through
whenever it is unconfident" as a safety property. `scripts/red_team.py` measures it, and on the
reference agent it is not true. The classifier does report its uncertainty honestly, at a contained
mass of 0.011 on pure noise against 0.576 on ordinary traffic, but the gate that consumes that
number is `worth_pruning`, which compares the tokens a prune removes against `(1 - mass) *
miss_cost_tokens`. Pruning ninety of a hundred tools is worth about 3,013 discounted tokens and the
most the gate can ever ask for is `miss_cost_tokens`, which is 1,600. The saving beats the ceiling
at every confidence level, so no input produces a decline on this agent.

That is the economically correct call: a miss costs one extra turn, which puts the break-even miss
rate at 58.8% against a measured 1.05%. It is stated here because a reader should not build on the
opposite. A garbage first message gets a confidently chosen menu, and the thing that makes that
safe is the subset property at the top of this document rather than any decline.

**It never fails a request.** Every callback swallows its own exceptions and returns "no opinion",
because ADK turns a plugin exception into a `RuntimeError` that aborts the whole run. A broken
artifact, an unreachable GCS bucket or a timed-out embedding call degrades to an unrouted agent.
That is an availability property rather than a security one, but it is the reason no failure mode
here turns into an outage.

## What an operator has to do

1. **Put safety-critical tools in `never_prune`.** Escalation, refusal, audit, handoff. Anything
   that must be reachable from any message regardless of what the message says.
2. **Do not cache a tool behind an authorization gate.** The plugin refuses this when it can see
   the gate as a `before_tool_callback`, which is where it usually lives. A gate implemented
   somewhere else, inside the tool function or in a service the tool calls, is invisible to that
   scan, so do not name that tool in `cacheable_tools`.
3. **Let one identity write the artifact bucket.** The retraining job's service account, not
   humans and not the serving identity. Check the project-level roles too, since those write the
   bucket without appearing in its policy. `scripts/provision.sh --check` prints the bucket's
   bindings.
4. **Register the router before your analytics plugin.** Not a security control, but an integrity
   one: reversed, the analytics plugin logs every cache hit as a real tool call, and the next model
   trains on the router's own output. `smart-router doctor` checks the ordering.

`smart-router doctor` checks the fourth of these and reports on the second, and
`provision.sh --check` reports on the third. The first is yours.

## Where the enforcement lives

| Property | Code | Test |
| --- | --- | --- |
| Pruning is subset-only | `pruner.decide` | `tests/test_pruner.py`, randomised walk |
| `never_prune` survives everything | `pruner.decide`, `protected` | `tests/test_pruner.py`, randomised walk |
| Neither survives a real attack | the whole serving path | `scripts/red_team.py`, 33 attacks |
| Cold-start & hybrid judges obey subset + rescue caps | `judges.combine_predictions` | `tests/test_judges.py` |
| Prototype menu cache & docstring rebuilds are bounded | `PrototypeJudge._ensure_index`, `menus_for` | `tests/test_judges.py` |
| Live LLM judge separates `system_instruction` and caps prompt | `LiveLlmJudge.predict` | `tests/test_judges.py` |
| No two callers share a cache key | `tool_cache.cache_key` | `tests/test_cache_key_fuzz.py` |
| Identity is always in the cache key | `tool_cache.cache_key` | `tests/test_tool_cache.py` |
| Nothing cached unless named | `cache_hooks`, `cacheable_tools` | `tests/test_cache_safety.py` |
| Side-effecting tools are blocklisted | `cache_hooks`, `actions_are_default` | `tests/test_cache_safety.py` |
| Gated agents are never served from cache | `startup_checks.gates_own_tools` | `tests/test_cache_safety.py` |
| An artifact is the file its digest names | `model_store._ensure_local` | `tests/test_model_store.py` |
| An artifact cannot write outside the cache dir | `model_store._check_segment` | `tests/test_model_store.py` |
| An oversized object is refused unread | `model_store._sized_blob` | `tests/test_model_store.py` |
| Menus and classes reaching declarations are validated | `model_store.parse_metadata` | `tests/test_model_store.py` |
| No callback can fail a request | every callback in `plugin.py` | `tests/test_plugin.py` |

"""Who gets routed, and which decision belongs to which session. Pure and deterministic.

Two questions are answered here and nowhere else:

1. Is this session in the holdout, and is it inside the rollout ramp? Both are answered by
   hashing the same three identifiers under different salts, so the control arm and the ramp
   are statistically independent. Nothing is stored, so every replica agrees.
2. What is this session's scope key? One routing decision is made per (session, agent), and
   this is the key it is filed under.

The hash is over the session, never over the turn. Under session-stable pruning a turn-hashed
split is contaminated: the decision is already frozen for the conversation, so roughly half of
the "control" turns would carry the treatment (DESIGN.md 11.4).
"""

from __future__ import annotations

import hashlib

from adk_smart_router.config import RouterConfig

_HOLDOUT_SALT = "smart_router/holdout"
_ROUTE_SALT = "smart_router/route"
_SCOPE_SALT = "smart_router/scope"

# Length-agnostic field separator. Salts are fixed strings and the identifiers come from ADK,
# so there is no adversary to defend against here; this only stops "ab"+"c" colliding with
# "a"+"bc".
_FIELD_SEP = "\x1f"


def _digest(salt: str, *fields: str) -> bytes:
    return hashlib.sha256(_FIELD_SEP.join((salt, *fields)).encode("utf-8")).digest()


def session_fraction(salt: str, app_name: str, user_id: str, session_id: str) -> float:
    """A stable point in [0, 1) for one session under one salt."""
    return int.from_bytes(_digest(salt, app_name, user_id, session_id)[:8], "big") / 2.0**64


def in_holdout(cfg: RouterConfig, app_name: str, user_id: str, session_id: str) -> bool:
    """True for the permanent control arm.

    Not a rollout phase that ends. It is the only unbiased training corpus and the only valid
    A/B baseline, so it stays switched on at full rollout (DESIGN.md 11.4).
    """
    return session_fraction(_HOLDOUT_SALT, app_name, user_id, session_id) < cfg.holdout_fraction


def is_routed(cfg: RouterConfig, app_name: str, user_id: str, session_id: str) -> bool:
    """True if this session is inside the rollout ramp. Independent of the holdout by salt."""
    return session_fraction(_ROUTE_SALT, app_name, user_id, session_id) < cfg.route_fraction


def scope_key(app_name: str, user_id: str, session_id: str, agent_name: str) -> str:
    """The key one session's routing decision is filed under.

    The agent is part of it because plugins install on the Runner, not on an agent, so a single
    plugin instance sees every agent in the tree and each one has its own model (DESIGN.md 7.6,
    D9).
    """
    return _digest(_SCOPE_SALT, app_name, user_id, session_id, agent_name).hex()

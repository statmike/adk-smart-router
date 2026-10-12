"""Reading from and writing to an `LlmRequest`. The only module that touches one.

Two operations, in the order the plugin uses them: find the prompt to route on, and apply a
decision to the declared tool list.

The prune deliberately edits `config.tools` and leaves `tools_dict` alone. `tools_dict` is what
ADK uses to execute a call, so filtering it would turn a stray call to a pruned tool into an
exception inside `_get_tool`, and BQAA would write a TOOL_ERROR row blaming the tool. The
router's own mistakes would then look like tool failures in the exact telemetry used to detect
over-pruning. Left intact, the same event is a silent, correct recovery (DESIGN.md 7.4).

There used to be a third operation here, a one-line advisory naming the kept tools. It was
removed because it cost more than it was worth: it is inserted only on the turn the router
decides, and ADK places transient content ahead of the latest user message, so the deciding turn
and every turn after it disagreed about the third element of `contents` and no later turn could
reuse the session's own cached prefix. DESIGN.md 7.5 and 18.18 have the measurement.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.adk.models.llm_request import LlmRequest


@dataclass(frozen=True)
class ToolSpec:
    """A tool's name, full docstring/description, and parameter names -> descriptions."""

    name: str
    description: str
    parameters: tuple[tuple[str, str], ...] = ()

    @property
    def param_names(self) -> frozenset[str]:
        """The parameter names accepted by this tool."""
        return frozenset(k for k, _ in self.parameters)

    def document(self) -> str:
        """Rich text representation for semantic prototype embedding."""
        params = sorted(self.param_names - {"tool_context"})
        param_str = f" (parameters: {', '.join(params)})" if params else ""
        return f"{self.name.replace('_', ' ')}{param_str}: {self.description}"


def tool_specs(llm_request: LlmRequest) -> dict[str, ToolSpec]:
    """Extract `ToolSpec` metadata for every live tool on `llm_request`.

    Reads both `config.tools[].function_declarations` and `tools_dict` because ADK splits
    what it knows across the two: `JSON_SCHEMA_FOR_FUNC_DECL` puts parameter schemas in
    `declaration.parameters_json_schema` (leaving `declaration.parameters` None), while
    `declaration.description` strips everything after the first paragraph of the Python
    docstring (`Returns:` included), which still lives on `tools_dict[name].func.__doc__`.
    """
    tools_dict: Mapping[str, Any] = getattr(llm_request, "tools_dict", None) or {}
    decls_by_name: dict[str, Any] = {}
    config = getattr(llm_request, "config", None)
    for tool in list(getattr(config, "tools", None) or []):
        for decl in getattr(tool, "function_declarations", None) or []:
            name = getattr(decl, "name", None)
            if name:
                decls_by_name[name] = decl

    all_names = list(dict.fromkeys([*tools_dict.keys(), *decls_by_name.keys()]))
    specs: dict[str, ToolSpec] = {}
    for name in all_names:
        tool_obj = tools_dict.get(name)
        decl = decls_by_name.get(name)
        func = getattr(tool_obj, "func", None)
        raw_doc = (
            getattr(func, "__doc__", None)
            or getattr(decl, "description", None)
            or getattr(tool_obj, "description", None)
            or ""
        )
        desc = inspect.cleandoc(str(raw_doc)).strip() if raw_doc else ""
        params = _extract_parameters(decl, func)
        specs[name] = ToolSpec(name=name, description=desc, parameters=params)
    return specs


def _extract_parameters(decl: Any, func: Any) -> tuple[tuple[str, str], ...]:  # noqa: ANN401
    """Read parameter `(name, description)` pairs from either ADK declaration representation."""
    if decl is not None:
        json_schema = getattr(decl, "parameters_json_schema", None)
        if isinstance(json_schema, Mapping):
            props = json_schema.get("properties")
            if isinstance(props, Mapping):
                return tuple(
                    (
                        str(k),
                        str(v.get("description", "") or "") if isinstance(v, Mapping) else "",
                    )
                    for k, v in props.items()
                )
        params_obj = getattr(decl, "parameters", None)
        props = getattr(params_obj, "properties", None)
        if isinstance(props, Mapping):
            return tuple(
                (str(k), str(getattr(v, "description", "") or "")) for k, v in props.items()
            )
    if callable(func):
        try:
            sig = inspect.signature(func)
            return tuple(
                (k, "")
                for k in sig.parameters
                if k not in ("self", "cls", "tool_context", "callback_context")
            )
        except (ValueError, TypeError):
            pass
    return ()


def latest_user_text(llm_request: LlmRequest) -> str | None:
    """The current user prompt: the last user Content, its text parts joined.

    None when there is nothing new to route on, which is the normal case for a tool-call
    continuation turn.
    """
    for content in reversed(llm_request.contents or []):
        if content.role != "user":
            continue
        text = " ".join(part.text for part in (content.parts or []) if part.text).strip()
        if text:
            return text
    return None


def apply_prune(llm_request: LlmRequest, kept: frozenset[str]) -> None:
    """Filter `config.tools[].function_declarations` down to `kept`. `tools_dict` is untouched."""
    config = llm_request.config
    tools = list(getattr(config, "tools", None) or [])
    if not tools:
        return

    survivors: list[Any] = []
    for tool in tools:
        declarations = getattr(tool, "function_declarations", None)
        if not declarations:
            survivors.append(tool)  # google_search, code_execution and friends are not prunable
            continue
        filtered = [d for d in declarations if d.name is None or d.name in kept]
        if len(filtered) == len(declarations):
            survivors.append(tool)
            continue
        if not filtered:
            continue  # drop the Tool rather than leaving an empty shell
        # A copy, not an in-place edit: the declaration objects can be shared with the tool that
        # produced them, and under live/bidi `config` itself is shared by reference.
        survivors.append(tool.model_copy(update={"function_declarations": filtered}))

    config.tools = survivors or None

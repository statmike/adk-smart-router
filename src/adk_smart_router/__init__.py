"""adk-smart-router: predictive tool routing and result memoization for Google ADK.

The public API is three names. Everything else in the package is an implementation detail and
may change without notice: install the plugin, hand it a `RouterConfig`, and it does the rest
from the callback context (DESIGN.md 6.3).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from adk_smart_router.config import RouterConfig, load_config
from adk_smart_router.judges import ToolJudge
from adk_smart_router.plugin import SmartRouterPlugin

try:
    # Single-sourced from pyproject.toml when installed; fallback is tripwired
    # against pyproject.toml in tests/test_packaging.py for zip-staged workers.
    __version__ = version("adk-smart-router")
except PackageNotFoundError:  # running from a source tree or zip bundle without .dist-info
    __version__ = "1.0.1"

__all__ = ["RouterConfig", "SmartRouterPlugin", "ToolJudge", "__version__", "load_config"]

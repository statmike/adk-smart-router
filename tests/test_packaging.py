"""Packaging, version-parity, and PyPI README link-rewriting tripwires."""

from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path

import adk_smart_router

REPO_ROOT = Path(__file__).resolve().parents[1]


def _extract_init_fallback_version(init_py: Path) -> str:
    """Return the string literal assigned to __version__ in the except block."""
    tree = ast.parse(init_py.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == "__version__"
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    return stmt.value.value
    raise AssertionError("Could not find __version__ fallback literal in __init__.py")


def _extract_skill_version(skill_md: Path) -> str:
    """Extract metadata.version from the YAML frontmatter of SKILL.md."""
    text = skill_md.read_text(encoding="utf-8")
    frontmatter = text.split("---", 2)[1]
    match = re.search(r"^\s*version:\s*(\S+)\s*$", frontmatter, re.MULTILINE)
    assert match is not None, "No version field in SKILL.md frontmatter"
    return match.group(1).strip("\"'")


def test_version_parity_across_pyproject_init_plugin_skill_and_changelog():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = pyproject["project"]["version"]

    fallback = _extract_init_fallback_version(REPO_ROOT / "src/adk_smart_router/__init__.py")
    assert fallback == expected, (
        f"__init__.py fallback version {fallback!r} != pyproject.toml {expected!r}"
    )
    assert adk_smart_router.__version__ == expected

    plugin_json = json.loads((REPO_ROOT / "plugin.json").read_text(encoding="utf-8"))
    assert plugin_json["version"] == expected, (
        f"plugin.json version {plugin_json['version']!r} != pyproject.toml {expected!r}"
    )

    skill_ver = _extract_skill_version(REPO_ROOT / "skills/adk-smart-router/SKILL.md")
    assert skill_ver == expected, (
        f"SKILL.md metadata.version {skill_ver!r} != pyproject.toml {expected!r}"
    )

    base_version = re.sub(r"(?:rc|a|b)\d+$", "", expected)
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{expected}]" in changelog or f"## [{base_version}]" in changelog, (
        f"CHANGELOG.md has no ## [{expected}] or ## [{base_version}] heading"
    )


def test_readme_relative_links_exist_and_rewrite_cleanly_for_pypi():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    hook = pyproject["tool"]["hatch"]["metadata"]["hooks"]["fancy-pypi-readme"]
    substitutions = hook["substitutions"]
    assert substitutions, "Expected fancy-pypi-readme substitutions in pyproject.toml"

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    rel_links = re.findall(r"\]\((?!https?://|#|mailto:)([^)]+)\)", readme)
    assert rel_links, "Expected relative links in README.md"

    # Every relative link in README.md must point to a real file in the repository.
    for target in rel_links:
        file_part = target.split("#", 1)[0]
        assert (REPO_ROOT / file_part).exists(), f"README.md links to non-existent path: {target!r}"

    # Applying the fancy-pypi-readme substitutions must leave zero relative links.
    rendered = readme
    for sub in substitutions:
        rendered = re.sub(sub["pattern"], sub["replacement"], rendered)

    remaining = re.findall(r"\]\((?!https?://|#|mailto:)([^)]+)\)", rendered)
    assert not remaining, (
        f"Un-rewritten relative links remain after fancy-pypi-readme substitution: {remaining}"
    )
    assert "https://github.com/statmike/adk-smart-router/blob/main/docs/evidence.md" in rendered
    assert (
        "https://github.com/statmike/adk-smart-router/blob/main/docs/operations.md#the-cache-guards"
        in rendered
    )

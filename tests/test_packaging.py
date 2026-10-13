"""Packaging, version-parity, and PyPI README link-rewriting tripwires."""

from __future__ import annotations

import ast
import json
import re
import struct
import tomllib
import types
from collections.abc import Mapping
from pathlib import Path

import adk_smart_router

REPO_ROOT = Path(__file__).resolve().parents[1]

_SKILL_VERSION_PATTERN = re.compile(r"^\s*version:\s*(\S+)\s*$", re.MULTILINE)
_CITATION_VERSION_PATTERN = re.compile(r'^version:\s*"([^"]+)"', re.MULTILINE)
_RELEASE_CANDIDATE_SUFFIX_PATTERN = re.compile(r"(?:rc|a|b)\d+$")
_RELATIVE_LINK_PATTERN = re.compile(r"\]\((?!https?://|#|mailto:)([^)]+)\)")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_IHDR_CHUNK = b"IHDR"

_EXPECTED_README_ASSET_PNGS: tuple[str, ...] = (
    "social_card.png",
    "readme_judge_tiers.png",
    "readme_turn_sequence.png",
    "readme_training_loop.png",
)

_EXPECTED_ASSET_DIMENSIONS: Mapping[str, tuple[int, int]] = types.MappingProxyType(
    {
        "social_card": (1280, 640),
        "readme_judge_tiers": (1060, 410),
        "readme_turn_sequence": (1120, 660),
        "readme_training_loop": (600, 440),
    }
)


def _extract_init_fallback_version(init_py: Path) -> str:
    """Return the string literal assigned to __version__ in the except block."""
    tree = ast.parse(init_py.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            for statement in node.body:
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == "__version__"
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                ):
                    return statement.value.value
    raise AssertionError("Could not find __version__ fallback literal in __init__.py")


def _extract_skill_version(skill_md: Path) -> str:
    """Extract metadata.version from the YAML frontmatter of SKILL.md."""
    text = skill_md.read_text(encoding="utf-8")
    frontmatter = text.split("---", 2)[1]
    match = _SKILL_VERSION_PATTERN.search(frontmatter)
    assert match is not None, "No version field in SKILL.md frontmatter"
    return match.group(1).strip("\"'")


def test_version_parity_across_pyproject_init_plugin_skill_and_changelog() -> None:
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

    skill_version = _extract_skill_version(REPO_ROOT / "skills/adk-smart-router/SKILL.md")
    assert skill_version == expected, (
        f"SKILL.md metadata.version {skill_version!r} != pyproject.toml {expected!r}"
    )

    base_version = _RELEASE_CANDIDATE_SUFFIX_PATTERN.sub("", expected)
    citation_text = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation_match = _CITATION_VERSION_PATTERN.search(citation_text)
    assert citation_match is not None, "No version field in CITATION.cff"
    assert citation_match.group(1) in {expected, base_version}, (
        f"CITATION.cff version {citation_match.group(1)!r} != pyproject.toml {expected!r}"
    )

    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{expected}]" in changelog or f"## [{base_version}]" in changelog, (
        f"CHANGELOG.md has no ## [{expected}] or ## [{base_version}] heading"
    )


def test_readme_relative_links_exist_and_rewrite_cleanly_for_pypi() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    hook = pyproject["tool"]["hatch"]["metadata"]["hooks"]["fancy-pypi-readme"]
    substitutions = hook["substitutions"]
    assert substitutions, "Expected fancy-pypi-readme substitutions in pyproject.toml"

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    relative_links = _RELATIVE_LINK_PATTERN.findall(readme)
    assert relative_links, "Expected relative links in README.md"

    # Every relative link in README.md must point to a real file in the repository.
    for target in relative_links:
        file_part = target.split("#", 1)[0]
        assert (REPO_ROOT / file_part).exists(), f"README.md links to non-existent path: {target!r}"

    # Applying the fancy-pypi-readme substitutions must leave zero relative links.
    rendered = readme
    for substitution in substitutions:
        rendered = re.sub(substitution["pattern"], substitution["replacement"], rendered)

    remaining = _RELATIVE_LINK_PATTERN.findall(rendered)
    assert not remaining, (
        f"Un-rewritten relative links remain after fancy-pypi-readme substitution: {remaining}"
    )
    assert "```mermaid" not in rendered, (
        "Un-substituted ```mermaid block remains in PyPI README rendering"
    )
    assert "https://github.com/statmike/adk-smart-router/blob/main/docs/evidence.md" in rendered
    assert (
        "https://github.com/statmike/adk-smart-router/blob/main/docs/operations.md#the-cache-guards"
        in rendered
    )
    for asset_png in _EXPECTED_README_ASSET_PNGS:
        assert (
            f"https://raw.githubusercontent.com/statmike/adk-smart-router/main/docs/assets/{asset_png}"
            in rendered
        ), f"Missing {asset_png} in rendered PyPI README"


def test_social_card_and_mermaid_fallback_pngs() -> None:
    assets_dir = REPO_ROOT / "docs/assets"
    assert (assets_dir / "README.md").is_file(), "Missing docs/assets/README.md"

    for stem, expected_dimensions in _EXPECTED_ASSET_DIMENSIONS.items():
        html_path = assets_dir / f"{stem}.html"
        png_path = assets_dir / f"{stem}.png"
        assert html_path.is_file(), f"Missing docs/assets/{stem}.html"
        assert png_path.is_file(), f"Missing docs/assets/{stem}.png"

        data = png_path.read_bytes()
        assert data[:8] == _PNG_SIGNATURE, f"docs/assets/{stem}.png is not a valid PNG"
        assert data[12:16] == _PNG_IHDR_CHUNK, f"docs/assets/{stem}.png missing IHDR chunk"
        width, height = struct.unpack(">II", data[16:24])
        assert (width, height) == expected_dimensions, (
            f"docs/assets/{stem}.png must be {expected_dimensions}, got {(width, height)}"
        )

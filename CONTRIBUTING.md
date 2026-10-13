# Contributing

Issues and pull requests are welcome. This guide covers how the repository is organized, how to run
the verification suite, and the architectural invariants every change must preserve.

## Before you write code

Open an issue first for anything larger than a bug fix:

- [docs/architecture.md](docs/architecture.md) walks a single request through the plugin and
  explains why each module is separate.
- [docs/evidence.md](docs/evidence.md) and [docs/scenarios.md](docs/scenarios.md) document every
  measured claim, including the experiments and agent shapes where pruning lost money and the guards
  that were added in response.
- [docs/threat-model.md](docs/threat-model.md) documents the security boundaries (subset-only
  pruning, principal-scoped tool caching, and the artifact supply chain).

## The verification gate

One command, and it is the same one CI runs across Python 3.11, 3.12, and 3.13:

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest -q
```

`uv` is the supported way to manage the environment (`uv sync` installs the package and the `dev`
dependency group).

Four checks gate every merge:

- **Formatted and linted** by `ruff`, at 100 columns.
- **Typed.** `mypy` runs in strict mode over `src/`. Every function signature carries type
  annotations.
- **Tested.** The test suite is offline by construction: `tests/conftest.py` fails any test that
  opens a network socket off the machine or reads ambient `SMART_ROUTER_*` environment variables. If
  your change needs a live cloud call to be verified, that belongs in `scripts/`, not in `tests/`.
- **Packaged.** `./scripts/check_install.sh` builds the wheel, installs it into an isolated
  environment, and runs the CLI against `examples/tiny_agent`.

## The shape of the code

Flat, single-responsibility modules, one capability per file. Avoid deep inheritance and clever
abstractions. A reader should understand any module in seconds, and legibility wins over elegance
when the two disagree.

Two rules that are not style:

- **Never load a service account key.** Authentication is Application Default Credentials (ADC)
  only. Clients are constructed so they inherit ADC locally and the attached service account in
  production. `.gitignore` blocks committing key files, and tests verify that no credential file is
  loaded.
- **Never name a live project, bucket, or dataset.** Nothing in this repository hardcodes a live
  cloud resource. Configuration comes from `RouterConfig` or `SMART_ROUTER_*` environment variables
  at runtime.

## Tests

Most of the test suite is about failure rather than success: what the router does when the artifact
is missing, when the embedding call times out, when two replicas disagree, or when the model reports
token counts that contradict our own arithmetic. A change to serving behaviour wants a test in that
spirit—asserting that the agent still works and only the token saving degrades.

Test names are descriptive sentences (`test_the_parents_decision_does_not_prune_the_sub_agent`).

## Commits

Explain *why*, not *what*. The diff already says what.

## Releasing

Releases are published to TestPyPI and PyPI via GitHub Actions OIDC Trusted Publishing ([`.github/workflows/release.yml`](.github/workflows/release.yml)) whenever a `v*` tag is pushed.

### 1. Version bump & changelog
Update the version string in all four single-sourced/tripwired locations (enforced by `tests/test_packaging.py`):
1. `pyproject.toml` (`[project].version`)
2. `src/adk_smart_router/__init__.py` (`PackageNotFoundError` fallback literal)
3. `plugin.json` (`"version"`)
4. `skills/adk-smart-router/SKILL.md` (`metadata.version`)

Add or update the corresponding `## [X.Y.Z] - YYYY-MM-DD` section in [`CHANGELOG.md`](CHANGELOG.md), then run the full local gate (`uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest -q && ./scripts/check_install.sh`) and merge via pull request to `main`.

### 2. Pre-release verification on TestPyPI (`vX.Y.ZrcN`)
Before cutting a new major or minor release on production PyPI (where version numbers cannot be reused or edited), validate the build and OIDC pipeline with a release candidate tag directly on `main` (zero version-bump commits required):
1. Push an annotated `vX.Y.Zrc1` tag on the `main` commit whose `pyproject.toml` is at `X.Y.Z`:
   ```bash
   git tag -a vX.Y.Zrc1 -m "Release candidate vX.Y.Zrc1"
   git push origin vX.Y.Zrc1
   ```
2. `.github/workflows/release.yml` verifies `vX.Y.Zrc1` matches base version `X.Y.Z`, stamps `X.Y.Zrc1` into the build environment, and routes the tag automatically to the `publish-testpypi` job (`testpypi` environment).
3. Verify the published package in an isolated environment from TestPyPI and inspect the rendered project page on `https://test.pypi.org/project/adk-smart-router/X.Y.Zrc1/`:
   ```bash
   uvx --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ \
       --from adk-smart-router==X.Y.Zrc1 \
       smart-router --help
   ```

### 3. Production release on PyPI (`vX.Y.Z`)
1. Push the final annotated `vX.Y.Z` tag on the same `main` commit:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```
2. Non-`rc` tags route to `publish-pypi` (`pypi` environment) and then `github-release` (attaching the `.whl` and `.tar.gz` distributions).
3. Verify the live PyPI release:
   ```bash
   uvx --from adk-smart-router==X.Y.Z smart-router --help
   ```

### 4. Patch / hotfix releases & CI enforcement
- For patch releases (`X.Y.Z+1`), branch from `main` (or from the `vX.Y.Z` tag if `main` has diverged with next-minor work), cherry-pick the fix, bump the patch version and `CHANGELOG.md`, and tag `vX.Y.Z+1`.
- **What CI refuses automatically:**
  - `tests/test_packaging.py` fails if `pyproject.toml`, `src/adk_smart_router/__init__.py`, `plugin.json`, `skills/adk-smart-router/SKILL.md`, or `CHANGELOG.md` disagree on the version, if the built PyPI `Description` contains any un-rewritten relative links or un-substituted ` ```mermaid ` blocks, or if the `sdist` includes excluded development artifacts.
  - `.github/workflows/release.yml` refuses to build or publish if the pushed git tag (`v<version>` or `v<version>rcN`) does not match `pyproject.toml` `[project].version`.

## Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA). You (or your
employer) retain the copyright to your contribution; this simply gives permission to use and
redistribute your contributions as part of the project.

If you have not already done so, please complete your CLA at
<https://cla.developers.google.com/> before submitting a pull request.

## Community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## License

By contributing to `adk-smart-router`, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE).

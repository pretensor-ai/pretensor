# Releases

How Pretensor versions and ships.

## Versioning policy

Pretensor follows [PEP 440](https://peps.python.org/pep-0440/) with SemVer
semantics for `major.minor.patch`.

### Scheme

`major.minor.patch` with optional PEP 440 suffixes:

- `aN` — alpha
- `bN` — beta
- `rcN` — release candidate
- `.postN` / `.devN` exist but are not part of the routine flow

### Stability

Pretensor is in the `0.x.y` series. **CLI flags, MCP tools, and graph schema
can change in any `0.x` minor bump.** `1.0.0` will be the first stability
commitment. Pin exact versions until then.

### Prereleases

| Suffix | Stage | What can change between iterations                          |
|--------|-------|-------------------------------------------------------------|
| `aN`   | Alpha | Anything: CLI flags, MCP tools, graph schema                |
| `bN`   | Beta  | Bug fixes only; no incompatible surface changes within `bN` |
| `rcN`  | RC    | Only blocker-bug fixes; no new functionality                |

### Bump rules

- Bug fix → patch (`0.1.0` → `0.1.1`).
- New backward-compatible feature → minor.
- Breaking change while in `0.x` → minor.
- Breaking change post-1.0 → major.
- Prerelease iteration → increment the suffix (`0.2.0a1` → `0.2.0a2`).

### Choosing the next tag

> **The version for a tag cut is chosen at tag time** from the current state
> of `main`, not pre-committed in any backlog or roadmap. Inspect what has
> landed since the last tag, apply the bump rules, and pick the version then.

Examples:

- Bug fix during the `0.1.0` alpha series → `0.1.0a2`. Fixes inside a
  prerelease iterate the suffix, not the patch field.
- Breaking change pre-1.0 against `0.1.x` → bump the minor and reset to a
  fresh prerelease, e.g. `0.2.0a0`.

### References

- [PEP 440 — Pre-releases](https://peps.python.org/pep-0440/#pre-releases)
- [CHANGELOG.md](../CHANGELOG.md)

## Publishing

The `.github/workflows/release.yml` workflow publishes tagged versions to
PyPI via OIDC trusted publishing. It runs on pushes to tags matching
`v*.*.*` and has four jobs:

1. `build` — builds sdist + wheel with `uv build`, validates with
   `twine check --strict`, uploads the `dist/` artifact.
2. `detect` — classifies the tag as prerelease or stable via regex
   (`^v[0-9]+\.[0-9]+\.[0-9]+(a|b|rc)[0-9]+$`).
3. `test-pypi` — runs only for prerelease tags. Publishes to TestPyPI
   through the `testpypi-release` GitHub Environment.
4. `pypi` — runs for all `v*.*.*` tags. Publishes to PyPI through the
   `pypi-release` GitHub Environment (gated by a required reviewer).
   For prerelease tags, waits on `test-pypi` success. For stable tags,
   `test-pypi` is skipped and `pypi` proceeds directly after `build`.

No long-lived PyPI tokens live in GitHub. OIDC trusted publishing is the
only authentication path.

## One-time OIDC setup

Configured once per PyPI project. These steps do not live in the repo.

1. **PyPI trusted publisher** — on PyPI, open the `pretensor` project →
   Settings → Publishing → Add a pending publisher:
   - Owner: `pretensor-ai`
   - Repository: `pretensor`
   - Workflow: `release.yml`
   - Environment: `pypi-release`
2. **TestPyPI trusted publisher** — same UI on test.pypi.org:
   - Owner: `pretensor-ai`
   - Repository: `pretensor`
   - Workflow: `release.yml`
   - Environment: `testpypi-release`
3. **GitHub Environments** on `pretensor-ai/pretensor` (Settings →
   Environments):
   - Create `pypi-release` with a required-reviewer protection rule
     (at least one release approver).
   - Create `testpypi-release` with no gating.

## Pre-merge dry-run (optional)

Maintainers can exercise the workflow end-to-end against TestPyPI from a
non-canonical source (a separate repository, fork, or branch) before
cutting a real tag from this repo:

1. On **TestPyPI only**, add a secondary trusted publisher that points at
   the non-canonical source — workflow `release.yml`, environment
   `testpypi-release`. Do **not** add this on real PyPI.
2. On that source repo, create the two GitHub Environments
   (`pypi-release` and `testpypi-release`) with the same names and
   protection rules described above.
3. From a branch there, push an annotated dry-run tag:

   ```bash
   git tag -a v0.0.0rc0 -m "dry-run"
   git push origin v0.0.0rc0
   ```

4. Expected: `build` → `detect` → `test-pypi` all green. OIDC against
   TestPyPI succeeds because the secondary source is temporarily
   trusted there. The `pypi` job then pauses at the `pypi-release`
   environment approval gate — **reject** the deployment. If the
   deployment is approved by mistake, OIDC against real PyPI will still
   fail because the secondary source is not a trusted publisher there,
   so no version is burned on real PyPI either way.
5. Verify the release appears at
   `https://test.pypi.org/project/pretensor/0.0.0rc0/`.
6. Clean up afterwards: remove the secondary trusted publisher from
   TestPyPI settings so only this repo remains trusted.

Note: TestPyPI does not allow re-uploading the same version. If the
dry-run needs to re-run, bump the suffix (`v0.0.0rc1`) and repeat.

## Cutting a release

1. Land all release content on the mainline branch. Bump
   `pyproject.toml` version per the versioning policy above.
2. Create and push an annotated tag:

   ```bash
   git tag -a v0.2.0 -m "v0.2.0"
   git push origin v0.2.0
   ```

3. Workflow shape by tag kind:
   - Prerelease (`v0.2.0rc1`): `build` → `detect` → `test-pypi` →
     (approval) → `pypi`.
   - Stable (`v0.2.0`): `build` → `detect` → `test-pypi` skipped →
     (approval) → `pypi`.
4. Approve the `pypi-release` deployment in the GitHub UI when ready.

## Re-pointing after an ownership change

If the PyPI project is transferred between accounts or organizations (for
example, from a personal account to `pretensor-ai`), no workflow change
is required. On PyPI, re-configure the trusted publisher under the new
owner and re-verify that the GitHub Environment names match. The
workflow file stays identical.

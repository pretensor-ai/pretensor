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

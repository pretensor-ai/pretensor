# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [PEP 440](https://peps.python.org/pep-0440/) versioning.

Pretensor is currently in alpha on PyPI (`pip install pretensor` —
no `--pre` needed yet because no stable release exists). Until `1.0.0`
ships, the schema and CLI surface can change between versions; this
changelog tracks what moves between releases.

## [Unreleased]

### Added
- CD pipeline at `.github/workflows/release.yml` that publishes tagged
  `v*.*.*` versions to PyPI via OIDC trusted publishing, routing
  prereleases through TestPyPI first.
- `"Typing :: Typed"` PyPI classifier, pairing the existing
  `src/pretensor/py.typed` PEP 561 marker with its public advertisement.
- Release runbook sections in `docs/releases.md` covering publishing,
  one-time OIDC setup, an optional pre-merge dry-run, cutting a release,
  and the post-transfer re-point procedure.

<!--
When cutting a release, copy the contents of [Unreleased] into a new
`## [X.Y.Z] - YYYY-MM-DD` section above, then reset [Unreleased] to empty
subsections (Added / Changed / Deprecated / Removed / Fixed / Security).
-->

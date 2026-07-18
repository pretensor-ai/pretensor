# v0.1.0a3 — bootstrap baseline

The release gate compares each tag cut against the previous tag's stored
results. `v0.1.0a3` predates the benchmark harness (the harness merged two
days after that tag was cut), so no results were — or can be — generated
from the `v0.1.0a3` code itself.

These files are a copy of the PR-lane reference baselines
(`tests/benchmark/baselines/<dataset>-<level>.json`) at the time the first
gated release (`v0.1.0a4`) was cut. The PR lane verifies every pull request
against those same numbers, so this seed holds the gate to the exact bar
`main` was already held to.

From `v0.1.0a4` onward, the post-publish `archive-results` job stores each
tag's real run output, and this bootstrap convention is never needed again.

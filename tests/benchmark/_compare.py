"""Helpers for comparing benchmark runner output against committed baselines."""

from __future__ import annotations

import re

# `pretensor_version` is emitted into every baseline JSON as a debugging
# aid, but it bumps on every package release. Treating it as part of the
# byte-equality contract would force every baseline to be re-committed on
# each version bump, which the underlying metrics did not actually drift
# through. Normalize it before comparing.
_VOLATILE_VERSION_RE = re.compile(rb'"pretensor_version": "[^"]*"')
_NORMALIZED_VERSION = b'"pretensor_version": "<normalized>"'


def normalize_baseline_bytes(content: bytes) -> bytes:
    """Replace the running pretensor_version with a fixed placeholder.

    Leaves every other byte of the baseline intact so the deterministic
    serialization contract (sorted keys, indent 2, trailing newline) is
    still enforced.
    """
    return _VOLATILE_VERSION_RE.sub(_NORMALIZED_VERSION, content)

"""Tests for ``scripts/check_no_internal_refs.py``.

Covers every branch the script's logic depends on: clean input, the two
configured patterns (issue ID and tracker URL), the ``allowlist:`` bypass,
the suffix-based skip list, the self-path exemption, and the directory /
``OSError`` swallowing path.

Test fixtures necessarily contain literal tracker patterns. Each such
literal is bound to a single source line carrying an ``allowlist:`` marker
(the script's own bypass mechanism, itself exercised by
``test_allowlist_marker_suppresses_match``); downstream assertions then
reference the local variable so they don't reintroduce the pattern.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr
from pathlib import Path

from scripts.check_no_internal_refs import _SELF_PATHS, main

_SAMPLE_ID = "PRE-1234"  # allowlist: test fixture binds the matched literal
_SAMPLE_ID_SHORT = "PRE-99"  # allowlist: test fixture binds the matched literal
_SAMPLE_URL = "linear.app"  # allowlist: test fixture binds the matched literal


def _run(paths: list[str]) -> tuple[int, str]:
    """Invoke ``main`` and capture stderr."""
    buf = io.StringIO()
    with redirect_stderr(buf):
        code = main(paths)
    return code, buf.getvalue()


def test_clean_file_exits_zero(tmp_path: Path) -> None:
    """A file with no offending tokens returns exit 0."""
    f = tmp_path / "clean.py"
    f.write_text("'''Plain module.'''\n\nx = 1\n", encoding="utf-8")

    code, err = _run([str(f)])

    assert code == 0
    assert err == ""


def test_issue_id_hit_exits_one_with_path_line_and_token(tmp_path: Path) -> None:
    """An issue-ID reference fails with the path, line number, and token in stderr."""
    f = tmp_path / "leaky.py"
    f.write_text(f"# header\n# implements {_SAMPLE_ID} acceptance\nx = 1\n")

    code, err = _run([str(f)])

    assert code == 1
    assert str(f) in err
    assert ":2:" in err
    assert _SAMPLE_ID in err


def test_tracker_url_hit_exits_one(tmp_path: Path) -> None:
    """A tracker-URL reference fails."""
    f = tmp_path / "url.md"
    f.write_text(f"See https://{_SAMPLE_URL}/foo/issue/X for context.\n")

    code, err = _run([str(f)])

    assert code == 1
    assert _SAMPLE_URL in err


def test_allowlist_marker_suppresses_match(tmp_path: Path) -> None:
    """A line carrying ``allowlist: <reason>`` is skipped even if it matches."""
    f = tmp_path / "allow.py"
    f.write_text(
        f"# Historical mention of {_SAMPLE_ID}  # allowlist: changelog backfill\n"
    )

    code, err = _run([str(f)])

    assert code == 0, err


def test_skip_suffix_file_is_not_scanned(tmp_path: Path) -> None:
    """A file whose suffix is in ``_SKIP_SUFFIXES`` is not opened or scanned."""
    f = tmp_path / "image.png"
    # The fixture bytes contain a matching token; if the suffix gate failed
    # the script would still open and flag the file.
    f.write_bytes(_SAMPLE_ID.encode("utf-8"))

    code, err = _run([str(f)])

    assert code == 0
    assert err == ""


def test_self_path_is_exempt() -> None:
    """The check script itself is allowed to contain the patterns it matches."""
    (self_path,) = _SELF_PATHS
    # _SELF_PATHS is matched against the literal path argument the script
    # receives, not its filesystem location, so passing the configured path
    # exercises the exemption regardless of cwd.
    code, err = _run([self_path])

    assert code == 0, err


def test_directory_path_is_swallowed(tmp_path: Path) -> None:
    """A directory passed as a path is silently skipped (OSError is swallowed)."""
    d = tmp_path / "sub"
    d.mkdir()

    code, err = _run([str(d)])

    assert code == 0
    assert err == ""


def test_mixed_clean_and_dirty_reports_only_dirty(tmp_path: Path) -> None:
    """When several files are scanned, only offending files appear in stderr."""
    clean = tmp_path / "clean.py"
    clean.write_text("x = 1\n", encoding="utf-8")
    dirty = tmp_path / "dirty.py"
    dirty.write_text(f"# refers to {_SAMPLE_ID_SHORT}\n", encoding="utf-8")

    code, err = _run([str(clean), str(dirty)])

    assert code == 1
    assert str(dirty) in err
    assert str(clean) not in err
    assert _SAMPLE_ID_SHORT in err


def test_no_arguments_exits_zero() -> None:
    """An empty path list is a no-op (no errors to report)."""
    code, err = _run([])

    assert code == 0
    assert err == ""

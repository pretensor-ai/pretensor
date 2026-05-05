"""L3 gold-question loader.

The L3 gold corpus lives at ``scripts/data/<dataset>_nl2sql_bench.json`` —
the same file L2's ``query_recall`` reads. The L3 baseline runner only
needs three fields per question (``id``, ``question``, ``expected_sql``),
so this loader strips the rest and returns a sorted list for byte-stable
output ordering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pretensor.benchmark.fixtures import Fixture

__all__ = ["L3GoldEntry", "load_l3_gold"]


@dataclass(frozen=True, slots=True)
class L3GoldEntry:
    """One NL-to-SQL question to grade.

    ``id`` is the stable identifier used in the JSON envelope's
    ``per_item`` list. ``question`` is the natural-language prompt the
    runner sends to the LLM. ``expected_sql`` is the human-authored gold
    query whose result rows define the correct answer.
    """

    id: str
    question: str
    expected_sql: str


def load_l3_gold(fixture: Fixture) -> tuple[Path, bytes, list[L3GoldEntry]]:
    """Read and parse the dataset's NL-to-SQL gold file.

    Returns ``(questions_path, raw_bytes, entries)``. ``raw_bytes`` is
    the file contents exactly as read from disk — exposed so callers can
    fingerprint the fixture (e.g. ``hashlib.sha256``) without re-reading
    the file from a second I/O. ``questions_path`` makes
    ``fixture.questions_path``'s resolved value part of the function's
    contract so callers don't need ``Optional`` narrowing.

    Entries are sorted by ``id`` so the runner's per-item output stays
    byte-stable across re-runs even if the source JSON's array order
    shifts. Raises :class:`FileNotFoundError` if the dataset has no
    question set on disk; raises :class:`ValueError` for malformed
    records.
    """
    if fixture.questions_path is None:
        raise FileNotFoundError(
            f"Dataset {fixture.name.value!r} has no NL-to-SQL gold "
            "questions file checked in (looked for "
            f"scripts/data/{fixture.name.value}_nl2sql_bench.json)."
        )
    questions_path = fixture.questions_path
    raw_bytes = questions_path.read_bytes()
    raw = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"{questions_path}: expected a JSON array, got {type(raw).__name__}."
        )
    entries: list[L3GoldEntry] = []
    for index, record in enumerate(raw):
        if not isinstance(record, dict):
            raise ValueError(
                f"{questions_path}[{index}]: expected an object, got "
                f"{type(record).__name__}."
            )
        try:
            entries.append(
                L3GoldEntry(
                    id=str(record["id"]),
                    question=str(record["question"]),
                    expected_sql=str(record["expected_sql"]),
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"{questions_path}[{index}]: missing field {exc.args[0]!r}."
            ) from exc
    entries.sort(key=lambda e: e.id)
    return questions_path, raw_bytes, entries

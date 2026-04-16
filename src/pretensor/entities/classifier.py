"""Weighted multi-signal table classification for schema intelligence.

Scores naming patterns, FK fan-out/fan-in, column composition, row cardinality,
and optional usage counters. Returns a single :class:`TableClassification` with
role, confidence in ``[0, 1]``, and human-readable signals.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "TABLE_ROLES",
    "TableClassification",
    "TableClassifier",
    "TableClassifierInput",
    "TableRole",
    "is_entity_extraction_candidate",
]

TABLE_ROLES = (
    "system",
    "staging",
    "audit",
    "junction",
    "fact",
    "dimension",
    "bridge",
    "aggregate",
    "snapshot_scd",
    "entity_candidate",
    "unknown",
)

TableRole = Literal[
    "system",
    "staging",
    "audit",
    "junction",
    "fact",
    "dimension",
    "bridge",
    "aggregate",
    "snapshot_scd",
    "entity_candidate",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class TableClassification:
    """Result of :meth:`TableClassifier.classify`."""

    role: TableRole
    confidence: float
    signals: tuple[str, ...] = ()


class TableClassifierInput(BaseModel):
    """Table shape used by :class:`TableClassifier`."""

    name: str = Field(description="Physical table name (no schema).")
    schema_name: str = Field(default="public", description="Schema qualifier.")
    columns: list[str] = Field(
        default_factory=list, description="Column names in declaration order."
    )
    row_count: int | None = Field(default=None, description="Optional row estimate.")
    fk_out_degree: int = Field(
        default=0, ge=0, description="Count of outgoing FK_REFERENCES from this table."
    )
    fk_in_degree: int = Field(
        default=0, ge=0, description="Count of incoming FK_REFERENCES to this table."
    )
    seq_scan_count: int | None = Field(
        default=None, description="Optional sequential scan count (usage signal)."
    )
    idx_scan_count: int | None = Field(
        default=None, description="Optional index scan count (usage signal)."
    )
    insert_count: int | None = Field(default=None)
    update_count: int | None = Field(default=None)


_SNAKE_WORD = re.compile(r"^[a-z][a-z0-9]*$")
_ID_COLUMN = re.compile(r"^id$", re.IGNORECASE)
_SURROGATE_KEY = re.compile(r"^[a-z][a-z0-9_]*_id$", re.IGNORECASE)
_FK_SUFFIX = re.compile(r"_id$", re.IGNORECASE)
_DATE_TOKEN = re.compile(
    r"(^|_)(date|time|timestamp|day|month|year)($|_)", re.IGNORECASE
)


class TableClassifier:
    """Classify a table using weighted heuristics (no graph-wide context)."""

    def classify(self, table: TableClassifierInput) -> TableClassification:
        """Return the highest-scoring role with confidence and contributing signals."""
        name = table.name.strip()
        lower = name.lower()
        cols = table.columns
        weights = self._score_all(table, name, lower, cols)
        best_role, raw = max(weights.items(), key=lambda kv: kv[1])
        second = sorted(weights.values(), reverse=True)
        runner = second[1] if len(second) > 1 else 0.0
        margin = raw - runner
        confidence = self._confidence_from_score(raw, margin)
        signals = self._signals_for_winner(table, lower, cols, best_role)
        return TableClassification(
            role=best_role, confidence=confidence, signals=tuple(signals)
        )

    def _confidence_from_score(self, raw: float, margin: float) -> float:
        """Map unbounded score and margin to ``[0, 1]``."""
        base = 1.0 - math.exp(-raw / 1.4)
        boost = min(0.35, margin * 0.45)
        return max(0.0, min(1.0, base + boost))

    def _score_all(
        self,
        table: TableClassifierInput,
        name: str,
        lower: str,
        cols: list[str],
    ) -> dict[TableRole, float]:
        out: dict[TableRole, float] = {r: 0.0 for r in TABLE_ROLES}
        out["system"] = self._score_system(lower)
        out["staging"] = self._score_staging(lower)
        out["audit"] = self._score_audit(lower)
        j_score, _ = self._junction_signals(name, cols)
        out["junction"] = j_score
        struct = self._score_structure(table, cols)
        for role, v in struct.items():
            out[role] += v
        out["dimension"] += self._score_dimension_name(lower)
        out["fact"] += self._score_fact_name(lower)
        out["aggregate"] += self._score_aggregate_name(lower)
        out["snapshot_scd"] += self._score_snapshot_name(lower)
        out["bridge"] += self._score_bridge_name(lower)
        out["entity_candidate"] += self._score_entity_fallback(cols)
        if max(out.values()) < 0.15:
            out["unknown"] = 0.25
        return out

    def _score_system(self, lower_name: str) -> float:
        if lower_name in ("migrations", "schema_migrations"):
            return 2.5
        if lower_name.startswith("ar_internal_"):
            return 2.5
        if lower_name.startswith("pg_"):
            return 2.5
        return 0.0

    def _score_staging(self, lower_name: str) -> float:
        if (
            lower_name.startswith("stg_")
            or lower_name.startswith("raw_")
            or lower_name.startswith("tmp_")
            or lower_name.startswith("staging_")
        ):
            return 2.2
        return 0.0

    def _score_audit(self, lower_name: str) -> float:
        if lower_name.endswith(("_audit", "_log", "_history", "_changelog")):
            return 2.0
        return 0.0

    def _junction_signals(
        self, name: str, columns: list[str]
    ) -> tuple[float, list[str]]:
        """Score junction pattern; dim_/fact_ names are excluded."""
        sigs: list[str] = []
        lower = name.lower()
        if lower.startswith("dim_") or lower.startswith("fact_"):
            return 0.0, sigs
        parts = name.split("_")
        if len(parts) != 2:
            return 0.0, sigs
        a, b = parts
        if not _SNAKE_WORD.match(a) or not _SNAKE_WORD.match(b):
            return 0.0, sigs
        if any(_ID_COLUMN.match(c.strip()) for c in columns):
            return 0.0, sigs
        n = len(columns)
        if 2 <= n <= 3:
            sigs.append("two-part snake name with 2–3 columns (junction pattern)")
            return 1.8, sigs
        return 0.0, sigs

    def _score_structure(
        self, table: TableClassifierInput, cols: list[str]
    ) -> dict[TableRole, float]:
        fk_out = table.fk_out_degree
        fk_in = table.fk_in_degree
        ncols = len(cols)
        fk_cols = sum(1 for c in cols if _FK_SUFFIX.search(c.strip() or ""))
        date_cols = sum(1 for c in cols if _DATE_TOKEN.search(c.strip() or ""))
        row_count = table.row_count

        scores: dict[TableRole, float] = {
            "fact": 0.0,
            "staging": 0.0,
            "dimension": 0.0,
            "bridge": 0.0,
            "aggregate": 0.0,
            "snapshot_scd": 0.0,
            "entity_candidate": 0.0,
        }
        if fk_out >= 3:
            scores["fact"] += 0.55 + min(0.45, (fk_out - 3) * 0.12)
        elif fk_out == 2:
            scores["fact"] += 0.25
        if fk_in >= 2 and fk_out <= 1:
            scores["dimension"] += 0.4 + min(0.35, (fk_in - 2) * 0.12)
        if fk_out >= 2 and fk_in >= 2:
            scores["bridge"] += 0.85 + min(0.4, (min(fk_out, fk_in) - 2) * 0.1)
        if fk_cols >= 3 and fk_out >= 2:
            scores["fact"] += 0.15
        if ncols >= 2 and fk_cols <= 1 and fk_out <= 1:
            scores["dimension"] += 0.2
        if date_cols >= 2 and fk_out >= 1:
            scores["snapshot_scd"] += 0.5
        if row_count is not None and row_count > 500_000 and fk_out >= 2:
            scores["fact"] += 0.2
        if row_count is not None and row_count < 50_000 and fk_in >= 1 and fk_out <= 1:
            scores["dimension"] += 0.12
        if "total" in " ".join(c.lower() for c in cols) or any(
            c.lower().startswith(("sum_", "avg_", "count_")) for c in cols
        ):
            scores["aggregate"] += 0.55
        scores["entity_candidate"] += 0.08 * min(ncols, 8) / 8.0
        self._apply_usage_scores(table, scores)
        self._apply_seq_idx_ratio(table, scores)
        return scores

    def _apply_usage_scores(
        self, table: TableClassifierInput, scores: dict[TableRole, float]
    ) -> None:
        seq = table.seq_scan_count
        ins = table.insert_count
        upd = table.update_count
        if seq is None and ins is None and upd is None:
            return
        writes = (ins or 0) + (upd or 0)
        reads = seq or 0
        if writes > 500 and reads > 0 and writes > reads * 0.15:
            scores["fact"] += min(0.35, math.log1p(writes) / 25.0)

    def _apply_seq_idx_ratio(
        self, table: TableClassifierInput, scores: dict[TableRole, float]
    ) -> None:
        """Boost fact/staging when sequential scans dominate on a large table."""
        seq = table.seq_scan_count
        idx = table.idx_scan_count
        if seq is None and idx is None:
            return
        s = seq or 0
        i = idx or 0
        total = s + i
        if total <= 0:
            return
        rc = table.row_count
        if rc is None or rc <= 10_000:
            return
        if s / total > 0.9:
            scores["fact"] += 0.15
            scores["staging"] += 0.15

    def _score_dimension_name(self, lower: str) -> float:
        if lower.startswith("dim_") or lower.endswith("_dim") or "_dim_" in lower:
            return 1.1
        return 0.0

    def _score_fact_name(self, lower: str) -> float:
        if lower.startswith("fact_") or lower.startswith("fct_"):
            return 1.1
        if lower.endswith("_fact") or "_fact_" in lower:
            return 0.85
        return 0.0

    def _score_aggregate_name(self, lower: str) -> float:
        if lower.startswith(("agg_", "aggregate_", "summary_", "rollup_")):
            return 1.2
        return 0.0

    def _score_snapshot_name(self, lower: str) -> float:
        if "snapshot" in lower or lower.startswith("snap_") or "_scd" in lower:
            return 1.0
        return 0.0

    def _score_bridge_name(self, lower: str) -> float:
        if (
            lower.startswith("bridge_")
            or "_bridge_" in lower
            or lower.endswith("_bridge")
        ):
            return 1.0
        return 0.0

    def _score_entity_fallback(self, cols: list[str]) -> float:
        for c in cols:
            s = c.strip()
            if _ID_COLUMN.match(s) or _SURROGATE_KEY.match(s):
                return 0.35
        return 0.05

    def _signals_for_winner(
        self,
        table: TableClassifierInput,
        lower: str,
        cols: list[str],
        role: TableRole,
    ) -> list[str]:
        sigs: list[str] = []
        if table.fk_out_degree:
            sigs.append(f"FK out-degree {table.fk_out_degree}")
        if table.fk_in_degree:
            sigs.append(f"FK in-degree {table.fk_in_degree}")
        if table.row_count is not None:
            sigs.append(f"row_count {table.row_count}")
        seq = table.seq_scan_count
        idx = table.idx_scan_count
        rc_sig = table.row_count
        if (
            role in ("fact", "staging")
            and (seq is not None or idx is not None)
            and rc_sig is not None
            and rc_sig > 10_000
        ):
            s, i = seq or 0, idx or 0
            tot = s + i
            if tot > 0 and s / tot > 0.9:
                sigs.append("high seq vs idx scan ratio (large table)")
        j_score, j_sigs = self._junction_signals(table.name.strip(), cols)
        if role == "junction" and j_sigs:
            sigs.extend(j_sigs)
        if role == "dimension" and (lower.startswith("dim_") or "_dim" in lower):
            sigs.append("dimension name prefix/pattern")
        if role == "fact" and (lower.startswith("fact_") or lower.startswith("fct_")):
            sigs.append("fact name prefix")
        if role == "aggregate":
            sigs.append("aggregate naming or measure-like columns")
        if role == "snapshot_scd":
            sigs.append("snapshot / SCD naming or multiple date columns")
        if role == "bridge":
            sigs.append("high FK fan-in and fan-out (bridge pattern)")
        if role == "unknown":
            sigs.append("weak signal mix")
        return sigs[:8] if sigs else ["heuristic winner"]


def is_entity_extraction_candidate(classification: TableClassification) -> bool:
    """Whether a table should be offered to LLM entity grouping."""
    return classification.role in ("entity_candidate", "dimension", "fact")

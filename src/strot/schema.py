"""Phase 1 of the STROT framework: schema-aware context construction.

The :class:`SchemaProfiler` (the paper's profiler ``phi``) inspects a raw
dataset sample and derives a verified structural schema — semantic types,
cardinality, null rates, and representative values — that grounds the
compile-time prompt in actual data characteristics instead of guesses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Mapping, Sequence, Union

from .exceptions import SchemaIntrospectionError

Record = Mapping[str, Any]
DatasetSample = Union[Sequence[Record], Mapping[str, Sequence[Any]]]

_MAX_SAMPLE_VALUES = 5
_CATEGORICAL_CARDINALITY_RATIO = 0.5

_TEMPORAL_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y",
    "%m/%d/%Y",
)


@dataclass(frozen=True)
class FieldProfile:
    """Structural and distributional signature of a single column."""

    name: str
    semantic_type: str  # numerical | temporal | categorical | boolean | text | unknown
    python_type: str
    cardinality: int
    null_rate: float
    sample_values: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "semantic_type": self.semantic_type,
            "python_type": self.python_type,
            "cardinality": self.cardinality,
            "null_rate": round(self.null_rate, 4),
            "sample_values": self.sample_values,
        }


@dataclass(frozen=True)
class DatasetSchema:
    """Verified schema for a dataset sample: the structural context mapping
    that is injected verbatim into the synthesizer prompt."""

    fields: List[FieldProfile]
    row_count: int

    @property
    def headers(self) -> List[str]:
        return [f.name for f in self.fields]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_count": self.row_count,
            "fields": [f.to_dict() for f in self.fields],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class SchemaProfiler:
    """Derives a :class:`DatasetSchema` from a raw sample.

    Accepts either record-oriented data (``[{col: value, ...}, ...]``),
    column-oriented data (``{col: [values...]}``), or any object exposing a
    pandas-style ``to_dict(orient="records")``.
    """

    def profile(self, sample: Any) -> DatasetSchema:
        columns = self._normalize(sample)
        if not columns:
            raise SchemaIntrospectionError("Dataset sample contains no columns to profile.")
        row_count = max(len(values) for values in columns.values())
        fields = [self._profile_field(name, values) for name, values in columns.items()]
        return DatasetSchema(fields=fields, row_count=row_count)

    def _normalize(self, sample: Any) -> Dict[str, List[Any]]:
        if hasattr(sample, "to_dict") and not isinstance(sample, Mapping):
            sample = sample.to_dict(orient="records")
        if isinstance(sample, Mapping):
            if any(not isinstance(v, (list, tuple)) for v in sample.values()):
                raise SchemaIntrospectionError(
                    "Column-oriented samples must map each header to a sequence of values."
                )
            return {str(k): list(v) for k, v in sample.items()}
        if isinstance(sample, Sequence) and not isinstance(sample, (str, bytes)):
            records = list(sample)
            if not records:
                raise SchemaIntrospectionError("Dataset sample is empty; cannot derive a schema.")
            if not all(isinstance(r, Mapping) for r in records):
                raise SchemaIntrospectionError(
                    "Record-oriented samples must be sequences of mappings (dict rows)."
                )
            columns: Dict[str, List[Any]] = {}
            for record in records:
                for key in record:
                    columns.setdefault(str(key), [])
            for record in records:
                for key in columns:
                    columns[key].append(record.get(key))
            return columns
        raise SchemaIntrospectionError(
            f"Unsupported sample type {type(sample).__name__!r}; "
            "expected list of dicts, dict of lists, or a DataFrame-like object."
        )

    def _profile_field(self, name: str, values: Sequence[Any]) -> FieldProfile:
        non_null = [v for v in values if v is not None]
        null_rate = 1.0 - (len(non_null) / len(values)) if values else 1.0
        try:
            distinct = list(dict.fromkeys(non_null))
        except TypeError:  # unhashable values (lists, dicts)
            distinct = non_null
        samples = distinct[:_MAX_SAMPLE_VALUES]
        return FieldProfile(
            name=name,
            semantic_type=self._semantic_type(non_null, len(distinct)),
            python_type=type(non_null[0]).__name__ if non_null else "NoneType",
            cardinality=len(distinct),
            null_rate=null_rate,
            sample_values=samples,
        )

    def _semantic_type(self, non_null: Sequence[Any], cardinality: int) -> str:
        if not non_null:
            return "unknown"
        if all(isinstance(v, bool) for v in non_null):
            return "boolean"
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
            return "numerical"
        if all(isinstance(v, (date, datetime)) for v in non_null):
            return "temporal"
        if all(isinstance(v, str) for v in non_null):
            if all(self._parses_as_temporal(v) for v in non_null):
                return "temporal"
            # Categorical: repeated values whose distinct count stays small
            # relative to the sample (or under an absolute enum-size cap).
            repeats = cardinality < len(non_null)
            small_enum = cardinality <= max(
                20, int(len(non_null) * _CATEGORICAL_CARDINALITY_RATIO)
            )
            if repeats and small_enum:
                return "categorical"
            return "text"
        return "unknown"

    @staticmethod
    def _parses_as_temporal(value: str) -> bool:
        try:
            datetime.fromisoformat(value)
            return True
        except ValueError:
            pass
        for fmt in _TEMPORAL_FORMATS:
            try:
                datetime.strptime(value, fmt)
                return True
            except ValueError:
                continue
        return False


def introspect_schema(sample: Any) -> DatasetSchema:
    """Convenience wrapper around :meth:`SchemaProfiler.profile`."""
    return SchemaProfiler().profile(sample)

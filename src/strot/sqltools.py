"""Pre-coded SQL template tools: the compile agent configures, never authors.

Each tool is a hand-written, parameterized DuckDB SQL template over the table
``records``. At compile time the synthesis agent emits a PLAN of tool calls;
every parameter is validated against the profiled schema (columns must exist,
numeric params must be numeric columns, enums must match) and identifiers are
quoted during expansion - so a plan cannot reference anything outside the
dataset and, ``raw_sql`` aside, cannot produce invalid SQL at all.

Forks extend STROT by adding entries to :data:`SQL_TOOLS`; every escape-hatch
``raw_sql`` call observed in the wild is a candidate for a new template.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .schema import DatasetSchema

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


#: Tool name -> {"params": {param: kind}, "doc": one-line description}.
#: Param kinds: column | numeric_column | int | enum:<a,b,...> | sql
SQL_TOOLS: Dict[str, Dict[str, Any]] = {
    "count_records": {
        "params": {},
        "doc": "Total record count.",
    },
    "column_stats": {
        "params": {"column": "numeric_column"},
        "doc": "count / mean / median / min / max / stddev of a numeric column.",
    },
    "value_counts": {
        "params": {"column": "column"},
        "doc": "Frequency of each distinct value of a column.",
    },
    "group_stats": {
        "params": {"group_by": "column", "metric": "numeric_column"},
        "doc": "Per-group count and metric mean/median/min/max.",
    },
    "group_stats_2d": {
        "params": {"rows": "column", "cols": "column", "metric": "numeric_column"},
        "doc": "Count and metric mean for every rows x cols group combination.",
    },
    "top_k": {
        "params": {"order_by": "numeric_column", "k": "int", "direction": "enum:asc,desc"},
        "doc": "The k records with the lowest (asc) or highest (desc) value.",
    },
    "correlation": {
        "params": {"x": "numeric_column", "y": "numeric_column"},
        "doc": "Pearson correlation between two numeric columns.",
    },
    "raw_sql": {
        "params": {"sql": "sql"},
        "doc": "Escape hatch: one DuckDB SELECT over `records`. Use ONLY when no template fits.",
    },
}


def render_catalog() -> str:
    """Render the tool catalog for the synthesis prompt."""
    lines = []
    for name, spec in SQL_TOOLS.items():
        params = ", ".join(f"{p}: {kind}" for p, kind in spec["params"].items()) or "none"
        lines.append(f"- `{name}` (params: {params}) - {spec['doc']}")
    return "\n".join(lines)


def _validate_param(kind: str, value: Any, schema: DatasetSchema) -> Optional[str]:
    numeric = {f.name for f in schema.fields if f.semantic_type == "numerical"}
    all_columns = {f.name for f in schema.fields}
    if kind == "column":
        if value not in all_columns:
            return f"unknown column {value!r}; schema columns: {sorted(all_columns)}"
    elif kind == "numeric_column":
        if value not in numeric:
            return f"{value!r} is not a numerical column; numerical columns: {sorted(numeric)}"
    elif kind == "int":
        if type(value) is not int or not 1 <= value <= 1000:
            return f"expected an int in [1, 1000], got {value!r}"
    elif kind.startswith("enum:"):
        allowed = kind.split(":", 1)[1].split(",")
        if value not in allowed:
            return f"expected one of {allowed}, got {value!r}"
    elif kind == "sql":
        if not isinstance(value, str) or not value.strip():
            return "sql must be a non-empty string"
        if ";" in value.strip().rstrip(";"):
            return "sql must be a single statement (no internal semicolons)"
        head = value.strip().split(None, 1)[0].lower()
        if head not in ("select", "with"):
            return "sql must be a read-only SELECT (or WITH ... SELECT) statement"
    else:  # pragma: no cover - registry integrity
        return f"unknown param kind {kind!r}"
    return None


def expand_call(call: Mapping[str, Any], schema: DatasetSchema) -> Tuple[str, str]:
    """Validate one plan item and expand it to (insight_name, sql).

    Raises ValueError with a repair-loop-friendly message on any violation.
    """
    if not isinstance(call, Mapping):
        raise ValueError(f"plan items must be objects, got {type(call).__name__}")
    extra = set(call) - {"name", "tool", "params"}
    if extra:
        raise ValueError(f"plan item has unexpected keys {sorted(extra)}")
    name, tool, params = call.get("name"), call.get("tool"), call.get("params", {})
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f"insight name must be snake_case, got {name!r}")
    spec = SQL_TOOLS.get(tool)
    if spec is None:
        raise ValueError(f"unknown tool {tool!r}; available tools: {sorted(SQL_TOOLS)}")
    if not isinstance(params, Mapping):
        raise ValueError(f"{name}: params must be an object")
    if set(params) != set(spec["params"]):
        raise ValueError(
            f"{name}: tool {tool!r} takes exactly params "
            f"{sorted(spec['params'])}, got {sorted(params)}"
        )
    for param, kind in spec["params"].items():
        error = _validate_param(kind, params[param], schema)
        if error:
            raise ValueError(f"{name}: param {param!r}: {error}")
    return name, _render(tool, dict(params))


def _render(tool: str, p: Dict[str, Any]) -> str:
    if tool == "count_records":
        return "SELECT COUNT(*) AS record_count FROM records"
    if tool == "column_stats":
        c = _quote(p["column"])
        return (
            f"SELECT COUNT({c}) AS count, ROUND(AVG({c}), 4) AS mean, "
            f"ROUND(MEDIAN({c}), 4) AS median, MIN({c}) AS min, MAX({c}) AS max, "
            f"ROUND(STDDEV({c}), 4) AS stddev FROM records WHERE {c} IS NOT NULL"
        )
    if tool == "value_counts":
        c = _quote(p["column"])
        return (
            f"SELECT {c} AS value, COUNT(*) AS count FROM records "
            f"GROUP BY {c} ORDER BY count DESC, value ASC"
        )
    if tool == "group_stats":
        g, m = _quote(p["group_by"]), _quote(p["metric"])
        return (
            f"SELECT {g} AS group_value, COUNT(*) AS count, "
            f"ROUND(AVG({m}), 4) AS mean, ROUND(MEDIAN({m}), 4) AS median, "
            f"MIN({m}) AS min, MAX({m}) AS max FROM records "
            f"WHERE {m} IS NOT NULL GROUP BY {g} ORDER BY {g}"
        )
    if tool == "group_stats_2d":
        r, c, m = _quote(p["rows"]), _quote(p["cols"]), _quote(p["metric"])
        return (
            f"SELECT {r} AS row_value, {c} AS col_value, COUNT(*) AS count, "
            f"ROUND(AVG({m}), 4) AS mean FROM records WHERE {m} IS NOT NULL "
            f"GROUP BY {r}, {c} ORDER BY {r}, {c}"
        )
    if tool == "top_k":
        o = _quote(p["order_by"])
        direction = "ASC" if p["direction"] == "asc" else "DESC"
        return (
            f"SELECT * FROM records WHERE {o} IS NOT NULL "
            f"ORDER BY {o} {direction} LIMIT {int(p['k'])}"
        )
    if tool == "correlation":
        x, y = _quote(p["x"]), _quote(p["y"])
        return (
            f"SELECT ROUND(CORR({x}, {y}), 4) AS correlation FROM records "
            f"WHERE {x} IS NOT NULL AND {y} IS NOT NULL"
        )
    if tool == "raw_sql":
        return p["sql"].strip().rstrip(";")
    raise ValueError(f"unknown tool {tool!r}")  # pragma: no cover


def expand_plan(plan: Any, schema: DatasetSchema) -> "Dict[str, str]":
    """Validate a whole PLAN literal and expand it to {insight_name: sql}."""
    if not isinstance(plan, list) or not plan:
        raise ValueError("PLAN must be a non-empty list of tool calls")
    queries: Dict[str, str] = {}
    for item in plan:
        name, sql = expand_call(item, schema)
        if name in queries:
            raise ValueError(f"duplicate insight name {name!r}")
        queries[name] = sql
    return queries

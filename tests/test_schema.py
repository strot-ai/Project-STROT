import pytest

from strot.exceptions import SchemaIntrospectionError
from strot.schema import introspect_schema

RECORDS = [
    {"id": 1, "created_at": "2026-10-01", "status": "open", "note": None},
    {"id": 2, "created_at": "2026-10-02", "status": "open", "note": "call back"},
    {"id": 3, "created_at": "2026-10-03", "status": "closed", "note": "resolved via email thread"},
]


def test_profiles_record_oriented_sample():
    schema = introspect_schema(RECORDS)
    assert schema.row_count == 3
    assert schema.headers == ["id", "created_at", "status", "note"]
    by_name = {f.name: f for f in schema.fields}
    assert by_name["id"].semantic_type == "numerical"
    assert by_name["created_at"].semantic_type == "temporal"
    assert by_name["status"].semantic_type == "categorical"
    assert by_name["status"].cardinality == 2


def test_profiles_column_oriented_sample():
    schema = introspect_schema({"flag": [True, False, True], "score": [0.1, 0.9, 0.5]})
    by_name = {f.name: f for f in schema.fields}
    assert by_name["flag"].semantic_type == "boolean"
    assert by_name["score"].semantic_type == "numerical"


def test_null_rate_and_python_type():
    schema = introspect_schema(RECORDS)
    note = next(f for f in schema.fields if f.name == "note")
    assert note.null_rate == pytest.approx(1 / 3)
    assert note.python_type == "str"


def test_ragged_records_fill_missing_keys_as_null():
    schema = introspect_schema([{"a": 1}, {"a": 2, "b": "x"}])
    b = next(f for f in schema.fields if f.name == "b")
    assert b.null_rate == pytest.approx(0.5)


def test_schema_json_is_prompt_ready():
    payload = introspect_schema(RECORDS).to_json()
    assert '"semantic_type"' in payload and '"null_rate"' in payload


@pytest.mark.parametrize("bad", [[], "not a dataset", [1, 2, 3], {"col": 42}])
def test_rejects_unprofilable_samples(bad):
    with pytest.raises(SchemaIntrospectionError):
        introspect_schema(bad)

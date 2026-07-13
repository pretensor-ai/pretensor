"""Tests for pretensor.integrations._base — no framework dependency required."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch


def test_make_tool_functions_returns_six_callables(tmp_path: Path) -> None:
    from pretensor.integrations._base import make_tool_functions

    fns = make_tool_functions(tmp_path)
    assert len(fns) == 6
    for fn in fns:
        assert callable(fn)


def test_make_tool_functions_names(tmp_path: Path) -> None:
    from pretensor.integrations._base import make_tool_functions

    fns = make_tool_functions(tmp_path)
    names = [fn.__name__ for fn in fns]
    assert names == ["schema", "context", "traverse", "impact", "query", "validate_sql"]


def test_make_tool_functions_all_have_docstrings(tmp_path: Path) -> None:
    from pretensor.integrations._base import make_tool_functions

    for fn in make_tool_functions(tmp_path):
        assert fn.__doc__ and fn.__doc__.strip(), f"{fn.__name__} has no docstring"


def test_schema_fn_signature(tmp_path: Path) -> None:
    from pretensor.integrations._base import make_tool_functions

    schema_fn = make_tool_functions(tmp_path)[0]
    sig = inspect.signature(schema_fn)
    params = list(sig.parameters)
    assert params == ["database", "label"]
    assert sig.parameters["label"].default is None


def test_context_fn_signature(tmp_path: Path) -> None:
    from pretensor.integrations._base import make_tool_functions

    context_fn = make_tool_functions(tmp_path)[1]
    sig = inspect.signature(context_fn)
    params = list(sig.parameters)
    assert params == ["table", "db", "detail"]
    assert sig.parameters["detail"].default == "standard"


def test_traverse_fn_signature(tmp_path: Path) -> None:
    from pretensor.integrations._base import make_tool_functions

    traverse_fn = make_tool_functions(tmp_path)[2]
    sig = inspect.signature(traverse_fn)
    params = list(sig.parameters)
    assert "from_table" in params
    assert "to_table" in params
    assert "database" in params
    assert sig.parameters["max_depth"].default == 4
    assert sig.parameters["top_k"].default == 3


def test_impact_fn_signature(tmp_path: Path) -> None:
    from pretensor.integrations._base import make_tool_functions

    impact_fn = make_tool_functions(tmp_path)[3]
    sig = inspect.signature(impact_fn)
    params = list(sig.parameters)
    assert params == ["table", "database", "column", "max_depth"]
    assert sig.parameters["max_depth"].default == 3


def test_query_fn_signature(tmp_path: Path) -> None:
    from pretensor.integrations._base import make_tool_functions

    query_fn = make_tool_functions(tmp_path)[4]
    sig = inspect.signature(query_fn)
    params = list(sig.parameters)
    assert params == ["q", "db", "limit"]
    assert sig.parameters["limit"].default == 10


def test_validate_sql_fn_signature(tmp_path: Path) -> None:
    from pretensor.integrations._base import make_tool_functions

    validate_fn = make_tool_functions(tmp_path)[5]
    sig = inspect.signature(validate_fn)
    params = list(sig.parameters)
    assert params == ["sql", "database", "dialect"]
    assert sig.parameters["dialect"].default == "postgres"


def test_schema_fn_dispatches_to_payload(tmp_path: Path) -> None:
    from pretensor.integrations._base import make_tool_functions

    with patch(
        "pretensor.integrations._base.schema_payload", return_value={"nodes": [], "edges": []}
    ) as mock_payload:
        schema_fn = make_tool_functions(tmp_path)[0]
        result = schema_fn("mydb", label="Table")

    mock_payload.assert_called_once_with(tmp_path.resolve(), database="mydb", label="Table")
    assert result == {"nodes": [], "edges": []}


def test_context_fn_dispatches_to_payload(tmp_path: Path) -> None:
    from pretensor.integrations._base import make_tool_functions

    with patch(
        "pretensor.integrations._base.context_payload", return_value={"table": "orders"}
    ) as mock_payload:
        context_fn = make_tool_functions(tmp_path)[1]
        result = context_fn("orders", db="mydb", detail="full")

    mock_payload.assert_called_once_with(
        tmp_path.resolve(), table="orders", db="mydb", detail="full"
    )
    assert result == {"table": "orders"}


def test_traverse_fn_dispatches_to_payload(tmp_path: Path) -> None:
    from pretensor.integrations._base import make_tool_functions

    with patch(
        "pretensor.integrations._base.traverse_payload", return_value={"paths": []}
    ) as mock_payload:
        traverse_fn = make_tool_functions(tmp_path)[2]
        result = traverse_fn("orders", "customers", "mydb", edge_types=["fk"])

    mock_payload.assert_called_once_with(
        tmp_path.resolve(),
        from_table="orders",
        to_table="customers",
        database="mydb",
        max_depth=4,
        top_k=3,
        edge_types=("fk",),
        max_inferred_hops=2,
    )
    assert result == {"paths": []}


def test_traverse_fn_none_edge_types(tmp_path: Path) -> None:
    from pretensor.integrations._base import make_tool_functions

    with patch(
        "pretensor.integrations._base.traverse_payload", return_value={}
    ) as mock_payload:
        traverse_fn = make_tool_functions(tmp_path)[2]
        traverse_fn("a", "b", "db")

    call_kwargs = mock_payload.call_args[1]
    assert call_kwargs["edge_types"] is None


def test_impact_fn_dispatches_to_payload(tmp_path: Path) -> None:
    from pretensor.integrations._base import make_tool_functions

    with patch(
        "pretensor.integrations._base.impact_payload", return_value={"hops": {}}
    ) as mock_payload:
        impact_fn = make_tool_functions(tmp_path)[3]
        result = impact_fn("orders", "mydb", column="id")

    mock_payload.assert_called_once_with(
        tmp_path.resolve(), table="orders", database="mydb", column="id", max_depth=3
    )
    assert result == {"hops": {}}


def test_query_fn_dispatches_to_payload(tmp_path: Path) -> None:
    from pretensor.integrations._base import make_tool_functions

    with patch(
        "pretensor.integrations._base.query_payload", return_value={"hits": []}
    ) as mock_payload:
        query_fn = make_tool_functions(tmp_path)[4]
        result = query_fn("customer orders", db="mydb", limit=5)

    mock_payload.assert_called_once_with(
        tmp_path.resolve(), q="customer orders", db="mydb", limit=5
    )
    assert result == {"hits": []}


def test_validate_sql_fn_dispatches_to_payload(tmp_path: Path) -> None:
    from pretensor.integrations._base import make_tool_functions

    with patch(
        "pretensor.integrations._base.validate_sql_payload",
        return_value={"valid": True},
    ) as mock_payload:
        validate_fn = make_tool_functions(tmp_path)[5]
        result = validate_fn("SELECT 1", "mydb", dialect="snowflake")

    mock_payload.assert_called_once_with(
        tmp_path.resolve(), sql="SELECT 1", database="mydb", dialect="snowflake"
    )
    assert result == {"valid": True}


def test_graph_dir_is_resolved(tmp_path: Path) -> None:
    from pretensor.integrations._base import make_tool_functions

    relative = Path(".")
    with patch(
        "pretensor.integrations._base.schema_payload", return_value={}
    ) as mock_payload:
        schema_fn = make_tool_functions(relative)[0]
        schema_fn("db")

    called_graph_dir = mock_payload.call_args[0][0]
    assert called_graph_dir.is_absolute()

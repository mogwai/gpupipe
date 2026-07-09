"""Drift guard for preserved-but-unwired feature code under src/pipe/_planned/.

Nothing live imports pipe._planned and its own test suite (tests/planned/) is
never collected, so without this file the preserved code has zero CI signal —
it would rot silently and re-integration would hit the exact archaeology
PLANNED.md promises to avoid. These tests are cheap and always run.
"""
import ast
import inspect
from pathlib import Path

import pipe._planned.autoscale as planned_autoscale
from pipe.types import WorkerStop  # noqa: F401 — sentinel the autoscaler emits
from pipe.workers import _cpu_chunk, _threaded_worker_run, _worker_run  # noqa: F401


def test_planned_autoscale_imports():
    """Module-level import works and the lazily-imported live names still exist.

    _spawn_additional_worker does `from ..workers import _cpu_chunk,
    _threaded_worker_run, _worker_run` inside the function body, so importing
    the module alone does NOT validate them — the top-level imports above do.
    """
    assert callable(planned_autoscale._autoscaler_thread)
    assert callable(planned_autoscale._spawn_additional_worker)
    assert callable(planned_autoscale._signal_worker_to_stop)


def test_planned_autoscale_worker_args_arity_matches_live_signatures():
    """The hand-built positional args tuples must track the live run-loop signatures.

    _worker_run / _threaded_worker_run grow parameters regularly (cpu_affinity,
    cpu_threads, chunk_eff, chunk_ms were all appended in recent history). A new
    parameter threaded through lifecycle.py but missed in the preserved
    _spawn_additional_worker would otherwise surface only as a TypeError at
    re-integration time.
    """
    source = Path(planned_autoscale.__file__).read_text()
    tree = ast.parse(source)

    spawn_fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_spawn_additional_worker"
    )
    tuple_lengths = sorted(
        len(node.value.elts)
        for node in ast.walk(spawn_fn)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Tuple)
        and any(isinstance(t, ast.Name) and t.id == "args" for t in node.targets)
    )
    assert len(tuple_lengths) == 2, (
        f"expected the threaded and non-threaded args tuples, found {len(tuple_lengths)}"
    )

    expected = sorted([
        len(inspect.signature(_worker_run).parameters),
        len(inspect.signature(_threaded_worker_run).parameters),
    ])
    assert tuple_lengths == expected, (
        f"preserved _spawn_additional_worker builds args tuples of lengths "
        f"{tuple_lengths}, but the live worker run-loops take {expected} "
        f"parameters — pipe/_planned/autoscale.py has drifted from the live "
        f"signatures (update its args tuples; see PLANNED.md)"
    )

"""Tests for the pinned stats line + print-above behavior (pipe.utils / Pipe.log).

Run with: pytest test_stats_display.py -q
"""
import io
import sys

from pipe import Pipe, print_above
from pipe import utils as pu


class FakeTty(io.StringIO):
    def isatty(self):
        return True


def fake_tty(monkeypatch):
    """Route stdout to a fake TTY and reset pinned-line state.

    Must be called INSIDE the test body: pytest's capture manager re-binds
    sys.stdout at each test-phase boundary, so patching from a fixture (setup
    phase) gets clobbered before the call phase starts.
    """
    out = FakeTty()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(pu, "_pinned", None)
    return out


def test_print_above_plain_when_not_tty(capsys):
    print_above("hello")
    assert capsys.readouterr().out == "hello\n"


def test_pin_and_print_above_repaints(monkeypatch):
    out = fake_tty(monkeypatch)
    pu._pin_stats_line("[10] Gen|5/50")
    print_above("worker warning")
    # message printed on its own line, stats line repainted after it
    assert "\r\x1b[Kworker warning\n[10] Gen|5/50" in out.getvalue()
    # pinned frame stays current
    assert pu._pinned == "[10] Gen|5/50"


def test_pin_repaints_in_place(monkeypatch):
    out = fake_tty(monkeypatch)
    pu._pin_stats_line("[1] a")
    pu._pin_stats_line("[2] b")
    # both frames start with clear-line, no newline between them
    assert out.getvalue() == "\r\x1b[K[1] a\r\x1b[K[2] b"


def test_unpin_finalizes_with_newline(monkeypatch):
    out = fake_tty(monkeypatch)
    pu._pin_stats_line("[3] c")
    pu._unpin_stats_line()
    assert out.getvalue().endswith("[3] c\n")
    assert pu._pinned is None
    # unpin twice is harmless
    pu._unpin_stats_line()


def test_print_above_clears_line_even_without_pin(monkeypatch):
    """Worker processes have no pinned state but must still clear a half-drawn
    stats line before printing."""
    out = fake_tty(monkeypatch)
    print_above("from a worker")
    assert out.getvalue() == "\r\x1b[Kfrom a worker\n"


def test_pipe_print_plain_without_stats(capsys):
    pipe = Pipe(stats_interval=0, health_check_interval=0)
    pipe.print("status update")
    assert "status update" in capsys.readouterr().out


def test_log_gated_by_verbose(monkeypatch):
    out = fake_tty(monkeypatch)
    monkeypatch.delenv("PIPE_VERBOSE", raising=False)
    pu._log("hidden")
    assert out.getvalue() == ""
    monkeypatch.setenv("PIPE_VERBOSE", "1")
    pu._log("shown")
    assert "shown" in out.getvalue()

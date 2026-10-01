"""Tests for the HTTP server wrapper in pipe.web.

Run with: pytest test_web.py -q
"""
import io

import pytest
import torch

from pipe import End, Pipe

pytest.importorskip("fastapi")


class Source:
    """Return-based root: emits ids 0..n-1 then End."""
    def __init__(self, n):
        self.n = n
        self._idx = 0

    def __call__(self):
        if self._idx >= self.n:
            return End
        item = {"id": self._idx}
        self._idx += 1
        if self._idx >= self.n:
            return [item, End]
        return item


class TensorSource:
    """Root emitting one item holding a tensor, then End."""
    def __init__(self, n=4):
        self.n = n
        self._done = False

    def __call__(self):
        if self._done:
            return End
        self._done = True
        return [{"id": 0, "t": torch.arange(self.n, dtype=torch.float32)}, End]


def _client(server):
    from fastapi.testclient import TestClient
    return TestClient(server.app)


def test_next_timeout_returns_503():
    """Empty/idle pipe must return 503 (not 500), and the End sentinel is skipped."""
    from pipe.web import PipeServer
    pipe = Pipe(stats_interval=0, health_check_interval=0)
    pipe.add(Source(0), outqn=0)            # emits nothing, just End
    server = PipeServer(pipe, timeout=0.5)
    with _client(server) as client:
        r = client.get("/next")
        assert r.status_code == 503, r.text
        assert server.errors == 0           # a normal idle timeout is not an error


def test_next_serves_item():
    from pipe.web import PipeServer
    pipe = Pipe(stats_interval=0, health_check_interval=0)
    pipe.add(Source(1), outqn=0)
    server = PipeServer(pipe, timeout=5.0)
    with _client(server) as client:
        r = client.get("/next")
        assert r.status_code == 200, r.text
        item = torch.load(io.BytesIO(r.content), weights_only=False)
        assert item == {"id": 0}
        # subsequent request drains End then times out
        assert client.get("/next").status_code == 503


def test_next_serves_tensor_from_payload_store():
    """A tensor big enough for the payload store reaches /next as a real tensor
    (the server serializes the store-backed view, not a handle)."""
    from pipe.web import PipeServer
    n = 1 << 16  # 256 KiB of float32: travels through the shared payload store
    pipe = Pipe(stats_interval=0, health_check_interval=0)
    pipe.add(TensorSource(n), outqn=0)
    server = PipeServer(pipe, timeout=5.0)
    with _client(server) as client:
        r = client.get("/next")
        assert r.status_code == 200, r.text
        item = torch.load(io.BytesIO(r.content), weights_only=False)
        assert torch.equal(item["t"], torch.arange(n, dtype=torch.float32))


def test_health_endpoint():
    from pipe.web import PipeServer
    pipe = Pipe(stats_interval=0, health_check_interval=0)
    pipe.add(Source(1), outqn=0)
    server = PipeServer(pipe, timeout=5.0)
    with _client(server) as client:
        assert client.get("/next").status_code == 200
        h = client.get("/health").json()
        assert h["status"] == "healthy"
        assert h["items_served"] == 1
        assert h["uptime_seconds"] >= 0


def test_stats_endpoint():
    from pipe.web import PipeServer
    pipe = Pipe(stats_interval=0, health_check_interval=0)
    pipe.add(Source(1), outqn=0)
    server = PipeServer(pipe, timeout=5.0)
    with _client(server) as client:
        client.get("/next")
        s = client.get("/stats").json()
        assert s["items_served"] == 1
        assert s["total_bytes_served"] > 0
        assert s["compression"] == "none"

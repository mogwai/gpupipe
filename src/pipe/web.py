"""
Web server wrapper for Pipe objects.

Provides HTTP endpoints to serve items from any Pipe instance,
enabling distributed pipelines where producers and consumers
run on separate machines.

Usage:
    from pipe import Pipe
    from pipe.web import serve_pipe

    pipe = Pipe()
    pipe.add(MyLoader(), workers=2)
    pipe.add(MyProcessor(), workers=4)
    serve_pipe(pipe, port=8000, compression="lz4")
"""

import io
import time
import traceback
from contextlib import asynccontextmanager, suppress
from queue import Empty
from typing import Any

import torch

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

from .pipe import Pipe
from .queues import _InputChannel
from .shm import _item_from_shm
from .types import End



def _validate_compression(compression):
    if compression == "lz4" and not HAS_LZ4:
        raise ImportError("lz4 not installed. Install with: uv pip install lz4")
    if compression not in ("none", "lz4"):
        raise ValueError(f"Invalid compression: {compression}. Must be 'none' or 'lz4'")


def _serialize(item, compression):
    """torch-serialize an item, optionally lz4-compressed. Returns bytes."""
    buffer = io.BytesIO()
    torch.save(item, buffer)
    data = buffer.getvalue()
    if compression == "lz4":
        data = lz4.frame.compress(data, compression_level=0)
    return data


class SerializerWorker:
    """
    Pipe worker that pre-serializes items for efficient network transfer.

    Add as a final Pipe stage to offload serialization from the HTTP handler:
        pipe.add(SerializerWorker(compression="lz4"), workers=2)
    """

    def __init__(self, compression: str = "none"):
        self.compression = compression
        _validate_compression(compression)

    def __call__(self, item):
        serialized = _serialize(item, self.compression)

        return {
            "data": serialized,
            "compression": self.compression,
            "size_bytes": len(serialized),
        }


class PipeServer:
    """FastAPI server that wraps a Pipe and serves its output via HTTP."""

    def __init__(
        self,
        pipe: Pipe,
        compression: str = "none",
        timeout: float = 30.0,
        pre_serialized: bool = False,
    ):
        self.pipe = pipe
        self.compression = compression
        self.timeout = timeout
        self.pre_serialized = pre_serialized

        if not pre_serialized:
            _validate_compression(compression)

        self.items_served = 0
        self.errors = 0
        self.start_time = None
        self.last_item_time = None
        self.total_bytes_served = 0

        self.app = self._create_app()

    def _create_app(self):
        from fastapi import FastAPI, Response
        from fastapi.responses import JSONResponse

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            print("Starting pipe...")
            self.start_time = time.time()
            self.pipe.start()
            print("Pipe started. Server ready at /next")
            yield
            print("Shutting down pipe...")
            self.pipe.stop()
            print("Pipe stopped.")

        app = FastAPI(
            title="Pipe Web Server",
            description="Serves items from a Pipe via HTTP",
            lifespan=lifespan,
        )

        # Chunk-aware reader over the final queue (single-threaded handler).
        self._out_ch = _InputChannel(self.pipe.queues[-1]) if self.pipe.queues else None

        @app.get("/next")
        async def get_next():
            try:
                if self._out_ch is None or self._out_ch.queue is not self.pipe.queues[-1]:
                    self._out_ch = _InputChannel(self.pipe.queues[-1])
                deadline = time.time() + self.timeout
                while True:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise Empty
                    raw = self._out_ch.get(timeout=remaining)
                    item = _item_from_shm(raw)
                    if item is End:
                        continue  # skip completion sentinel, keep waiting for a real item
                    break

                if self.pre_serialized:
                    if not isinstance(item, dict) or "data" not in item:
                        raise ValueError(f"Expected pre-serialized item but got: {type(item)}")
                    serialized = item["data"]
                    compression = item.get("compression", "none")
                    size_bytes = item.get("size_bytes", len(serialized))
                else:
                    serialized = _serialize(item, self.compression)
                    compression = self.compression
                    size_bytes = len(serialized)

                self.items_served += 1
                self.last_item_time = time.time()
                self.total_bytes_served += size_bytes

                return Response(
                    content=serialized,
                    media_type="application/octet-stream",
                    headers={
                        "X-Compression": compression,
                        "X-Items-Served": str(self.items_served),
                        "X-Size-Bytes": str(size_bytes),
                    },
                )
            except (Empty, TimeoutError):
                return JSONResponse(
                    status_code=503,
                    content={"error": "Pipe timeout", "message": f"No item available within {self.timeout}s"},
                )
            except Exception as e:
                self.errors += 1
                return JSONResponse(
                    status_code=500,
                    content={"error": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()},
                )

        @app.get("/health")
        async def health():
            is_alive = all(p.is_alive() for p in self.pipe.processes) if self.pipe.processes else True
            return {
                "status": "healthy" if is_alive else "unhealthy",
                "pipe_alive": is_alive,
                "items_served": self.items_served,
                "errors": self.errors,
                "uptime_seconds": time.time() - self.start_time if self.start_time else 0,
                "last_item_ago_seconds": time.time() - self.last_item_time if self.last_item_time else None,
            }

        @app.get("/stats")
        async def stats():
            uptime = time.time() - self.start_time if self.start_time else 0
            throughput_items = self.items_served / uptime if uptime > 0 else 0
            throughput_bytes = self.total_bytes_served / uptime if uptime > 0 else 0
            stats_dict = {
                "items_served": self.items_served,
                "errors": self.errors,
                "uptime_seconds": uptime,
                "throughput_items_per_sec": throughput_items,
                "throughput_mb_per_sec": throughput_bytes / 1024 / 1024,
                "total_bytes_served": self.total_bytes_served,
                "total_mb_served": self.total_bytes_served / 1024 / 1024,
                "compression": self.compression,
                "timeout": self.timeout,
                "pre_serialized": self.pre_serialized,
            }
            if self.pipe.queues:
                with suppress(Exception):
                    stats_dict["queue_depth"] = self.pipe.queues[-1].qsize()
            return stats_dict

        @app.get("/")
        async def root():
            return {
                "name": "Pipe Web Server",
                "endpoints": {
                    "/next": "Get next item (serialized PyTorch object)",
                    "/health": "Health check",
                    "/stats": "Detailed statistics",
                },
                "compression": self.compression,
            }

        return app

    def start(self, host: str = "0.0.0.0", port: int = 8000, **uvicorn_kwargs: Any):
        import uvicorn
        print(f"Starting Pipe Web Server on {host}:{port}")
        print(f"Compression: {self.compression}, Timeout: {self.timeout}s")
        uvicorn.run(self.app, host=host, port=port, **uvicorn_kwargs)


def serve_pipe(
    pipe: Pipe,
    host: str = "0.0.0.0",
    port: int = 8000,
    compression: str = "none",
    timeout: float = 30.0,
    pre_serialized: bool = False,
    **uvicorn_kwargs: Any,
):
    """
    Convenience function to create and start a PipeServer.

    Example (on-demand serialization):
        pipe = Pipe()
        pipe.add(MyLoader(), workers=2)
        serve_pipe(pipe, port=8000, compression="lz4")

    Example (pre-serialization with workers):
        pipe = Pipe()
        pipe.add(MyLoader(), workers=2)
        pipe.add(SerializerWorker(compression="lz4"), workers=2)
        serve_pipe(pipe, port=8000, pre_serialized=True)
    """
    server = PipeServer(pipe, compression=compression, timeout=timeout, pre_serialized=pre_serialized)
    server.start(host=host, port=port, **uvicorn_kwargs)

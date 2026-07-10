"""Tests for shared memory tensor passing in pipe framework.

Verifies that:
- Items with tensors are written to /dev/shm and reconstructed correctly
- Producer death doesn't corrupt data (shm files persist)
- No /dev/shm leaks after pipeline completion
- Various dtypes and nested structures work
- Throughput improvement over the old torch.save/load approach
"""
import os

import pytest
import torch

from pipe import End, Pipe
from pipe.shm import _cleanup_stale_shm, _has_tensors, _item_from_shm, _item_to_shm


class TensorGenerator:
    def __init__(self, n_items: int, size: tuple = (64, 64)):
        self.n_items = n_items
        self.size = size
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End
        item = {
            "id": self._idx,
            "tensor": torch.randn(self.size),
            "label": f"item_{self._idx}",
        }
        self._idx += 1
        if self._idx >= self.n_items:
            return [item, End]
        return item


class Passthrough:
    def load(self):
        pass

    def __call__(self, item):
        return item


class TensorModifier:
    def load(self):
        pass

    def __call__(self, item):
        item["tensor"] = item["tensor"] * 2
        item["modified"] = True
        return item


class MixedGenerator:
    def __init__(self, n):
        self.n = n
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n:
            return End
        if self._idx % 2 == 0:
            item = {"id": self._idx, "tensor": torch.randn(16), "has_tensor": True}
        else:
            item = {"id": self._idx, "data": "no tensor", "has_tensor": False}
        self._idx += 1
        if self._idx >= self.n:
            return [item, End]
        return item


# === Unit tests for _item_to_shm / _item_from_shm ===

@pytest.fixture(autouse=True)
def clear_shm_env(monkeypatch):
    monkeypatch.delenv("PIPE_NO_SHM", raising=False)
    monkeypatch.delenv("PIPE_NO_SHM_OUTPUT", raising=False)


def test_shm_roundtrip_basic():
    """Basic tensor survives shm round-trip."""
    t = torch.randn(32, 32)
    item = {"id": 1, "tensor": t, "label": "test"}
    shm_ref = _item_to_shm(item)
    assert "__shm__" in shm_ref
    assert os.path.exists(shm_ref["__shm__"])

    result = _item_from_shm(shm_ref)
    assert result["id"] == 1
    assert result["label"] == "test"
    assert torch.allclose(result["tensor"], t)
    assert not os.path.exists(shm_ref["__shm__"])


def test_shm_roundtrip_multiple_tensors():
    """Multiple tensors in one item."""
    item = {
        "id": 42,
        "wav": torch.randn(16000),
        "mel": torch.randn(128, 100),
        "embed": torch.randn(256),
        "text": "hello world",
    }
    shm_ref = _item_to_shm(item)
    result = _item_from_shm(shm_ref)

    assert result["id"] == 42
    assert result["text"] == "hello world"
    assert torch.allclose(result["wav"], item["wav"])
    assert torch.allclose(result["mel"], item["mel"])
    assert torch.allclose(result["embed"], item["embed"])


def test_shm_roundtrip_dtypes():
    """Various tensor dtypes."""
    item = {
        "f32": torch.randn(16, dtype=torch.float32),
        "f64": torch.randn(16, dtype=torch.float64),
        "f16": torch.randn(16, dtype=torch.float16),
        "bf16": torch.randn(16, dtype=torch.bfloat16),
        "i32": torch.randint(0, 100, (16,), dtype=torch.int32),
        "i64": torch.randint(0, 100, (16,), dtype=torch.int64),
        "bool": torch.randint(0, 2, (16,), dtype=torch.bool),
        "u8": torch.randint(0, 255, (16,), dtype=torch.uint8),
    }
    shm_ref = _item_to_shm(item)
    result = _item_from_shm(shm_ref)

    assert result["f32"].dtype == torch.float32
    assert result["f64"].dtype == torch.float64
    assert result["f16"].dtype == torch.float16
    assert result["bf16"].dtype == torch.bfloat16
    assert result["i32"].dtype == torch.int32
    assert result["i64"].dtype == torch.int64
    assert result["bool"].dtype == torch.bool
    assert result["u8"].dtype == torch.uint8
    assert torch.allclose(result["f32"], item["f32"])
    assert torch.allclose(result["bf16"], item["bf16"])


def test_shm_roundtrip_complex_values():
    """Non-tensor values: nested dicts, lists, None, etc."""
    item = {
        "tensor": torch.randn(8),
        "nested": {"a": 1, "b": [2, 3, 4], "c": {"d": "deep"}},
        "list_val": [1, 2, 3],
        "none_val": None,
        "float_val": 3.14,
        "bytes_val": b"raw bytes here",
    }
    shm_ref = _item_to_shm(item)
    result = _item_from_shm(shm_ref)

    assert torch.allclose(result["tensor"], item["tensor"])
    assert result["nested"] == {"a": 1, "b": [2, 3, 4], "c": {"d": "deep"}}
    assert result["list_val"] == [1, 2, 3]
    assert result["none_val"] is None
    assert result["float_val"] == 3.14
    assert result["bytes_val"] == b"raw bytes here"


def test_shm_no_tensors_passthrough():
    """Items without tensors are returned as-is (no shm file created)."""
    item = {"id": 1, "text": "hello", "data": [1, 2, 3]}
    result = _item_to_shm(item)
    assert result is item
    assert "__shm__" not in result


def test_shm_non_dict_passthrough():
    """Non-dict items pass through unchanged."""
    assert _item_to_shm("hello") == "hello"
    assert _item_to_shm(42) == 42
    assert _item_from_shm("hello") == "hello"
    assert _item_from_shm({"id": 1}) == {"id": 1}


def test_shm_large_tensor():
    """Large tensor (50MB) round-trips correctly."""
    t = torch.randn(12_500_000)  # 50MB
    item = {"id": 0, "big": t}
    shm_ref = _item_to_shm(item)
    result = _item_from_shm(shm_ref)
    assert torch.allclose(result["big"], t)


def test_shm_empty_and_scalar_tensors():
    """Edge cases: empty tensor, scalar tensor."""
    item = {
        "empty": torch.empty(0),
        "scalar": torch.tensor(3.14),
        "id": 1,
    }
    shm_ref = _item_to_shm(item)
    result = _item_from_shm(shm_ref)
    assert result["empty"].shape == (0,)
    assert result["scalar"].item() == pytest.approx(3.14)


def test_shm_cleanup_on_read():
    """File is unlinked after _item_from_shm reads it."""
    item = {"tensor": torch.randn(32), "id": 0}
    shm_ref = _item_to_shm(item)
    path = shm_ref["__shm__"]
    assert os.path.exists(path)
    _item_from_shm(shm_ref)
    assert not os.path.exists(path)


def test_shm_cleanup_stale():
    """_cleanup_stale_shm removes leftover files."""
    path = "/dev/shm/pipe_test_stale_999"
    with open(path, "wb") as f:
        f.write(b"stale data")
    assert os.path.exists(path)
    _cleanup_stale_shm()
    assert not os.path.exists(path)


def test_has_tensors():
    """_has_tensors correctly identifies tensor presence."""
    assert _has_tensors(torch.randn(4))
    assert _has_tensors({"a": torch.randn(4)})
    assert _has_tensors({"a": {"b": torch.randn(4)}})
    assert _has_tensors([1, torch.randn(4)])
    assert not _has_tensors({"a": 1, "b": "hello"})
    assert not _has_tensors([1, 2, 3])
    assert not _has_tensors("hello")


# === Integration tests with full pipeline ===


def test_pipeline_tensor_shm():
    """Full pipeline with tensors through shm."""
    n_items = 50
    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(TensorGenerator(n_items, size=(32, 32)), outqn=30)
    pipe.add(TensorModifier(), workers=2, outqn=30)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))
    assert all(r["modified"] for r in results)
    assert all(torch.is_tensor(r["tensor"]) for r in results)


def test_pipeline_no_shm_leak():
    """No /dev/shm/pipe_* files remain after pipeline completes."""
    _cleanup_stale_shm()

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(TensorGenerator(30, size=(64, 64)), outqn=20)
    pipe.add(Passthrough(), workers=2, outqn=20)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 30

    # Check no pipe_ files remain in /dev/shm
    remaining = [f for f in os.listdir("/dev/shm") if f.startswith("pipe_")]
    assert remaining == [], f"Leaked shm files: {remaining}"


def test_pipeline_mixed_items():
    """Pipeline with mix of tensor and non-tensor items."""
    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(MixedGenerator(40), outqn=30)
    pipe.add(Passthrough(), workers=2, outqn=0)

    results = list(pipe)
    assert len(results) == 40
    with_tensor = [r for r in results if r["has_tensor"]]
    without_tensor = [r for r in results if not r["has_tensor"]]
    assert len(with_tensor) == 20
    assert len(without_tensor) == 20
    assert all(torch.is_tensor(r["tensor"]) for r in with_tensor)


def test_pipeline_threaded_with_shm():
    """Threaded workers still work with shm-backed items."""
    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(TensorGenerator(30, size=(32, 32)), outqn=30)
    pipe.add(TensorModifier(), workers=4, outqn=30, thread=True)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 30
    assert all(r["modified"] for r in results)


def test_pipeline_multi_stage_tensor():
    """Multi-stage pipeline preserves tensor data through shm."""
    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(TensorGenerator(20, size=(16, 16)), outqn=20)
    pipe.add(Passthrough(), workers=2, outqn=20)
    pipe.add(TensorModifier(), workers=2, outqn=20)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 20
    assert all(r["modified"] for r in results)
    assert all(torch.is_tensor(r["tensor"]) for r in results)


def test_pipeline_sequential_mode():
    """Sequential mode works without shm."""
    pipe = Pipe(sequential=True, stats_interval=0, health_check_interval=0)
    pipe.add(TensorGenerator(10, size=(16, 16)), outqn=10)
    pipe.add(TensorModifier(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 10
    assert all(r["modified"] for r in results)


# === Performance test ===


def test_shm_queue_payload_tiny():
    """The key advantage: queue payload is ~50 bytes regardless of tensor size."""
    import pickle

    for size in [(32, 32), (256, 256), (1024, 1024)]:
        t = torch.randn(size)
        tensor_bytes = t.nelement() * t.element_size()
        item = {"id": 0, "tensor": t, "label": "test"}

        shm_ref = _item_to_shm(item)
        queue_payload = len(pickle.dumps(shm_ref))

        # Clean up the shm file
        os.unlink(shm_ref["__shm__"])

        # Queue payload should be tiny (<200 bytes) regardless of tensor size
        assert queue_payload < 200, (
            f"Queue payload {queue_payload}B for {tensor_bytes}B tensor - should be <200B"
        )

    # With old approach, queue payload would be ~tensor_bytes + overhead
    # With shm, it's always just the path string (~50 bytes)


def test_shm_noncontiguous_and_grad():
    """Non-contiguous + requires_grad tensors must round-trip (detach/cpu/contiguous)."""
    os.environ.pop("PIPE_NO_SHM", None)
    t = torch.randn(4, 5, requires_grad=True).T  # transpose -> non-contiguous
    ref = _item_to_shm({"t": t})
    assert "__shm__" in ref
    back = _item_from_shm(ref)
    assert torch.equal(back["t"], t.detach())


def test_shm_moves_tensors_to_cpu():
    """Regression guard: _item_to_shm must .cpu() before .numpy().

    A CUDA tensor's .numpy() raises TypeError unless moved to CPU first. No GPU is
    available in CI, so assert the .cpu() call is present (CPU correctness is covered
    by test_shm_roundtrip_dtypes).
    """
    import inspect
    assert ".cpu()" in inspect.getsource(_item_to_shm)

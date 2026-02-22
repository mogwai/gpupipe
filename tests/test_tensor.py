"""PyTorch tensor tests for pipe framework.

The pipe framework automatically serializes/deserializes PyTorch tensors
when passing through multiprocessing queues. Users can return tensors
directly without manual conversion to bytes.
"""
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")

from pipe import Pipe, End


class Collector:
    def load(self):
        pass

    def __call__(self, item):
        return item


class TensorGenerator:
    """Generator that returns tensors directly."""
    def __init__(self, n_items: int, tensor_size: tuple = (64, 128)):
        self.n_items = n_items
        self.tensor_size = tensor_size
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End

        tensor = torch.randn(self.tensor_size)

        item = {
            "id": self._idx,
            "tensor": tensor,
        }
        self._idx += 1
        if self._idx >= self.n_items:
            return [item, End]
        return item


class TensorProcessor:
    """Processor that works with tensors directly."""
    def __init__(self, iterations: int = 10):
        self.iterations = iterations

    def load(self):
        pass

    def __call__(self, item):
        tensor = item["tensor"]
        for _ in range(self.iterations):
            tensor = tensor @ tensor.T
            tensor = torch.softmax(tensor, dim=-1)

        item["tensor"] = tensor
        item["processed"] = True
        item["result_shape"] = tuple(tensor.shape)
        return item


def test_tensor_basic_pipeline():
    """Basic pipeline with tensor processing."""
    n_items = 50

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(TensorGenerator(n_items, tensor_size=(32, 32)), outqn=30)
    pipe.add(TensorProcessor(iterations=3), workers=2, outqn=30)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))
    assert all(r["processed"] for r in results)
    assert all(torch.is_tensor(r["tensor"]) for r in results)


def test_tensor_single_worker():
    """Single worker tensor pipeline."""
    n_items = 30

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(TensorGenerator(n_items, tensor_size=(32, 32)), outqn=50)
    pipe.add(TensorProcessor(iterations=3), workers=1, outqn=50)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_tensor_multi_stage():
    """Multi-stage tensor pipeline."""
    n_items = 40

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(TensorGenerator(n_items, tensor_size=(32, 32)), outqn=50)
    pipe.add(TensorProcessor(iterations=2), workers=2, outqn=50)
    pipe.add(TensorProcessor(iterations=2), workers=2, outqn=50)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_tensor_threaded_workers():
    """Tensor pipeline with threaded workers."""
    n_items = 50

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(TensorGenerator(n_items, tensor_size=(32, 32)), outqn=50)
    pipe.add(TensorProcessor(iterations=3), workers=4, outqn=50, thread=True)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_tensor_large_tensors():
    """Pipeline with larger tensors."""
    n_items = 20

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(TensorGenerator(n_items, tensor_size=(128, 128)), outqn=20)
    pipe.add(TensorProcessor(iterations=2), workers=2, outqn=20)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == n_items
    assert all(torch.is_tensor(r["tensor"]) for r in results)


class NestedTensorGenerator:
    """Generator with deeply nested tensors."""
    def __init__(self, n_items: int):
        self.n_items = n_items
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End

        item = {
            "id": self._idx,
            "nested": {
                "level1": {
                    "level2": {
                        "tensor": torch.randn(16, 16),
                    }
                },
                "list_of_tensors": [torch.randn(8, 8), torch.randn(8, 8)],
                "tuple_of_tensors": (torch.randn(4, 4), torch.randn(4, 4)),
            },
        }
        self._idx += 1
        if self._idx >= self.n_items:
            return [item, End]
        return item


def test_tensor_nested_structures():
    """Tensors in deeply nested dicts/lists/tuples."""
    n_items = 20

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(NestedTensorGenerator(n_items), outqn=30)
    pipe.add(Collector(), workers=2, outqn=0)

    results = list(pipe)
    assert len(results) == n_items

    for r in results:
        assert torch.is_tensor(r["nested"]["level1"]["level2"]["tensor"])
        assert all(torch.is_tensor(t) for t in r["nested"]["list_of_tensors"])
        assert all(torch.is_tensor(t) for t in r["nested"]["tuple_of_tensors"])


class MultipleTensorGenerator:
    """Generator with multiple tensors per item."""
    def __init__(self, n_items: int):
        self.n_items = n_items
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End

        item = {
            "id": self._idx,
            "input": torch.randn(32, 32),
            "mask": torch.randint(0, 2, (32, 32)),
            "weights": torch.randn(32),
            "embeddings": torch.randn(64, 128),
        }
        self._idx += 1
        if self._idx >= self.n_items:
            return [item, End]
        return item


def test_tensor_multiple_per_item():
    """Multiple tensors in a single item."""
    n_items = 30

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(MultipleTensorGenerator(n_items), outqn=30)
    pipe.add(Collector(), workers=2, outqn=0)

    results = list(pipe)
    assert len(results) == n_items

    for r in results:
        assert torch.is_tensor(r["input"])
        assert torch.is_tensor(r["mask"])
        assert torch.is_tensor(r["weights"])
        assert torch.is_tensor(r["embeddings"])
        assert r["input"].shape == (32, 32)
        assert r["embeddings"].shape == (64, 128)


class DtypeGenerator:
    """Generator with various tensor dtypes."""
    def __init__(self, n_items: int):
        self.n_items = n_items
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End

        item = {
            "id": self._idx,
            "float32": torch.randn(16, 16, dtype=torch.float32),
            "float64": torch.randn(16, 16, dtype=torch.float64),
            "float16": torch.randn(16, 16, dtype=torch.float16),
            "int32": torch.randint(0, 100, (16, 16), dtype=torch.int32),
            "int64": torch.randint(0, 100, (16, 16), dtype=torch.int64),
            "bool": torch.randint(0, 2, (16, 16), dtype=torch.bool),
        }
        self._idx += 1
        if self._idx >= self.n_items:
            return [item, End]
        return item


def test_tensor_dtypes():
    """Various tensor dtypes."""
    n_items = 20

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(DtypeGenerator(n_items), outqn=30)
    pipe.add(Collector(), workers=2, outqn=0)

    results = list(pipe)
    assert len(results) == n_items

    for r in results:
        assert r["float32"].dtype == torch.float32
        assert r["float64"].dtype == torch.float64
        assert r["float16"].dtype == torch.float16
        assert r["int32"].dtype == torch.int32
        assert r["int64"].dtype == torch.int64
        assert r["bool"].dtype == torch.bool


class MixedGenerator:
    """Generator with mixed items - some with tensors, some without."""
    def __init__(self, n_items: int):
        self.n_items = n_items
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End

        if self._idx % 2 == 0:
            item = {"id": self._idx, "tensor": torch.randn(16, 16), "has_tensor": True}
        else:
            item = {"id": self._idx, "data": [1, 2, 3], "has_tensor": False}

        self._idx += 1
        if self._idx >= self.n_items:
            return [item, End]
        return item


def test_tensor_mixed_items():
    """Mixed items - some with tensors, some without."""
    n_items = 40

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(MixedGenerator(n_items), outqn=30)
    pipe.add(Collector(), workers=2, outqn=0)

    results = list(pipe)
    assert len(results) == n_items

    with_tensor = [r for r in results if r["has_tensor"]]
    without_tensor = [r for r in results if not r["has_tensor"]]

    assert len(with_tensor) == 20
    assert len(without_tensor) == 20
    assert all(torch.is_tensor(r["tensor"]) for r in with_tensor)
    assert all(r["data"] == [1, 2, 3] for r in without_tensor)


class EdgeCaseGenerator:
    """Generator with edge case tensors."""
    def __init__(self, n_items: int):
        self.n_items = n_items
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End

        base = torch.randn(32, 32)

        item = {
            "id": self._idx,
            "empty": torch.empty(0),
            "scalar": torch.tensor(3.14),
            "view": base[10:20, 10:20],  # Non-contiguous view
            "slice": base[:, 0],  # 1D slice
        }
        self._idx += 1
        if self._idx >= self.n_items:
            return [item, End]
        return item


def test_tensor_edge_cases():
    """Edge case tensors: empty, scalar, views, slices."""
    n_items = 20

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(EdgeCaseGenerator(n_items), outqn=30)
    pipe.add(Collector(), workers=2, outqn=0)

    results = list(pipe)
    assert len(results) == n_items

    for r in results:
        assert torch.is_tensor(r["empty"])
        assert r["empty"].shape == (0,)
        assert torch.is_tensor(r["scalar"])
        assert r["scalar"].dim() == 0
        assert torch.is_tensor(r["view"])
        assert r["view"].shape == (10, 10)
        assert torch.is_tensor(r["slice"])
        assert r["slice"].shape == (32,)

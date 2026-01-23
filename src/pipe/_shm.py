import json
import mmap
import os
import pickle
import struct
import uuid

import numpy as np
import torch

_TORCH_TO_NP = {
    torch.float32: np.float32, torch.float64: np.float64,
    torch.float16: np.float16, torch.bfloat16: np.uint16,
    torch.int32: np.int32, torch.int64: np.int64,
    torch.int16: np.int16, torch.int8: np.int8,
    torch.uint8: np.uint8, torch.bool: np.bool_,
}
_STR_TO_TORCH = {
    "float32": torch.float32, "float64": torch.float64,
    "float16": torch.float16, "bfloat16": torch.bfloat16,
    "int32": torch.int32, "int64": torch.int64,
    "int16": torch.int16, "int8": torch.int8,
    "uint8": torch.uint8, "bool": torch.bool,
}


def _has_tensors(obj):
    if torch.is_tensor(obj):
        return True
    if isinstance(obj, dict):
        return any(_has_tensors(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_tensors(v) for v in obj)
    return False


def _item_to_shm(item):
    """Write an item to /dev/shm. Returns metadata dict for the queue.

    File format: [4B header_len][header_json][field_bytes...]
    - Tensors stored as raw bytes at offsets specified in header
    - Non-tensor values pickled and stored at offsets in header
    - Queue only carries {"__shm__": path} (~50 bytes)
    """
    if not isinstance(item, dict) or not _has_tensors(item):
        return item

    path = f"/dev/shm/pipe_{os.getpid()}_{uuid.uuid4().hex[:12]}"
    header = {}
    chunks = []
    offset = 0

    for key, val in item.items():
        if torch.is_tensor(val):
            val = val.contiguous().detach()
            if val.dtype == torch.bfloat16:
                raw = val.view(torch.uint16).numpy().tobytes()
            else:
                raw = val.numpy().tobytes()
            header[key] = {
                "t": "T",
                "d": str(val.dtype).replace("torch.", ""),
                "s": list(val.shape),
                "o": offset,
                "n": len(raw),
            }
            chunks.append(raw)
            offset += len(raw)
        else:
            raw = pickle.dumps(val, protocol=pickle.HIGHEST_PROTOCOL)
            header[key] = {"t": "P", "o": offset, "n": len(raw)}
            chunks.append(raw)
            offset += len(raw)

    header_bytes = json.dumps(header).encode()
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, struct.pack("<I", len(header_bytes)))
        os.write(fd, header_bytes)
        for chunk in chunks:
            os.write(fd, chunk)
    finally:
        os.close(fd)

    return {"__shm__": path}


def _item_from_shm(item):
    """Read an item from /dev/shm. Returns a regular dict. Unlinks the file."""
    if not isinstance(item, dict) or "__shm__" not in item:
        return item

    path = item["__shm__"]
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
    except Exception:
        os.close(fd)
        raise

    try:
        header_len = struct.unpack("<I", mm[:4])[0]
        header = json.loads(mm[4:4 + header_len])
        data_start = 4 + header_len

        result = {}
        for key, meta in header.items():
            start = data_start + meta["o"]
            raw = mm[start:start + meta["n"]]

            if meta["t"] == "T":
                dtype_str = meta["d"]
                torch_dtype = _STR_TO_TORCH[dtype_str]
                np_dtype = _TORCH_TO_NP[torch_dtype]
                arr = np.frombuffer(bytearray(raw), dtype=np_dtype)
                if dtype_str == "bfloat16":
                    result[key] = torch.from_numpy(arr).view(torch.bfloat16).reshape(meta["s"])
                else:
                    result[key] = torch.from_numpy(arr).reshape(meta["s"])
            else:
                result[key] = pickle.loads(raw)
    finally:
        mm.close()
        os.close(fd)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    return result


def _cleanup_stale_shm():
    """Remove any leftover /dev/shm/pipe_* files from previous crashed runs."""
    try:
        for name in os.listdir("/dev/shm"):
            if name.startswith("pipe_"):
                try:
                    os.unlink(f"/dev/shm/{name}")
                except (OSError, FileNotFoundError):
                    pass
    except Exception:
        pass

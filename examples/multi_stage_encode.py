"""Multi-stage GPU encoding: DB -> download -> chunk -> batch -> GPU encode -> reassemble -> save.

Pattern: Full pipeline with chunking, collation, GPU inference, and reassembly.
From: fluac/encode-pipe.py
"""

import time

import torch
from pipe import Pipe, RetrieveSQL


QUERY = """
SELECT id, audio_url FROM items
WHERE encoded IS NULL
ORDER BY id
"""


def download(item):
    from myproject import storage
    item["audio"] = storage.download(item["audio_url"], memory=True)
    return item


def chunk_audio(item):
    """Split long audio into fixed-size chunks. Returns list of chunk items."""
    from torchcodec.decoders import AudioDecoder
    decoder = AudioDecoder(item["audio"], sample_rate=22050, num_channels=1)
    waveform = decoder.get_all_samples().data.squeeze(0)

    chunk_size = 375 * 2048
    chunks = []
    for i in range(0, len(waveform), chunk_size):
        chunk = waveform[i : i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
        chunks.append({
            "id": item["id"],
            "chunk_idx": len(chunks),
            "total_chunks": -1,
            "wav": chunk.half(),
        })

    for c in chunks:
        c["total_chunks"] = len(chunks)
    return chunks


class Collate:
    """Batch chunks together for efficient GPU processing."""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.buffer = []

    def __call__(self, item):
        self.buffer.append(item)
        if len(self.buffer) >= self.batch_size:
            batch = self.buffer
            self.buffer = []
            return [batch]
        return None

    def flush(self):
        if self.buffer:
            return [self.buffer]
        return []


class Encoder:
    """GPU encoder using torch.compile for max throughput."""

    def __init__(self):
        self.model = None

    def load(self):
        from myproject.model import MyCodec
        self.model = MyCodec.from_pretrained().half().eval().cuda()
        self.model.encode = torch.compile(self.model.encode, mode="max-autotune")

    def __call__(self, batch: list[dict]):
        wavs = torch.stack([it["wav"] for it in batch])[:, None].pin_memory()
        codes = self.model.encode(wavs.to(self.model.device))

        results = []
        for i, item in enumerate(batch):
            del item["wav"]
            item["codes"] = codes[i].cpu()
            results.append(item)
        return results


class Reassembler:
    """Reassemble chunks back into full sequences and save."""

    def __init__(self):
        self.pending: dict[str, dict] = {}
        self.conn = None
        self.t0 = time.perf_counter()
        self.total_seconds = 0

    def load(self):
        import psycopg
        self.conn = psycopg.connect(autocommit=True)

    def __call__(self, item):
        item_id = item["id"]
        if item_id not in self.pending:
            self.pending[item_id] = {"chunks": [], "total": item["total_chunks"]}

        entry = self.pending[item_id]
        entry["chunks"].append((item["chunk_idx"], item["codes"]))

        if len(entry["chunks"]) < entry["total"]:
            return None

        del self.pending[item_id]
        entry["chunks"].sort(key=lambda x: x[0])
        codes = torch.cat([c for _, c in entry["chunks"]], dim=-1)

        output_path = f"/data/encoded/{item_id}.pt"
        torch.save(codes, output_path)

        with self.conn.cursor() as cur:
            cur.execute("UPDATE items SET encoded = true WHERE id = %s", (item_id,))

        self.total_seconds += codes.shape[-1] * 1024 / 22050
        rtf = self.total_seconds / (time.perf_counter() - self.t0)
        print(f"{rtf:.1f}x realtime")
        return item_id


if __name__ == "__main__":
    pipe = Pipe(debug=False, stats_interval=5)
    pipe.add(RetrieveSQL(QUERY, chunk_size=20), outqn=10)
    pipe.add(download, workers=4, thread=True, outqn=20)
    pipe.add(chunk_audio, workers=3, outqn=100)
    pipe.add(Collate(batch_size=8), workers=1, outqn=20)
    pipe.add(Encoder(), pergpu=True, outqn=10)
    pipe.add(Reassembler(), workers=1, outqn=0)

    for item_id in pipe:
        pass

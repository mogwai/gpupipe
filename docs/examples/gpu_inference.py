"""GPU inference pipeline with pergpu and manual batching.

Pattern: source -> download -> batch -> GPU inference -> save
From: akro/esb/align.py, fluac/encode-pipe.py
"""

import torch
from pipe import Pipe


class ItemSource:
    """Source stage: yields items one at a time, returns 'end' when done."""

    def __init__(self, paths: list[str]):
        self.paths = paths
        self.idx = 0

    def __call__(self):
        if self.idx >= len(self.paths):
            return "end"
        item = {"path": self.paths[self.idx], "idx": self.idx}
        self.idx += 1
        return item


def load_audio(item):
    from torchcodec.decoders import AudioDecoder
    decoder = AudioDecoder(item["path"], sample_rate=16000, num_channels=1)
    item["wav"] = decoder.get_all_samples().data.squeeze(0)
    item["duration"] = len(item["wav"]) / 16000
    return item


class Batcher:
    """Collect items into batches. flush() handles the final partial batch."""

    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
        self._buffer = []

    def __call__(self, item):
        self._buffer.append(item)
        if len(self._buffer) >= self.batch_size:
            batch = self._buffer
            self._buffer = []
            return batch
        return None

    def flush(self):
        if self._buffer:
            batch = self._buffer
            self._buffer = []
            return batch
        return []


class GPUEncoder:
    """GPU worker. load() runs once per process on the assigned GPU."""

    def __init__(self):
        self.model = None

    def load(self):
        self.model = torch.hub.load("model_repo", "encoder").cuda().eval()

    def __call__(self, batch: list[dict]):
        wavs = torch.nn.utils.rnn.pad_sequence(
            [item["wav"] for item in batch], batch_first=True
        ).cuda()

        with torch.no_grad():
            embeddings = self.model(wavs)

        for i, item in enumerate(batch):
            del item["wav"]
            item["embedding"] = embeddings[i].cpu()
        return batch


def save_results(batch: list[dict]):
    for item in batch:
        torch.save(item["embedding"], f"/output/{item['idx']}.pt")
    return None


if __name__ == "__main__":
    paths = [f"/data/audio/{i}.wav" for i in range(1000)]

    pipe = Pipe(debug=False, stats_interval=5)
    pipe.add(ItemSource(paths), workers=1, outqn=50)
    pipe.add(load_audio, workers=4, thread=True, outqn=100)
    pipe.add(Batcher(batch_size=16), workers=1, outqn=20)
    pipe.add(GPUEncoder(), pergpu=True, outqn=10)
    pipe.add(save_results, workers=2, outqn=0)

    for _ in pipe:
        pass

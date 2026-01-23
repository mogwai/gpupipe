"""Pipe as a training data loader with shuffle and batching.

Pattern: DB source -> download -> process -> shuffle -> batch -> training loop
"""

import torch
from pipe import Pipe, Batcher, BufferAndShuffle, RetrieveSQL


QUERY = """
SELECT id, audio_url, text FROM samples
WHERE split = 'train'
ORDER BY RANDOM()
"""


def download(item):
    from urllib.request import urlopen
    audio_bytes = urlopen(item["audio_url"]).read()
    item["audio_bytes"] = audio_bytes
    return item


def load_and_process(item):
    from torchcodec.decoders import AudioDecoder
    decoder = AudioDecoder(item.pop("audio_bytes"), sample_rate=16000, num_channels=1)
    wav = decoder.get_all_samples().data.squeeze(0)
    duration = len(wav) / 16000
    if duration < 1.0 or duration > 30.0:
        return None
    item["wav"] = wav
    item["duration"] = duration
    return item


def collate(batch: list[dict]) -> dict:
    wavs = torch.nn.utils.rnn.pad_sequence(
        [item["wav"] for item in batch], batch_first=True
    )
    return {
        "wavs": wavs,
        "texts": [item["text"] for item in batch],
        "ids": [item["id"] for item in batch],
    }


if __name__ == "__main__":
    batch_size = 32

    pipe = Pipe(debug=False, stats_interval=30)
    pipe.add(RetrieveSQL(QUERY, chunk_size=100), outqn=200)
    pipe.add(download, workers=16, thread=True, outqn=200)
    pipe.add(load_and_process, workers=4, outqn=500)
    pipe.add(BufferAndShuffle(size=1000, batch_size=batch_size), outqn=100)
    pipe.add(Batcher(size=batch_size, collate_fn=collate), outqn=10)

    model = ...  # your model
    optimizer = ...  # your optimizer

    for batch in pipe:
        loss = model(batch["wavs"], batch["texts"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

"""Batched obstore downloads and uploads with thread=True and flush().

Pattern: DB source -> threaded batch download -> process -> batched async upload
From: akro/esb/align.py (AudioDownloader, ResultUploader)
"""

import asyncio
import json
import os

from pipe import Pipe, RetrieveSQL

QUERY = """
SELECT id, s3_key FROM items
WHERE processed IS NULL
ORDER BY id
"""


class BatchDownloader:
    """Download files from S3 in async batches. thread=True lets multiple workers run."""

    def __init__(self, buffer_size: int = 32, max_concurrent: int = 128):
        self._buffer_size = buffer_size
        self._max_concurrent = max_concurrent
        self._buffer = []
        self._store = None
        self._loop = None
        self._semaphore = None

    def load(self):
        from obstore.store import S3Store
        self._store = S3Store(
            bucket="my-bucket",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            aws_endpoint=os.environ.get("AWS_ENDPOINT_URL"),
            aws_region="auto",
        )
        self._loop = asyncio.new_event_loop()
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

    async def _download_batch(self, items: list[dict]) -> list[dict]:
        import obstore as obs

        async def download_one(item):
            async with self._semaphore:
                try:
                    result = await obs.get_async(self._store, f"{item['s3_key']}.opus")
                    return {**item, "audio_bytes": bytes(result.bytes())}
                except Exception:
                    return None

        results = await asyncio.gather(*[download_one(it) for it in items])
        return [r for r in results if r is not None]

    def _flush_buffer(self) -> list[dict]:
        if not self._buffer:
            return []
        items = self._buffer
        self._buffer = []
        return self._loop.run_until_complete(self._download_batch(items))

    def __call__(self, item):
        self._buffer.append(item)
        if len(self._buffer) >= self._buffer_size:
            return self._flush_buffer()
        return None

    def flush(self):
        return self._flush_buffer()


def process(item):
    audio_bytes = item.pop("audio_bytes")
    item["size"] = len(audio_bytes)
    item["result"] = {"id": item["id"], "size": item["size"]}
    return item


class BatchUploader:
    """Buffer results and upload in async batches. flush() drains on shutdown."""

    def __init__(self, batch_size: int = 64, max_concurrent: int = 128):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self._buffer = []
        self._store = None
        self._loop = None
        self._semaphore = None
        self.uploaded = 0

    def load(self):
        from obstore.store import S3Store
        self._store = S3Store(
            bucket="my-bucket",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            aws_endpoint=os.environ.get("AWS_ENDPOINT_URL"),
            aws_region="auto",
        )
        self._loop = asyncio.new_event_loop()
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    async def _upload_batch(self, items: list[dict]):
        import obstore as obs

        async def upload_one(item):
            async with self._semaphore:
                key = f"output/{item['id']}.json"
                data = json.dumps(item["result"]).encode()
                await obs.put_async(self._store, key, data)

        await asyncio.gather(*[upload_one(it) for it in items])

    def _flush_buffer(self):
        if not self._buffer:
            return []
        items = self._buffer
        self._buffer = []
        self._loop.run_until_complete(self._upload_batch(items))
        self.uploaded += len(items)
        print(f"Uploaded {self.uploaded} total")
        return items

    def __call__(self, item):
        self._buffer.append(item)
        if len(self._buffer) >= self.batch_size:
            return self._flush_buffer()
        return None

    def flush(self):
        return self._flush_buffer()


if __name__ == "__main__":
    pipe = Pipe(debug=False, stats_interval=5)
    pipe.add(RetrieveSQL(QUERY, chunk_size=100), outqn=200)
    pipe.add(BatchDownloader(buffer_size=32, max_concurrent=128), workers=4, thread=True, outqn=200)
    pipe.add(process, workers=2, outqn=100)
    pipe.add(BatchUploader(batch_size=64, max_concurrent=128), workers=1, outqn=0)

    for _ in pipe:
        pass

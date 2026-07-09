"""Batched async S3 downloader stage (obstore).

The canonical download stage used across data/fluac/akro jobs, distilled.
Instead of hand-rolling a buffer + flush() (the old pattern), let the framework
collect batches with `batch=N` — the worker just downloads one list concurrently.

Throughput knobs:
- batch=64        how many items each __call__ downloads concurrently
- workers=4       parallel downloader processes (each with its own event loop)
- thread=True     fine too for pure-IO; processes isolate the event loops better
  when downstream stages are CPU-heavy

Env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL (e.g. R2).
"""
import os

from pipe import Pipe


def s3_key(path: str) -> str:
    """s3://bucket/some/key -> some/key"""
    return path.split("/", 3)[3]


class S3Downloader:
    """Receives a list of items (batch=N), downloads all their S3 objects
    concurrently, emits items with `data` attached. Failed downloads are
    dropped (returning None from the list filters them)."""

    def __init__(self, bucket: str = "my-bucket"):
        self.bucket = bucket  # __init__ must stay picklable: config only

    def load(self):
        # Heavy/unpicklable state goes here (runs inside the worker process)
        import asyncio

        from obstore.store import S3Store

        self._store = S3Store(
            bucket=self.bucket,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_endpoint=os.environ.get("AWS_ENDPOINT_URL"),
            aws_region="auto",
        )
        self._loop = asyncio.new_event_loop()

    async def _fetch_all(self, items):
        import asyncio

        import obstore as obs

        async def fetch_one(item):
            try:
                result = await obs.get_async(self._store, s3_key(item["s3_path"]))
                item["data"] = bytes(result.bytes())
                return item
            except Exception as e:
                print(f"download failed for {item['s3_path']}: {e}")
                return None

        return await asyncio.gather(*[fetch_one(i) for i in items])

    def __call__(self, batch):
        # batch is a list of up to `batch=` items, collected by the framework —
        # partial batches arrive as-is, no flush() needed.
        results = self._loop.run_until_complete(self._fetch_all(batch))
        return [r for r in results if r is not None]


class KeySource:
    """Toy root worker; in real jobs this is a DB query (see RetrieveSQL)."""

    def __call__(self):
        for i in range(1000):
            yield {"id": i, "s3_path": f"s3://my-bucket/audio/{i}.flac"}


class Sink:
    def __call__(self, item):
        return {"id": item["id"], "size": len(item["data"])}


if __name__ == "__main__":
    pipe = Pipe(stats_interval=3)
    pipe.add(KeySource(), outqn=500)
    pipe.add(S3Downloader(bucket="my-bucket"), workers=4, batch=64, outqn=200)
    pipe.add(Sink(), workers=1, outqn=0)

    for result in pipe:
        pass

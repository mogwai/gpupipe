"""Downloader with capped retries via worker.push() — runnable as-is.

The basics of the pattern behind the real YouTube/S3 download jobs:
- classify failures: permanent -> drop the item (return None),
  transient -> send it BACK through the pipeline for another attempt
- `self.push(stage, item)` puts an item on an earlier stage's input queue;
  pushing to your own stage retries it (framework injects push on every
  non-root worker)
- cap attempts on the item itself so the retry cycle always terminates

Run: python retry_downloader.py
"""
import random

from pipe import Pipe


class UrlSource:
    def __call__(self):
        for i in range(50):
            yield {"id": i, "url": f"https://example.com/file_{i}", "attempts": 0}


class Downloader:
    MAX_ATTEMPTS = 3

    def _download(self, url):
        # Stand-in for requests/yt-dlp/obstore. Fails transiently ~30% of the
        # time and permanently for one specific item.
        if url.endswith("_13"):
            raise ValueError("404 not found")          # permanent
        if random.random() < 0.3:
            raise TimeoutError("connection timed out")  # transient
        return b"x" * 1024

    def __call__(self, item):
        item["attempts"] += 1
        try:
            item["data"] = self._download(item["url"])
            return item                                  # success -> downstream
        except TimeoutError:
            if item["attempts"] < self.MAX_ATTEMPTS:
                self.push(Downloader, item)              # retry: back onto our own input
                return None                              # nothing downstream this pass
            print(f"[{item['id']}] gave up after {item['attempts']} attempts")
            return None                                  # drop
        except ValueError as e:
            print(f"[{item['id']}] permanent failure: {e}")
            return None                                  # drop, no retry


class Save:
    def __call__(self, item):
        return {"id": item["id"], "attempts": item["attempts"], "bytes": len(item["data"])}


if __name__ == "__main__":
    pipe = Pipe(stats_interval=0)
    pipe.add(UrlSource(), outqn=100)
    pipe.add(Downloader(), workers=4, outqn=100)
    pipe.add(Save(), workers=1, outqn=0)

    results = list(pipe)
    retried = [r for r in results if r["attempts"] > 1]
    print(f"downloaded {len(results)}/50 ({len(retried)} needed retries)")

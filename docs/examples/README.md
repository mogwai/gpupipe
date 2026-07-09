# Examples

Runnable as-is (stdlib + torch only):

| Example | Shows |
|---------|-------|
| [retry_downloader.py](retry_downloader.py) | Capped retries via `worker.push()`; permanent vs transient failure handling |
| [group_by_key.py](group_by_key.py) | Stateful grouping stage (`workers=1`) with `flush()` for the stream tail |
| [ddp_training.py](ddp_training.py) | One shared pipeline feeding N DDP ranks: `expected_consumers` + `PipeIterator` |

Templates (need your own S3 bucket / DB — adapt the config):

| Example | Shows |
|---------|-------|
| [s3_downloader.py](s3_downloader.py) | Streaming S3 downloads: `AsyncPoolWorker` keeps N GETs in flight, no batch barrier |
| [basic_download.py](basic_download.py) | DB query → threaded downloads → process → DB update |
| [batch_s3_upload.py](batch_s3_upload.py) | Batched obstore download + upload with `thread=True` and `flush()` |
| [gpu_inference.py](gpu_inference.py) | GPU inference with `pergpu=True` and batching |
| [multi_stage_encode.py](multi_stage_encode.py) | Full audio pipeline: download → chunk → GPU encode → reassemble → save |
| [training_dataloader.py](training_dataloader.py) | Pipe as a training dataloader: `BufferAndShuffle` + `Batcher` |

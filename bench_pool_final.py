import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from vllm.model_executor.layers.pooler.seqwise.mean_pool import segmented_mean
from vllm.model_executor.layers.pooler.seqwise.methods import MeanPool
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingCursor, PoolingMetadata, PoolingStates


def measure(fn):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(100):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / 100)
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(42)
    shapes = [(1,32,384), (16,32,384), (64,32,384), (128,32,384),
              (64,128,384), (64,512,384), (1,512,768), (64,512,768),
              (128,128,768), (1,32768,768), (4,8192,768), (1,4096,4096)]
    rows = []
    stock = MeanPool()
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for batch, length, width in shapes:
            lengths = torch.full((batch,), length, dtype=torch.int64)
            offsets = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0))).cuda()
            starts, ends = offsets[:-1], offsets[1:] - 1
            x = torch.randn(batch * length, width, device='cuda', dtype=dtype)
            cursor = PoolingCursor(starts, ends, lengths, lengths, lengths, False, [True]*batch)
            metadata = PoolingMetadata(lengths, None, None,
                                      [PoolingParams(task='embed') for _ in range(batch)],
                                      [PoolingStates() for _ in range(batch)], cursor)
            reference = x.reshape(batch, length, width).double().mean(1).float()
            variants = {'stock': lambda: stock(x, metadata),
                        'segment_reduce': lambda: torch.segment_reduce(
                            x.float(), 'mean', offsets=offsets, unsafe=True),
                        'triton': lambda: segmented_mean(x, starts, ends)}
            for name, fn in variants.items():
                result = fn()
                torch.testing.assert_close(result, reference, rtol=1e-4, atol=2e-6)
                max_abs_error = (result-reference).abs().max().item()
                del result
                samples = measure(fn)
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                result = fn()
                torch.cuda.synchronize()
                extra_peak = torch.cuda.max_memory_allocated() - before
                del result
                row = dict(dtype=str(dtype), batch=batch, length=length, width=width,
                           variant=name, wall_us=samples,
                           median_wall_us=statistics.median(samples),
                           max_abs_error=max_abs_error,
                           extra_peak_allocated_bytes=extra_peak)
                print(json.dumps(row), flush=True)
                rows.append(row)
                args.output.write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()

# CUDA MEAN-pooling measurements

These files accompany the short-sequence CUDA MEAN-pooling contribution. Measurements use vLLM base `4643f41853146eb65d03e94538a09643e33710ad`, PyTorch 2.14.0+cu130 and Triton 3.8.0 on one NVIDIA L4. The production kernel and final method snapshots are included. AI assisted the investigation, code and measurements; the contribution was human-directed and reviewed.

The final end-to-end results are in `offline-guarded-final/`. Three pairs of fresh processes ran in AB/BA/AB order, with five measured calls per process. The median paired gain was 1.34% (range 0.85–1.52%). All individual calls, source hashes, commands and the separate first-pair profiles are retained.

`micro-dispatch-final.json` measures the full guarded pooler call, an explicitly loaded baseline, and the direct registered op. It includes the CPU length check. `micro-final.json` is the earlier direct-kernel comparison with stock and `torch.segment_reduce`; its long-sequence regressions motivated retaining the baseline beyond 512 tokens. The library comparator receives precomputed offsets and casts the whole input to FP32, unlike the incumbent's 16 MiB chunk cap.

## Reproduction

Use a CUDA machine. The scripts expect the candidate checkout at `/opt/vllm-dev` and this artifact branch at `/workspace/evidence`. The evidence branch is separate from the PR branch so benchmark outputs do not enter the production diff.

```bash
git clone --branch perf/segmented-mean-pool https://github.com/samir-sarwar/vllm.git /opt/vllm-dev
git clone --single-branch --branch mean-pool-evidence-20261022 https://github.com/samir-sarwar/vllm.git /workspace/evidence
cd /opt/vllm-dev
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=4643f41853146eb65d03e94538a09643e33710ad \
  uv pip install -e . --torch-backend=cu130

.venv/bin/python /workspace/evidence/run_offline_matrix.py --lengths 512 --label reproduction-01
.venv/bin/python /workspace/evidence/bench_pool_dispatch.py --output /workspace/evidence/reproduction-micro.json
```

Choose a fresh output label. The runner swaps `methods.py` between the included baseline and candidate snapshots, executes each arm in a fresh process, and restores the candidate in `finally`. Run it in a disposable checkout and keep the GPU exclusive to this experiment.

The model is `intfloat/e5-small`, revision `e272f3049e853b47cb5ca3952268c6662abda68f`, FP16. Each timed call embeds 8,192 seeded random-token inputs of 512 tokens through `LLM.embed`. Loading is excluded; request handling and output conversion are included. Prefix caching is disabled, sequence cap 128, token cap 32,768. Two calls using 512 inputs warm up each process before five full timed calls. The first pair also records a separate, untimed profiler call.

`packages-final.txt` records the complete measured environment. Test dependencies are not installed by the minimal benchmark commands above; use the repository's contribution guide and `requirements/test/cuda.in` to reproduce pytest validation.

## Correctness

- `unit-guarded.log`: 59 passed, including ragged/strided FP16/BF16/FP32, FP64 references, opcheck and the 512/513 dispatch boundary.
- `unit-guarded-flag-on.log`: 13 passed with batch-invariant mode enabled.
- `mteb-e5-small.log`: STS12 passed with the same registered kernel before the length guard. Score difference from reference: 4.67e-6.
- `model-parity.json`: final 64-text comparison, finite outputs, maximum absolute error 8.64e-5, minimum cosine 0.9999998463.
- `pre-commit-final.log`: repository hooks passed for the final code.

These results support the stated L4 workload. They do not establish a general serving-throughput improvement, a full MTEB pass, or physical ROCm validation. H100 exploration was inconclusive and is excluded from the performance claim.

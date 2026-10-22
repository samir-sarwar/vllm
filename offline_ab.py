"""Offline embedding timing; run each A/B case in a fresh process."""

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", default="intfloat/e5-small")
    parser.add_argument("--revision", default="e272f3049e853b47cb5ca3952268c6662abda68f")
    parser.add_argument("--lengths", default="32")
    parser.add_argument("--num-prompts", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    if min(lengths) < 2 or max(lengths) > args.max_model_len:
        parser.error("lengths must be between 2 and max-model-len")
    if min(args.num_prompts, args.max_num_seqs, args.repeats) < 1:
        parser.error("num-prompts, max-num-seqs and repeats must be positive")

    import vllm
    from vllm import LLM

    kwargs = {}
    if args.profile_dir:
        kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.profile_dir.resolve()),
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": True,
        }
    llm = LLM(
        model=args.model,
        revision=args.revision,
        runner="pooling",
        dtype=args.dtype,
        seed=args.seed,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.5,
        **kwargs,
    )
    tokenizer = llm.get_tokenizer()
    special_ids = set(tokenizer.all_special_ids)
    content_ids = [i for i in range(tokenizer.vocab_size) if i not in special_ids]
    first = tokenizer.cls_token_id
    if first is None:
        first = tokenizer.bos_token_id
    last = tokenizer.sep_token_id
    if last is None:
        last = tokenizer.eos_token_id
    first = first if first is not None else content_ids[0]
    last = last if last is not None else content_ids[-1]
    rng = random.Random(args.seed)
    prompts = []
    for i in range(args.num_prompts):
        length = lengths[i % len(lengths)]
        ids = [first, *rng.choices(content_ids, k=length - 2), last]
        prompts.append({"prompt_token_ids": ids})
    token_count = sum(len(prompt["prompt_token_ids"]) for prompt in prompts)
    prompt_sha = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    source_dir = Path(vllm.__file__).parent / "model_executor/layers/pooler/seqwise"
    source_hashes = {
        name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        for name in ("methods.py", "mean_pool.py")
        if (source_dir / name).exists()
    }
    metadata = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "vllm_version": vllm.__version__,
        "source_sha256": source_hashes,
        "prompt_sha256": prompt_sha,
        "total_input_tokens": token_count,
    }
    print("OFFLINE_CONFIG " + json.dumps(metadata), flush=True)
    warmup_prompts = prompts[: min(len(prompts), args.max_num_seqs * 4)]
    for _ in range(args.warmups):
        llm.embed(warmup_prompts, use_tqdm=False)

    runs = []
    for repeat in range(args.repeats):
        start = time.perf_counter()
        outputs = llm.embed(prompts, use_tqdm=False)
        elapsed = time.perf_counter() - start
        assert len(outputs) == len(prompts)
        assert all(output.outputs.embedding for output in outputs)
        result = {
            "repeat": repeat,
            "elapsed_seconds": elapsed,
            "embeddings_per_second": len(outputs) / elapsed,
            "input_tokens_per_second": token_count / elapsed,
        }
        runs.append(result)
        print("OFFLINE_RUN " + json.dumps(result), flush=True)
        del outputs
    if args.profile_dir:
        llm.start_profile()
        llm.embed(warmup_prompts, use_tqdm=False)
        llm.stop_profile()
    result = {
        "config": metadata,
        "runs": runs,
        "median_embeddings_per_second": statistics.median(
            run["embeddings_per_second"] for run in runs
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("OFFLINE_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

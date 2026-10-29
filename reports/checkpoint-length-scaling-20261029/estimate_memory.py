"""Estimate Qwen3-0.6B LoRA-r16 allocated memory using the measured equations."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--length", type=int, required=True, help="Padded sequence length in tokens")
    parser.add_argument("--checkpoint", choices=("on", "off"), required=True)
    parser.add_argument("--effective-batch", type=int, default=32)
    args = parser.parse_args()
    if min(args.microbatch, args.length, args.effective_batch) <= 0:
        parser.error("Batch sizes and length must be positive")
    if args.effective_batch % args.microbatch:
        parser.error("Effective batch must be divisible by physical microbatch")
    data = json.loads((ROOT / "memory-fit.json").read_text())
    fit = data[str(args.checkpoint == "on")]
    a, b, c = fit["coefficients_GiB"]
    x = args.microbatch * args.length / 16384
    y = (args.microbatch / 16) * (args.length / 1024) ** 2
    report = {"model": "Qwen3-0.6B SEQ_CLS LoRA r16/alpha32 BF16",
              "checkpointing": args.checkpoint == "on", "physical_microbatch": args.microbatch,
              "padded_sequence_length": args.length, "resident_padded_tokens": args.microbatch * args.length,
              "effective_batch": args.effective_batch, "accumulation": args.effective_batch // args.microbatch,
              "estimated_peak_allocated_GiB": round(a + b * x + c * y, 3),
              "memory_measurement": "Forward/backward with gradient accumulation; allocator reserved and CUDA context separate"}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

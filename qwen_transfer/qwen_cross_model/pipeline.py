#!/usr/bin/env python3
"""Run each stage in a subprocess so source and target models do not share VRAM."""

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Source SuperPos -> RFA refinement -> target evaluation")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--source-checkpoint", help="Skip source training and use this checkpoint")
    p.add_argument("--skip-evaluation", action="store_true")
    p.add_argument("--num-probes", type=int, default=64)
    p.add_argument("--refine-epochs", type=int, default=5)
    p.add_argument("--stay-weight", type=float, default=1.0)
    p.add_argument("--anchor-mode", choices=["full", "source-effect"], default="full")
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", default=None, help="For example: auto")
    return p.parse_args(argv)


def _run(arguments):
    print("+", " ".join(map(str, arguments)), flush=True)
    subprocess.run(arguments, check=True)


def run(args):
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_checkpoint = (Path(args.source_checkpoint).expanduser().resolve()
                         if args.source_checkpoint else root / "source" / "source_prompt.pt")
    if not args.source_checkpoint:
        _run([sys.executable, "-m", "qwen_cross_model.train_superpos",
              "--output-dir", str(root / "source"), "--dtype", args.dtype])
    refined = root / "refinement" / "refined_target.pt"
    refine_command = [sys.executable, "-m", "qwen_cross_model.refine_target",
          "--source-checkpoint", str(source_checkpoint), "--output-dir", str(root / "refinement"),
          "--num-probes", str(args.num_probes), "--epochs", str(args.refine_epochs),
          "--stay-weight", str(args.stay_weight), "--anchor-mode", args.anchor_mode,
          "--dtype", args.dtype, "--device", args.device]
    if args.device_map:
        refine_command.extend(["--device-map", args.device_map])
    _run(refine_command)
    if not args.skip_evaluation:
        evaluation_command = [sys.executable, "-m", "qwen_cross_model.evaluate",
              "--source-checkpoint", str(source_checkpoint), "--refined-checkpoint", str(refined),
              "--output-dir", str(root / "evaluation"), "--dtype", args.dtype,
              "--device", args.device]
        if args.device_map:
            evaluation_command.extend(["--device-map", args.device_map])
        _run(evaluation_command)
    return refined


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()

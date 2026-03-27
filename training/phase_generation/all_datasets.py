from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CHART phase labels for all datasets")
    parser.add_argument("--root", type=str, required=True, help="Dataset root folder")
    parser.add_argument(
        "--output_mode",
        type=str,
        choices=["inside", "separate"],
        default="inside",
        help="'inside': write to <root>/<dataset>/phases, 'separate': write to <out_root>/<dataset>",
    )
    parser.add_argument("--out_root", type=str, default=None, help="Output root when --output_mode separate")
    parser.add_argument("--fps", type=int, default=100, help="Target frame rate for phase targets")
    args = parser.parse_args()

    if args.output_mode == "separate" and args.out_root is None:
        raise ValueError("--out_root is required when --output_mode separate")

    modules = [
        "training.phase_generation.ballroom",
        "training.phase_generation.beatles",
        "training.phase_generation.gtzan",
        "training.phase_generation.hains",
        "training.phase_generation.rwc_popular",
    ]

    for module in modules:
        command = [
            sys.executable,
            "-m",
            module,
            "--root",
            args.root,
            "--output_mode",
            args.output_mode,
            "--fps",
            str(args.fps),
        ]
        if args.out_root is not None:
            command.extend(["--out_root", args.out_root])
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

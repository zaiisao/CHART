from __future__ import annotations

from pathlib import Path

from training.phase_generation.common import (
    build_standard_parser,
    convert_dataset_to_phase_npy,
    parse_two_column_annotation,
    print_stats,
    resolve_output_dir,
)


def main() -> None:
    parser = build_standard_parser("Generate CHART phase labels for beatles dataset")
    args = parser.parse_args()

    root = Path(args.root)
    data_dir = root / "beatles" / "data"
    label_dir = root / "beatles" / "label"
    out_dir = resolve_output_dir(root, args.out_root, args.output_mode, "beatles")

    def resolve_annotation(audio_path: Path) -> Path:
        stem = audio_path.stem.replace("_L+R", "")
        annot = label_dir / f"{stem}.txt"
        if not annot.is_file():
            raise FileNotFoundError(str(annot))
        return annot

    stats = convert_dataset_to_phase_npy(
        data_dir=data_dir,
        output_dir=out_dir,
        audio_patterns=["*L+R.wav", "*.wav"],
        annotation_resolver=resolve_annotation,
        annotation_parser=parse_two_column_annotation,
        fps=args.fps,
    )
    print_stats("beatles", stats, out_dir)


if __name__ == "__main__":
    main()

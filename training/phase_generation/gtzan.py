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
    parser = build_standard_parser("Generate CHART phase labels for gtzan dataset")
    args = parser.parse_args()

    root = Path(args.root)
    data_dir = root / "gtzan" / "data"
    label_dir = root / "gtzan" / "label"
    out_dir = resolve_output_dir(root, args.out_root, args.output_mode, "gtzan")

    def resolve_annotation(audio_path: Path) -> Path:
        stem = audio_path.stem
        annot_candidates: list[Path] = []

        annot_candidates.append(label_dir / f"{stem}.beats")

        parts = stem.split("_", maxsplit=1)
        if len(parts) == 2 and "." in parts[1]:
            genre, index = parts[1].split(".", maxsplit=1)
            annot_candidates.append(label_dir / f"gtzan_{genre}_{index}.beats")

        annot = next((candidate for candidate in annot_candidates if candidate.is_file()), None)
        if annot is None:
            raise FileNotFoundError(f"No GTZAN annotation found for {audio_path.name}")
        return annot

    stats = convert_dataset_to_phase_npy(
        data_dir=data_dir,
        output_dir=out_dir,
        audio_patterns=["*.wav"],
        annotation_resolver=resolve_annotation,
        annotation_parser=parse_two_column_annotation,
        fps=args.fps,
    )
    print_stats("gtzan", stats, out_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torchaudio
except Exception:
    torchaudio = None


@dataclass(frozen=True)
class PhaseGenerationStats:
    total_audio_files: int
    converted_files: int
    skipped_missing_annotation: int
    skipped_parse_error: int


def _audio_duration_seconds(audio_path: Path) -> float:
    if torchaudio is None:
        raise RuntimeError("torchaudio is required to infer audio duration for phase generation.")
    info_fn = getattr(torchaudio, "info", None)
    if info_fn is None:
        raise RuntimeError("torchaudio.info is unavailable in this environment.")
    metadata = info_fn(str(audio_path))
    if metadata.sample_rate <= 0:
        raise ValueError(f"Invalid sample_rate for {audio_path}")
    return float(metadata.num_frames) / float(metadata.sample_rate)


def _build_phase_array(
    beat_times_sec: np.ndarray,
    beat_indices: np.ndarray,
    duration_sec: float,
    fps: int,
) -> np.ndarray:
    if beat_times_sec.ndim != 1 or beat_indices.ndim != 1:
        raise ValueError("beat_times_sec and beat_indices must be 1D arrays.")
    if beat_times_sec.shape[0] != beat_indices.shape[0]:
        raise ValueError("beat_times_sec and beat_indices must have equal length.")
    if beat_times_sec.shape[0] < 2:
        raise ValueError("At least 2 beats are required to build phase targets.")

    beat_order = np.argsort(beat_times_sec)
    beat_times = beat_times_sec[beat_order]
    beat_ids = beat_indices[beat_order]

    keep_mask = np.ones(beat_times.shape[0], dtype=bool)
    keep_mask[1:] = beat_times[1:] > beat_times[:-1]
    beat_times = beat_times[keep_mask]
    beat_ids = beat_ids[keep_mask]

    if beat_times.shape[0] < 2:
        raise ValueError("Not enough strictly increasing beats after deduplication.")

    n_frames = max(1, int(np.ceil(duration_sec * fps)))
    frame_times = np.arange(n_frames, dtype=np.float64) / float(fps)

    next_idx = np.searchsorted(beat_times, frame_times, side="right")
    prev_idx = np.clip(next_idx - 1, 0, beat_times.shape[0] - 2)
    nxt_idx = np.clip(prev_idx + 1, 1, beat_times.shape[0] - 1)

    t0 = beat_times[prev_idx]
    t1 = beat_times[nxt_idx]
    interval = np.maximum(t1 - t0, 1e-6)

    beat_phase = np.clip((frame_times - t0) / interval, 0.0, 0.999999)
    tempo = 1.0 / np.maximum(interval * fps, 1e-6)

    meter = int(max(1, np.max(beat_ids)))
    beat_base = (np.maximum(beat_ids[prev_idx], 1) - 1).astype(np.float64)
    bar_phase = np.mod((beat_base + beat_phase) / float(meter), 1.0)

    phase = np.stack([tempo, beat_phase, bar_phase], axis=1).astype(np.float32)
    return phase


def parse_two_column_annotation(annotation_path: Path) -> tuple[np.ndarray, np.ndarray]:
    beat_times: list[float] = []
    beat_indices: list[int] = []

    with annotation_path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            parts = line.strip().replace("\t", " ").split()
            if len(parts) < 2:
                continue
            beat_times.append(float(parts[0]))
            beat_indices.append(max(1, int(float(parts[1]))))

    if len(beat_times) == 0:
        raise ValueError(f"No beats parsed from {annotation_path}")

    return np.asarray(beat_times, dtype=np.float64), np.asarray(beat_indices, dtype=np.int64)


def parse_rwc_annotation(annotation_path: Path) -> tuple[np.ndarray, np.ndarray]:
    beat_times: list[float] = []
    codes: list[int] = []

    with annotation_path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            code = int(parts[2])
            if code <= 0:
                continue
            beat_times.append(float(int(parts[0])) / 100.0)
            codes.append(code)

    if len(beat_times) == 0:
        raise ValueError(f"No beats parsed from {annotation_path}")

    unique_codes = sorted(set(codes))
    downbeat_code = max(unique_codes)
    non_downbeat_codes = [code for code in unique_codes if code != downbeat_code]

    code_to_index: dict[int, int] = {downbeat_code: 1}
    for offset, code in enumerate(non_downbeat_codes, start=2):
        code_to_index[code] = offset

    beat_indices = np.asarray([code_to_index[code] for code in codes], dtype=np.int64)
    beat_times_arr = np.asarray(beat_times, dtype=np.float64)
    return beat_times_arr, beat_indices


def convert_dataset_to_phase_npy(
    data_dir: Path,
    output_dir: Path,
    audio_patterns: list[str],
    annotation_resolver: Callable[[Path], Path],
    annotation_parser: Callable[[Path], tuple[np.ndarray, np.ndarray]],
    fps: int,
) -> PhaseGenerationStats:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files: list[Path] = []
    for pattern in audio_patterns:
        audio_files.extend(sorted(data_dir.rglob(pattern)))

    seen: set[Path] = set()
    dedup_audio_files: list[Path] = []
    for audio_path in audio_files:
        if audio_path not in seen:
            dedup_audio_files.append(audio_path)
            seen.add(audio_path)

    converted = 0
    missing_annot = 0
    parse_errors = 0

    for audio_path in dedup_audio_files:
        try:
            annotation_path = annotation_resolver(audio_path)
        except FileNotFoundError:
            missing_annot += 1
            continue

        try:
            beat_times, beat_indices = annotation_parser(annotation_path)
            duration_sec = _audio_duration_seconds(audio_path)
            phase = _build_phase_array(
                beat_times_sec=beat_times,
                beat_indices=beat_indices,
                duration_sec=duration_sec,
                fps=fps,
            )
        except Exception:
            parse_errors += 1
            continue

        stem = audio_path.stem.replace("_L+R", "")
        out_path = output_dir / f"{stem}.npy"
        np.save(out_path, phase)
        converted += 1

    return PhaseGenerationStats(
        total_audio_files=len(dedup_audio_files),
        converted_files=converted,
        skipped_missing_annotation=missing_annot,
        skipped_parse_error=parse_errors,
    )


def build_standard_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", type=str, required=True, help="Dataset root folder")
    parser.add_argument(
        "--output_mode",
        type=str,
        choices=["inside", "separate"],
        default="inside",
        help="'inside': write to <root>/<dataset>/phases, 'separate': write to <out_root>/<dataset>",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="Output root folder when --output_mode separate",
    )
    parser.add_argument("--fps", type=int, default=100, help="Target frame rate for phase targets")
    return parser


def resolve_output_dir(root: Path, out_root: str | None, output_mode: str, dataset_key: str) -> Path:
    if output_mode == "inside":
        return root / dataset_key / "phases"
    if output_mode == "separate":
        if out_root is None:
            raise ValueError("--out_root is required when --output_mode separate")
        return Path(out_root) / dataset_key
    raise ValueError(f"Unsupported output_mode: {output_mode}")


def print_stats(dataset_key: str, stats: PhaseGenerationStats, output_dir: Path) -> None:
    print(f"[{dataset_key}] output={output_dir}")
    print(
        f"[{dataset_key}] total={stats.total_audio_files}, "
        f"converted={stats.converted_files}, "
        f"missing_annotation={stats.skipped_missing_annotation}, "
        f"parse_error={stats.skipped_parse_error}"
    )

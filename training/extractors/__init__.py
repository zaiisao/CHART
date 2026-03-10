"""Extractor backend registry for modular CHART training."""

from __future__ import annotations

from training.extractors.base import ExtractorBackend
from training.extractors.wavebeat_backend import WaveBeatBackend


def get_extractor_backend(name: str) -> ExtractorBackend:
    normalized = name.lower()
    if normalized == "wavebeat":
        return WaveBeatBackend()
    raise ValueError(f"Unsupported extractor backend: {name}")


def list_extractor_backends() -> list[str]:
    return ["wavebeat"]

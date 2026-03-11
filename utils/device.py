from __future__ import annotations

import torch


def is_gpu_available() -> bool:
    return torch.cuda.is_available()


def preferred_device_option() -> str:
    return "gpu" if is_gpu_available() else "cpu"


def normalize_device_option(requested_device: str) -> str:
    normalized = (requested_device or preferred_device_option()).strip().lower()
    if normalized in {"", "auto", "gpu", "cuda", "cuda:0"}:
        return preferred_device_option() if normalized in {"", "auto"} else ("gpu" if is_gpu_available() else "cpu")
    if normalized == "cpu":
        return "cpu"
    return preferred_device_option()


def normalize_device_value(requested_device: str) -> str:
    normalized = normalize_device_option(requested_device)
    if normalized == "gpu":
        return "cuda:0" if is_gpu_available() else "cpu"
    if normalized == "cpu":
        return "cpu"
    return normalized


def resolve_device(requested_device: str) -> str:
    normalized = normalize_device_value(requested_device).strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda"):
        if not is_gpu_available():
            raise ValueError("CUDA device was requested but no GPU is available")
        return normalized
    raise ValueError(f'Unsupported device "{requested_device}". Use "gpu" or "cpu".')


def list_device_options() -> list[tuple[str, str]]:
    gpu_label = "GPU"
    if is_gpu_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_label = f"GPU ({gpu_name})"
    else:
        gpu_label = "GPU（未检测到，将自动回落 CPU）"
    return [
        ("gpu", gpu_label),
        ("cpu", "CPU"),
    ]

"""Helpers for higher-throughput SAM2 inference in the Qt GUI."""

from __future__ import annotations

import numpy as np
import torch

_ACCELERATED_SAM2_DEVICES = frozenset({"cuda", "mps"})


def apply_torch_inference_optimizations(device_name: str) -> None:
    """Enable safe global inference optimizations for the active torch device."""
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if device_name != "cuda":
        return

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def sam2_session_devices(device_name: str) -> dict[str, str]:
    """Keep streamed SAM2 session state on the accelerator when possible."""
    target = device_name if device_name in _ACCELERATED_SAM2_DEVICES else "cpu"
    return {
        "inference_device": target,
        "inference_state_device": target,
        "processing_device": target,
        "video_storage_device": target,
    }


def sam2_preprocess_batch_size(device_name: str, cache_size: int) -> int:
    """Return the user-requested preprocessing batch size."""
    del device_name
    return max(1, cache_size)


def merge_sam2_masks(object_masks: torch.Tensor | np.ndarray | list[torch.Tensor] | list[np.ndarray]) -> np.ndarray:
    """Merge per-object SAM2 masks into a single float32 mask on CPU."""
    if isinstance(object_masks, torch.Tensor):
        merged = torch.amax(object_masks, dim=0)
        while merged.ndim > 2:
            merged = merged[0]
        return merged.detach().to("cpu", dtype=torch.float32).numpy()

    if isinstance(object_masks, np.ndarray):
        merged = object_masks
        while merged.ndim > 2:
            merged = merged[0]
        return merged.astype(np.float32, copy=False)

    merged_mask: np.ndarray | None = None
    for mask in object_masks:
        current = merge_sam2_masks(mask)
        merged_mask = current if merged_mask is None else np.maximum(merged_mask, current)

    if merged_mask is None:
        raise ValueError("SAM2 returned no masks to merge.")

    return merged_mask.astype(np.float32, copy=False)

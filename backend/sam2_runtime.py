"""Helpers for higher-throughput SAM2 inference in the Qt GUI."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from queue import Full, Queue
from typing import TypeVar

import numpy as np
import torch

_ACCELERATED_SAM2_DEVICES = frozenset({"cuda", "mps"})
_PREFETCH_SENTINEL = object()
_PREFETCH_POLL_INTERVAL = 0.05
_PrefetchItem = TypeVar("_PrefetchItem")


@dataclass
class Sam2PrefetchedBatch:
    start_frame_index: int
    frame_batch_rgb: list[np.ndarray]
    pixel_values_batch: torch.Tensor
    original_sizes: list[list[int]] | torch.Tensor


@dataclass
class _PrefetchFailure:
    error: BaseException


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


def iter_prefetched_batches(
    load_next_batch: Callable[[], _PrefetchItem | None],
    *,
    prefetch_count: int = 2,
) -> Iterator[_PrefetchItem]:
    """Load future batches on a background thread while the caller consumes the current batch."""
    batch_queue: Queue[object] = Queue(maxsize=max(1, prefetch_count))
    stop_event = threading.Event()

    def _put(item: object) -> None:
        while not stop_event.is_set():
            try:
                batch_queue.put(item, timeout=_PREFETCH_POLL_INTERVAL)
                return
            except Full:
                continue

    def _producer() -> None:
        try:
            while not stop_event.is_set():
                batch = load_next_batch()
                if batch is None:
                    _put(_PREFETCH_SENTINEL)
                    return
                _put(batch)
            _put(_PREFETCH_SENTINEL)
        except BaseException as exc:
            _put(_PrefetchFailure(exc))

    producer = threading.Thread(target=_producer, name="sam2-prefetch", daemon=True)
    producer.start()

    try:
        while True:
            item = batch_queue.get()
            if item is _PREFETCH_SENTINEL:
                return
            if isinstance(item, _PrefetchFailure):
                raise item.error
            yield item
    finally:
        stop_event.set()
        producer.join(timeout=1.0)


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

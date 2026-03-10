from __future__ import annotations

import numpy as np
import pytest
import torch

from backend.sam2_runtime import (
    iter_prefetched_batches,
    merge_sam2_masks,
    sam2_preprocess_batch_size,
    sam2_session_devices,
)


def test_sam2_session_devices_keep_accelerated_state_on_device():
    assert sam2_session_devices("mps") == {
        "inference_device": "mps",
        "inference_state_device": "mps",
        "processing_device": "mps",
        "video_storage_device": "mps",
    }
    assert sam2_session_devices("cuda") == {
        "inference_device": "cuda",
        "inference_state_device": "cuda",
        "processing_device": "cuda",
        "video_storage_device": "cuda",
    }


def test_sam2_session_devices_fall_back_to_cpu():
    assert sam2_session_devices("cpu") == {
        "inference_device": "cpu",
        "inference_state_device": "cpu",
        "processing_device": "cpu",
        "video_storage_device": "cpu",
    }


def test_sam2_preprocess_batch_size_uses_user_value():
    assert sam2_preprocess_batch_size("mps", 64) == 64
    assert sam2_preprocess_batch_size("cuda", 256) == 256
    assert sam2_preprocess_batch_size("cpu", 64) == 64
    assert sam2_preprocess_batch_size("mps", 0) == 1


def test_iter_prefetched_batches_preserves_order():
    items = iter([1, 2, 3])

    def load_next():
        return next(items, None)

    assert list(iter_prefetched_batches(load_next, prefetch_count=2)) == [1, 2, 3]


def test_iter_prefetched_batches_raises_producer_error():
    state = {"calls": 0}

    def load_next():
        if state["calls"] == 0:
            state["calls"] += 1
            return "batch-0"
        raise RuntimeError("prefetch failed")

    iterator = iter_prefetched_batches(load_next, prefetch_count=2)
    assert next(iterator) == "batch-0"
    with pytest.raises(RuntimeError, match="prefetch failed"):
        next(iterator)


def test_merge_sam2_masks_merges_tensor_batch():
    masks = torch.tensor(
        [
            [[[0.0, 0.5], [0.2, 0.1]]],
            [[[0.9, 0.1], [0.3, 0.4]]],
        ],
        dtype=torch.float32,
    )

    merged = merge_sam2_masks(masks)

    assert merged.dtype == np.float32
    assert merged.shape == (2, 2)
    np.testing.assert_allclose(
        merged,
        np.array([[0.9, 0.5], [0.3, 0.4]], dtype=np.float32),
    )


def test_merge_sam2_masks_merges_list_inputs():
    merged = merge_sam2_masks(
        [
            np.array([[[0.1, 0.6], [0.0, 0.2]]], dtype=np.float32),
            torch.tensor([[[0.5, 0.2], [0.9, 0.1]]], dtype=torch.float32),
        ]
    )

    np.testing.assert_allclose(
        merged,
        np.array([[0.5, 0.6], [0.9, 0.2]], dtype=np.float32),
    )

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from backend.clip_state import ClipEntry
from backend.service import CorridorKeyService, InferenceParams, OutputConfig


def _fake_result(h: int = 4, w: int = 4) -> dict:
    return {
        "alpha": np.full((h, w, 1), 0.8, dtype=np.float32),
        "fg": np.full((h, w, 3), 0.6, dtype=np.float32),
        "comp": np.full((h, w, 3), 0.5, dtype=np.float32),
        "processed": np.full((h, w, 4), 0.4, dtype=np.float32),
    }


def test_service_run_inference_uses_process_batch(tmp_clip_dir):
    entry = ClipEntry("shot_a", str(tmp_clip_dir / "shot_a"))
    entry.find_assets()

    service = CorridorKeyService()
    mock_engine = MagicMock()
    mock_engine.prepare_batch = None
    mock_engine.process_prepared_batch = None
    mock_engine.process_batch.return_value = [_fake_result(), _fake_result()]
    service._get_engine = lambda: mock_engine  # type: ignore[method-assign]

    cfg = OutputConfig(
        fg_enabled=True,
        fg_format="png",
        matte_enabled=True,
        matte_format="png",
        comp_enabled=True,
        comp_format="png",
        processed_enabled=True,
        processed_format="png",
    )

    results = service.run_inference(
        entry,
        InferenceParams(),
        output_config=cfg,
        batch_size=8,
    )

    assert len(results) == 2
    mock_engine.process_batch.assert_called_once()
    args, kwargs = mock_engine.process_batch.call_args
    assert len(args[0]) == 2
    assert len(args[1]) == 2
    assert kwargs["input_is_linear"] is False


def test_service_get_engine_uses_backend_factory():
    service = CorridorKeyService()
    service._device = "mps"
    mock_engine = MagicMock()

    with (
        patch("CorridorKeyModule.backend.resolve_backend", return_value="mlx") as mock_resolve_backend,
        patch("CorridorKeyModule.backend.create_engine", return_value=mock_engine) as mock_create_engine,
    ):
        engine = service._get_engine()

    assert engine is mock_engine
    assert service.inference_backend == "mlx"
    mock_resolve_backend.assert_called_once_with()
    mock_create_engine.assert_called_once_with(backend="mlx", device="mps", img_size=2048)


def test_service_run_inference_uses_prepared_batch_when_available(tmp_clip_dir):
    entry = ClipEntry("shot_a", str(tmp_clip_dir / "shot_a"))
    entry.find_assets()

    service = CorridorKeyService()
    mock_engine = MagicMock()
    mock_engine.prepare_batch.return_value = "prepared-batch"
    mock_engine.process_prepared_batch.return_value = [_fake_result(), _fake_result()]
    service._get_engine = lambda: mock_engine  # type: ignore[method-assign]

    results = service.run_inference(
        entry,
        InferenceParams(),
        output_config=OutputConfig(),
        batch_size=8,
    )

    assert len(results) == 2
    mock_engine.prepare_batch.assert_called_once()
    mock_engine.process_prepared_batch.assert_called_once_with("prepared-batch")
    mock_engine.process_batch.assert_not_called()

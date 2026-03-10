from __future__ import annotations

from unittest.mock import MagicMock

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

from __future__ import annotations

from backend.clip_state import ClipAsset, ClipEntry, ClipState


def test_find_assets_accepts_alpha_hint_video(monkeypatch, tmp_path):
    clip_root = tmp_path / "shot_alpha_video"
    source_dir = clip_root / "Source"
    source_dir.mkdir(parents=True)
    (source_dir / "input.mp4").touch()
    (clip_root / "AlphaHint.mp4").touch()

    def fake_calculate_length(self: ClipAsset) -> None:
        self.frame_count = 2

    monkeypatch.setattr(ClipAsset, "_calculate_length", fake_calculate_length)

    entry = ClipEntry("shot_alpha_video", str(clip_root))
    entry.find_assets()

    assert entry.input_asset is not None
    assert entry.input_asset.asset_type == "video"
    assert entry.alpha_asset is not None
    assert entry.alpha_asset.asset_type == "video"
    assert entry.state == ClipState.READY

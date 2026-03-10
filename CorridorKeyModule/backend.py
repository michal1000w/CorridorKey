"""Backend factory — selects Torch or MLX engine and normalizes output contracts."""

from __future__ import annotations

import gc
import glob
import logging
import os
import platform
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
TORCH_EXT = ".pth"
MLX_EXT = ".safetensors"
DEFAULT_IMG_SIZE = 2048

BACKEND_ENV_VAR = "CORRIDORKEY_BACKEND"
VALID_BACKENDS = ("auto", "torch", "mlx")
_MLX_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_MLX_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_backend(requested: str | None = None) -> str:
    """Resolve backend: CLI flag > env var > auto-detect.

    Auto mode: Apple Silicon + corridorkey_mlx importable + .safetensors found → mlx.
    Otherwise → torch.

    Raises RuntimeError if explicit backend is unavailable.
    """
    if requested is None or requested.lower() == "auto":
        backend = os.environ.get(BACKEND_ENV_VAR, "auto").lower()
    else:
        backend = requested.lower()

    if backend == "auto":
        return _auto_detect_backend()

    if backend not in VALID_BACKENDS:
        raise RuntimeError(f"Unknown backend '{backend}'. Valid: {', '.join(VALID_BACKENDS)}")

    if backend == "mlx":
        _validate_mlx_available()

    return backend


def _auto_detect_backend() -> str:
    """Try MLX on Apple Silicon, fall back to Torch."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        logger.info("Not Apple Silicon — using torch backend")
        return "torch"

    try:
        import corridorkey_mlx  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        logger.info("corridorkey_mlx not installed — using torch backend")
        return "torch"

    safetensor_files = glob.glob(os.path.join(CHECKPOINT_DIR, f"*{MLX_EXT}"))
    if not safetensor_files:
        logger.info("No %s checkpoint found — using torch backend", MLX_EXT)
        return "torch"

    logger.info("Apple Silicon + MLX available — using mlx backend")
    return "mlx"


def _validate_mlx_available() -> None:
    """Raise RuntimeError with actionable message if MLX can't be used."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX backend requires Apple Silicon (M1+ Mac)")

    try:
        import corridorkey_mlx  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as err:
        raise RuntimeError(
            "MLX backend requested but corridorkey_mlx is not installed. "
            "Install with: uv pip install corridorkey-mlx@git+https://github.com/nikopueringer/corridorkey-mlx.git"
        ) from err


def _discover_checkpoint(ext: str) -> Path:
    """Find exactly one checkpoint with the given extension.

    Raises FileNotFoundError (0 found) or ValueError (>1 found).
    Includes cross-reference hints when wrong extension files exist.
    """
    matches = glob.glob(os.path.join(CHECKPOINT_DIR, f"*{ext}"))

    if len(matches) == 0:
        if ext == TORCH_EXT:
            logger.info(f"No {ext} checkpoint found in {CHECKPOINT_DIR}. Downloading from HuggingFace...")
            try:
                import huggingface_hub

                huggingface_hub.snapshot_download(
                    repo_id="nikopueringer/CorridorKey_v1.0", local_dir=CHECKPOINT_DIR, allow_patterns=[f"*{ext}"]
                )
            except Exception as e:
                logger.error(f"Failed to download {ext} checkpoints: {e}")
        elif ext == MLX_EXT:
            logger.info(f"No {ext} checkpoint found in {CHECKPOINT_DIR}. Downloading from GitHub Releases...")
            try:
                import urllib.request

                url = "https://github.com/nikopueringer/corridorkey-mlx/releases/download/v1.0.0/corridorkey_mlx.safetensors"
                dest = os.path.join(CHECKPOINT_DIR, "corridorkey_mlx.safetensors")
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                logger.error(f"Failed to download {ext} checkpoint: {e}")

        matches = glob.glob(os.path.join(CHECKPOINT_DIR, f"*{ext}"))

    if len(matches) == 0:
        other_ext = MLX_EXT if ext == TORCH_EXT else TORCH_EXT
        other_files = glob.glob(os.path.join(CHECKPOINT_DIR, f"*{other_ext}"))
        hint = ""
        if other_files:
            other_backend = "mlx" if other_ext == MLX_EXT else "torch"
            hint = f" (Found {other_ext} files — did you mean --backend={other_backend}?)"
        raise FileNotFoundError(f"No {ext} checkpoint found in {CHECKPOINT_DIR}.{hint}")

    if len(matches) > 1:
        names = [os.path.basename(f) for f in matches]
        raise ValueError(f"Multiple {ext} checkpoints in {CHECKPOINT_DIR}: {names}. Keep exactly one.")

    return Path(matches[0])


def _wrap_mlx_output(raw: dict, despill_strength: float, auto_despeckle: bool, despeckle_size: int) -> dict:
    """Normalize MLX uint8 output to match Torch float32 contract.

    Torch contract:
      alpha:     [H,W,1] float32 0-1
      fg:        [H,W,3] float32 0-1 sRGB
      comp:      [H,W,3] float32 0-1 sRGB
      processed: [H,W,4] float32 linear premul RGBA
    """
    from CorridorKeyModule.core import color_utils as cu

    # alpha: uint8 [H,W] → float32 [H,W,1]
    alpha_raw = raw["alpha"]
    alpha = alpha_raw.astype(np.float32) / 255.0
    if alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]

    # fg: uint8 [H,W,3] → float32 [H,W,3] (sRGB)
    fg = raw["fg"].astype(np.float32) / 255.0

    # Apply despeckle (MLX stubs this)
    if auto_despeckle:
        processed_alpha = cu.clean_matte(alpha, area_threshold=despeckle_size, dilation=25, blur_size=5)
    else:
        processed_alpha = alpha

    # Apply despill (MLX stubs this)
    fg_despilled = cu.despill(fg, green_limit_mode="average", strength=despill_strength)

    # Composite over checkerboard for comp output
    h, w = fg.shape[:2]
    bg_srgb = cu.create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55)
    bg_lin = cu.srgb_to_linear(bg_srgb)
    fg_despilled_lin = cu.srgb_to_linear(fg_despilled)
    comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
    comp_srgb = cu.linear_to_srgb(comp_lin)

    # Build processed: [H,W,4] linear premul RGBA
    fg_premul_lin = cu.premultiply(fg_despilled_lin, processed_alpha)
    processed_rgba = np.concatenate([fg_premul_lin, processed_alpha], axis=-1)

    return {
        "alpha": alpha,  # raw prediction (before despeckle), matches Torch
        "fg": fg,  # raw sRGB prediction, matches Torch
        "comp": comp_srgb,  # sRGB composite on checker
        "processed": processed_rgba,  # linear premul RGBA
    }


def _compute_mlx_tile_coords(image_size: int, tile_size: int, overlap: int) -> list[tuple[int, int]]:
    """Compute tile start/end coordinates for batched MLX tiled inference."""
    if overlap >= tile_size:
        raise ValueError(f"overlap ({overlap}) must be less than tile_size ({tile_size})")

    if image_size <= tile_size:
        return [(0, image_size)]

    stride = tile_size - overlap
    coords: list[tuple[int, int]] = []
    start = 0
    while start < image_size:
        end = min(start + tile_size, image_size)
        if end - start < tile_size and start > 0:
            start = max(0, end - tile_size)
        coords.append((start, end))
        if end == image_size:
            break
        start += stride
    return coords


def _make_mlx_blend_weights_2d(
    tile_h: int,
    tile_w: int,
    overlap: int,
    position: tuple[bool, bool, bool, bool],
) -> np.ndarray:
    """Create per-tile blend weights matching corridorkey-mlx tiling behavior."""
    weights = np.ones((tile_h, tile_w), dtype=np.float32)
    has_top, has_bottom, has_left, has_right = position

    if overlap <= 0:
        return weights

    ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)

    if has_top and overlap <= tile_h:
        weights[:overlap, :] *= ramp[:, None]
    if has_bottom and overlap <= tile_h:
        weights[-overlap:, :] *= ramp[::-1, None]
    if has_left and overlap <= tile_w:
        weights[:, :overlap] *= ramp[None, :]
    if has_right and overlap <= tile_w:
        weights[:, -overlap:] *= ramp[None, ::-1]

    return weights


class _MLXEngineAdapter:
    """Wraps CorridorKeyMLXEngine to match Torch output contract."""

    def __init__(self, raw_engine):
        self._engine = raw_engine
        logger.info("MLX adapter active: despill and despeckle are handled by the adapter layer, not native MLX")

    @staticmethod
    def _to_u8_image(image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be (H, W, 3), got {image.shape}")
        return image

    @staticmethod
    def _to_u8_mask(mask_linear: np.ndarray) -> np.ndarray:
        if mask_linear.dtype != np.uint8:
            mask_linear = (np.clip(mask_linear, 0.0, 1.0) * 255).astype(np.uint8)
        if mask_linear.ndim == 3:
            mask_linear = mask_linear[:, :, 0]
        if mask_linear.ndim != 2:
            raise ValueError(f"mask must be (H, W) or (H, W, 1), got {mask_linear.shape}")
        return mask_linear

    @staticmethod
    def _build_raw_result(alpha_u8: np.ndarray, fg_u8: np.ndarray, fg_is_straight: bool) -> dict[str, np.ndarray]:
        alpha_3ch = alpha_u8[:, :, np.newaxis].astype(np.float32) / 255.0
        fg_float = fg_u8.astype(np.float32)
        comp = (fg_float * alpha_3ch).astype(np.uint8) if fg_is_straight else fg_u8.copy()
        return {
            "alpha": alpha_u8,
            "fg": fg_u8,
            "comp": comp,
            "processed": fg_u8,
        }

    def _select_mlx_outputs(self, outputs: dict, refiner_scale: float):
        if self._engine._tiled:
            return outputs["alpha_final"], outputs["fg_final"]

        alpha_coarse = outputs["alpha_coarse"]
        fg_coarse = outputs["fg_coarse"]
        alpha_refined = outputs["alpha_final"]
        fg_refined = outputs["fg_final"]

        if not self._engine._use_refiner or refiner_scale == 0.0:
            return alpha_coarse, fg_coarse
        if refiner_scale == 1.0:
            return alpha_refined, fg_refined

        s = refiner_scale
        return (
            alpha_coarse * (1.0 - s) + alpha_refined * s,
            fg_coarse * (1.0 - s) + fg_refined * s,
        )

    def _prepare_full_frame_batch(self, images_u8: list[np.ndarray], masks_u8: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]]]:
        from PIL import Image

        rgb_batch: list[np.ndarray] = []
        mask_batch: list[np.ndarray] = []
        original_sizes: list[tuple[int, int]] = []
        target_size = self._engine._img_size

        for image_u8, mask_u8 in zip(images_u8, masks_u8):
            original_sizes.append(image_u8.shape[:2])

            if image_u8.shape[0] != target_size or image_u8.shape[1] != target_size:
                rgb_resized = np.asarray(
                    Image.fromarray(image_u8, mode="RGB").resize((target_size, target_size), Image.Resampling.BICUBIC),
                    dtype=np.float32,
                )
                mask_resized = np.asarray(
                    Image.fromarray(mask_u8, mode="L").resize((target_size, target_size), Image.Resampling.BICUBIC),
                    dtype=np.float32,
                )
            else:
                rgb_resized = image_u8.astype(np.float32)
                mask_resized = mask_u8.astype(np.float32)

            rgb_batch.append(rgb_resized / 255.0)
            mask_batch.append(mask_resized[:, :, np.newaxis] / 255.0)

        rgb_np = np.stack(rgb_batch, axis=0)
        mask_np = np.stack(mask_batch, axis=0)
        combined = np.concatenate([(rgb_np - _MLX_IMAGENET_MEAN) / _MLX_IMAGENET_STD, mask_np], axis=-1)
        return combined.astype(np.float32), original_sizes

    def _finalize_full_frame_batch(
        self,
        alpha_out,
        fg_out,
        original_sizes: list[tuple[int, int]],
        fg_is_straight: bool,
    ) -> list[dict[str, np.ndarray]]:
        from PIL import Image

        alpha_np = np.clip(np.array(alpha_out), 0.0, 1.0)
        fg_np = np.clip(np.array(fg_out), 0.0, 1.0)

        results: list[dict[str, np.ndarray]] = []
        for idx, (original_h, original_w) in enumerate(original_sizes):
            alpha_u8 = (alpha_np[idx, :, :, 0] * 255.0).astype(np.uint8)
            fg_u8 = (fg_np[idx] * 255.0).astype(np.uint8)

            if alpha_u8.shape != (original_h, original_w):
                target = (original_w, original_h)
                alpha_u8 = np.asarray(
                    Image.fromarray(alpha_u8, mode="L").resize(target, Image.Resampling.BICUBIC),
                    dtype=np.uint8,
                )
                fg_u8 = np.asarray(
                    Image.fromarray(fg_u8, mode="RGB").resize(target, Image.Resampling.BICUBIC),
                    dtype=np.uint8,
                )

            results.append(self._build_raw_result(alpha_u8, fg_u8, fg_is_straight))

        return results

    def _run_full_frame_batch(
        self,
        images_u8: list[np.ndarray],
        masks_u8: list[np.ndarray],
        refiner_scale: float,
        fg_is_straight: bool,
    ) -> list[dict[str, np.ndarray]]:
        import mlx.core as mx

        inputs_np, original_sizes = self._prepare_full_frame_batch(images_u8, masks_u8)
        x = mx.array(inputs_np)
        outputs = self._engine._model(x)
        alpha_out, fg_out = self._select_mlx_outputs(outputs, refiner_scale)
        mx.eval(alpha_out, fg_out)  # noqa: S307

        results = self._finalize_full_frame_batch(alpha_out, fg_out, original_sizes, fg_is_straight)

        del outputs, alpha_out, fg_out, x
        gc.collect()
        mx.clear_cache()
        return results

    def _run_tiled_batch_same_shape(
        self,
        images_u8: list[np.ndarray],
        masks_u8: list[np.ndarray],
        refiner_scale: float,
        fg_is_straight: bool,
    ) -> list[dict[str, np.ndarray]]:
        import mlx.core as mx

        rgb_np = np.stack([image.astype(np.float32) / 255.0 for image in images_u8], axis=0)
        mask_np = np.stack([mask[:, :, np.newaxis].astype(np.float32) / 255.0 for mask in masks_u8], axis=0)
        x = mx.array(np.concatenate([(rgb_np - _MLX_IMAGENET_MEAN) / _MLX_IMAGENET_STD, mask_np], axis=-1))

        batch_size, full_h, full_w, _ = x.shape
        tile_size = self._engine._tile_size
        overlap = self._engine._overlap

        if full_h <= tile_size and full_w <= tile_size:
            outputs = self._engine._model(x)
            alpha_out, fg_out = self._select_mlx_outputs(outputs, refiner_scale)
            mx.eval(alpha_out, fg_out)  # noqa: S307
            results = self._finalize_full_frame_batch(alpha_out, fg_out, [img.shape[:2] for img in images_u8], fg_is_straight)
            del outputs, alpha_out, fg_out, x
            gc.collect()
            mx.clear_cache()
            return results

        y_coords = _compute_mlx_tile_coords(full_h, tile_size, overlap)
        x_coords = _compute_mlx_tile_coords(full_w, tile_size, overlap)

        alpha_accum = np.zeros((batch_size, full_h, full_w, 1), dtype=np.float32)
        fg_accum = np.zeros((batch_size, full_h, full_w, 3), dtype=np.float32)
        weight_accum = np.zeros((1, full_h, full_w, 1), dtype=np.float32)

        for yi, (y_start, y_end) in enumerate(y_coords):
            for xi, (x_start, x_end) in enumerate(x_coords):
                tile = x[:, y_start:y_end, x_start:x_end, :]

                pad_h = tile_size - (y_end - y_start)
                pad_w = tile_size - (x_end - x_start)
                if pad_h > 0 or pad_w > 0:
                    tile = mx.pad(tile, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)])

                out = self._engine._model(tile)
                alpha_tile, fg_tile = self._select_mlx_outputs(out, refiner_scale)
                mx.eval(alpha_tile, fg_tile)  # noqa: S307

                alpha_tile_np = np.array(alpha_tile)[:, : y_end - y_start, : x_end - x_start, :]
                fg_tile_np = np.array(fg_tile)[:, : y_end - y_start, : x_end - x_start, :]

                position = (
                    yi > 0,
                    yi < len(y_coords) - 1,
                    xi > 0,
                    xi < len(x_coords) - 1,
                )
                weights = _make_mlx_blend_weights_2d(y_end - y_start, x_end - x_start, overlap, position)[:, :, None]

                alpha_accum[:, y_start:y_end, x_start:x_end, :] += alpha_tile_np * weights[None, :, :, :]
                fg_accum[:, y_start:y_end, x_start:x_end, :] += fg_tile_np * weights[None, :, :, :]
                weight_accum[:, y_start:y_end, x_start:x_end, :] += weights[None, :, :, :]

                del out, tile, alpha_tile, fg_tile, alpha_tile_np, fg_tile_np
                gc.collect()
                mx.clear_cache()

        alpha_final = alpha_accum / np.maximum(weight_accum, 1e-8)
        fg_final = fg_accum / np.maximum(weight_accum, 1e-8)

        results = self._finalize_full_frame_batch(alpha_final, fg_final, [img.shape[:2] for img in images_u8], fg_is_straight)
        del x
        gc.collect()
        mx.clear_cache()
        return results

    def _run_tiled_batch(
        self,
        images_u8: list[np.ndarray],
        masks_u8: list[np.ndarray],
        refiner_scale: float,
        fg_is_straight: bool,
    ) -> list[dict[str, np.ndarray]]:
        grouped_indices: dict[tuple[int, int], list[int]] = {}
        for idx, image_u8 in enumerate(images_u8):
            grouped_indices.setdefault(image_u8.shape[:2], []).append(idx)

        raw_results: list[dict[str, np.ndarray] | None] = [None] * len(images_u8)
        for indices in grouped_indices.values():
            batch_results = self._run_tiled_batch_same_shape(
                [images_u8[idx] for idx in indices],
                [masks_u8[idx] for idx in indices],
                refiner_scale,
                fg_is_straight,
            )
            for idx, result in zip(indices, batch_results):
                raw_results[idx] = result

        if any(result is None for result in raw_results):
            raise RuntimeError("MLX tiled batch inference failed to produce outputs for all frames")
        return [result for result in raw_results if result is not None]

    def process_batch(
        self,
        images,
        masks_linear,
        refiner_scale=1.0,
        input_is_linear=False,
        fg_is_straight=True,
        despill_strength=1.0,
        auto_despeckle=True,
        despeckle_size=400,
    ):
        """Run MLX inference on a batch of frames and normalize to Torch contract."""
        if len(images) != len(masks_linear):
            raise ValueError("images and masks_linear must have the same length")
        if not images:
            return []
        if len(images) == 1:
            return [
                self.process_frame(
                    images[0],
                    masks_linear[0],
                    refiner_scale=refiner_scale,
                    input_is_linear=input_is_linear,
                    fg_is_straight=fg_is_straight,
                    despill_strength=despill_strength,
                    auto_despeckle=auto_despeckle,
                    despeckle_size=despeckle_size,
                )
            ]

        images_u8 = [self._to_u8_image(image) for image in images]
        masks_u8 = [self._to_u8_mask(mask_linear) for mask_linear in masks_linear]

        if self._engine._tiled:
            raw_outputs = self._run_tiled_batch(images_u8, masks_u8, refiner_scale, fg_is_straight)
        else:
            raw_outputs = self._run_full_frame_batch(images_u8, masks_u8, refiner_scale, fg_is_straight)

        return [
            _wrap_mlx_output(raw, despill_strength, auto_despeckle, despeckle_size)
            for raw in raw_outputs
        ]

    def process_frame(
        self,
        image,
        mask_linear,
        refiner_scale=1.0,
        input_is_linear=False,
        fg_is_straight=True,
        despill_strength=1.0,
        auto_despeckle=True,
        despeckle_size=400,
    ):
        """Delegate to MLX engine, then normalize output to Torch contract."""
        image_u8 = self._to_u8_image(image)
        mask_u8 = self._to_u8_mask(mask_linear)

        raw = self._engine.process_frame(
            image_u8,
            mask_u8,
            refiner_scale=refiner_scale,
            input_is_linear=input_is_linear,
            fg_is_straight=fg_is_straight,
            despill_strength=0.0,  # disable MLX stubs — adapter applies these
            auto_despeckle=False,
            despeckle_size=despeckle_size,
        )

        return _wrap_mlx_output(raw, despill_strength, auto_despeckle, despeckle_size)


DEFAULT_MLX_TILE_SIZE = 512
DEFAULT_MLX_TILE_OVERLAP = 64


def create_engine(
    backend: str | None = None,
    device: str | None = None,
    img_size: int = DEFAULT_IMG_SIZE,
    tile_size: int | None = DEFAULT_MLX_TILE_SIZE,
    overlap: int = DEFAULT_MLX_TILE_OVERLAP,
):
    """Factory: returns an engine with process_frame() matching the Torch contract.

    Args:
        tile_size: MLX only — tile size for tiled inference (default 512).
            Set to None to disable tiling and use full-frame inference.
        overlap: MLX only — overlap pixels between tiles (default 64).
    """
    backend = resolve_backend(backend)

    if backend == "mlx":
        ckpt = _discover_checkpoint(MLX_EXT)
        from corridorkey_mlx import CorridorKeyMLXEngine  # type: ignore[import-not-found]

        raw_engine = CorridorKeyMLXEngine(str(ckpt), img_size=img_size, tile_size=tile_size, overlap=overlap)
        mode = f"tiled (tile={tile_size}, overlap={overlap})" if tile_size else "full-frame"
        logger.info("MLX engine loaded: %s [%s]", ckpt.name, mode)
        return _MLXEngineAdapter(raw_engine)
    else:
        ckpt = _discover_checkpoint(TORCH_EXT)
        from CorridorKeyModule.inference_engine import CorridorKeyEngine

        logger.info("Torch engine loaded: %s (device=%s)", ckpt.name, device)
        return CorridorKeyEngine(checkpoint_path=str(ckpt), device=device or "cpu", img_size=img_size)

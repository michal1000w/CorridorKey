from __future__ import annotations

import logging
import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .core import color_utils as cu
from .core.model_transformer import GreenFormer

logger = logging.getLogger(__name__)


def _disable_fused_attention_for_mps(model: torch.nn.Module, device: torch.device) -> int:
    """Disable fused SDPA in Hiera attention blocks on MPS.

    Some Apple Silicon / MPS runs hit an internal view/stride error inside
    torch.scaled_dot_product_attention() when timm's Hiera backbone uses the
    fused attention path. Falling back to the explicit math path is slower,
    but stable for export.
    """
    if device.type != "mps":
        return 0

    disabled = 0
    for module in model.modules():
        if hasattr(module, "fused_attn") and isinstance(getattr(module, "fused_attn"), bool):
            if module.fused_attn:
                module.fused_attn = False
                disabled += 1

    if disabled:
        logger.info("Disabled fused SDPA on MPS for %d attention block(s)", disabled)
    return disabled


class CorridorKeyEngine:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        img_size: int = 2048,
        use_refiner: bool = True,
        mixed_precision: bool = True,
        model_precision: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.img_size = img_size
        self.checkpoint_path = checkpoint_path
        self.use_refiner = use_refiner

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        if mixed_precision or model_precision != torch.float32:
            # Use faster matrix multiplication implementation
            # This reduces the floating point precision a little bit,
            # but it should be negligible compared to fp16 precision
            torch.set_float32_matmul_precision("high")

        self.mixed_precision = mixed_precision
        if mixed_precision and model_precision == torch.float16:
            # using mixed precision, when the precision is already fp16, is slower
            self.mixed_precision = False

        self.model_precision = model_precision

        self.model = self._load_model().to(model_precision)

    def _load_model(self) -> GreenFormer:
        logger.info("Loading CorridorKey from %s", self.checkpoint_path)
        # Initialize Model (Hiera Backbone)
        model = GreenFormer(
            encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k", img_size=self.img_size, use_refiner=self.use_refiner
        )
        model = model.to(self.device)
        model.eval()

        # Load Weights
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Fix Compiled Model Prefix & Handle PosEmbed Mismatch
        new_state_dict = {}
        model_state = model.state_dict()

        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                k = k[10:]

            # Check for PosEmbed Mismatch
            if "pos_embed" in k and k in model_state:
                if v.shape != model_state[k].shape:
                    print(f"Resizing {k} from {v.shape} to {model_state[k].shape}")
                    # v: [1, N_src, C]
                    # target: [1, N_dst, C]
                    # We assume square grid
                    N_src = v.shape[1]
                    N_dst = model_state[k].shape[1]
                    C = v.shape[2]

                    grid_src = int(math.sqrt(N_src))
                    grid_dst = int(math.sqrt(N_dst))

                    # Reshape to [1, C, H, W]
                    v_img = v.permute(0, 2, 1).reshape(1, C, grid_src, grid_src)

                    # Interpolate
                    v_resized = F.interpolate(v_img, size=(grid_dst, grid_dst), mode="bicubic", align_corners=False)

                    # Reshape back
                    v = v_resized.flatten(2).transpose(1, 2)

            new_state_dict[k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if len(missing) > 0:
            print(f"[Warning] Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"[Warning] Unexpected keys: {unexpected}")

        _disable_fused_attention_for_mps(model, self.device)
        return model

    @torch.inference_mode()
    def process_batch(
        self,
        images: list[np.ndarray],
        masks_linear: list[np.ndarray],
        refiner_scale: float = 1.0,
        input_is_linear: bool = False,
        fg_is_straight: bool = True,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
    ) -> list[dict[str, np.ndarray]]:
        """Process a batch of frames with one model forward pass."""
        if len(images) != len(masks_linear):
            raise ValueError("images and masks_linear must have the same length")
        if not images:
            return []

        batch_inputs: list[np.ndarray] = []
        original_sizes: list[tuple[int, int]] = []

        for image, mask_linear in zip(images, masks_linear):
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0

            if mask_linear.dtype == np.uint8:
                mask_linear = mask_linear.astype(np.float32) / 255.0

            h, w = image.shape[:2]
            original_sizes.append((h, w))

            if mask_linear.ndim == 2:
                mask_linear = mask_linear[:, :, np.newaxis]

            if input_is_linear:
                img_resized_lin = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                img_resized = cu.linear_to_srgb(img_resized_lin)
            else:
                img_resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

            mask_resized = cv2.resize(mask_linear, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            if mask_resized.ndim == 2:
                mask_resized = mask_resized[:, :, np.newaxis]

            img_norm = (img_resized - self.mean) / self.std
            batch_inputs.append(np.concatenate([img_norm, mask_resized], axis=-1))

        inp_np = np.stack(batch_inputs, axis=0)
        inp_nchw = np.ascontiguousarray(inp_np.transpose((0, 3, 1, 2)))
        inp_t = torch.from_numpy(inp_nchw).to(self.model_precision).to(self.device).contiguous()

        handle = None
        if refiner_scale != 1.0 and self.model.refiner is not None:

            def scale_hook(module, input, output):
                return output * refiner_scale

            handle = self.model.refiner.register_forward_hook(scale_hook)

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.mixed_precision):
            out = self.model(inp_t)

        if handle:
            handle.remove()

        pred_alpha_batch = out["alpha"].permute(0, 2, 3, 1).float().cpu().numpy()
        pred_fg_batch = out["fg"].permute(0, 2, 3, 1).float().cpu().numpy()

        results: list[dict[str, np.ndarray]] = []
        for (h, w), pred_alpha, pred_fg in zip(original_sizes, pred_alpha_batch, pred_fg_batch):
            res_alpha = cv2.resize(pred_alpha, (w, h), interpolation=cv2.INTER_LANCZOS4)
            res_fg = cv2.resize(pred_fg, (w, h), interpolation=cv2.INTER_LANCZOS4)

            if res_alpha.ndim == 2:
                res_alpha = res_alpha[:, :, np.newaxis]

            if auto_despeckle:
                processed_alpha = cu.clean_matte(res_alpha, area_threshold=despeckle_size, dilation=25, blur_size=5)
            else:
                processed_alpha = res_alpha

            fg_despilled = cu.despill(res_fg, green_limit_mode="average", strength=despill_strength)
            fg_despilled_lin = cu.srgb_to_linear(fg_despilled)
            fg_premul_lin = cu.premultiply(fg_despilled_lin, processed_alpha)
            processed_rgba = np.concatenate([fg_premul_lin, processed_alpha], axis=-1)

            bg_srgb = cu.create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55)
            bg_lin = cu.srgb_to_linear(bg_srgb)

            if fg_is_straight:
                comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
            else:
                comp_lin = cu.composite_premul(fg_despilled_lin, bg_lin, processed_alpha)

            comp_srgb = cu.linear_to_srgb(comp_lin)
            results.append(
                {
                    "alpha": res_alpha,
                    "fg": res_fg,
                    "comp": comp_srgb,
                    "processed": processed_rgba,
                }
            )

        return results

    @torch.inference_mode()
    def process_frame(
        self,
        image: np.ndarray,
        mask_linear: np.ndarray,
        refiner_scale: float = 1.0,
        input_is_linear: bool = False,
        fg_is_straight: bool = True,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
    ) -> dict[str, np.ndarray]:
        """
        Process a single frame.
        Args:
            image: Numpy array [H, W, 3] (0.0-1.0 or 0-255).
                   - If input_is_linear=False (Default): Assumed sRGB.
                   - If input_is_linear=True: Assumed Linear.
            mask_linear: Numpy array [H, W] or [H, W, 1] (0.0-1.0). Assumed Linear.
            refiner_scale: Multiplier for Refiner Deltas (default 1.0).
            input_is_linear: bool. If True, resizes in Linear then transforms to sRGB.
                             If False, resizes in sRGB (standard).
            fg_is_straight: bool. If True, assumes FG output is Straight (unpremultiplied).
                            If False, assumes FG output is Premultiplied.
            despill_strength: float. 0.0 to 1.0 multiplier for the despill effect.
            auto_despeckle: bool. If True, cleans up small disconnected components from the predicted alpha matte.
            despeckle_size: int. Minimum number of consecutive pixels required to keep an island.
        Returns:
             dict: {'alpha': np, 'fg': np (sRGB), 'comp': np (sRGB on Gray)}
        """
        return self.process_batch(
            [image],
            [mask_linear],
            refiner_scale=refiner_scale,
            input_is_linear=input_is_linear,
            fg_is_straight=fg_is_straight,
            despill_strength=despill_strength,
            auto_despeckle=auto_despeckle,
            despeckle_size=despeckle_size,
        )[0]

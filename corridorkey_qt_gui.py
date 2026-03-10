'''
GUI v1
author: Michal Wieczorek
nick: michal1000w
email: michal_wieczorek@hotmail.com
'''
from __future__ import annotations

import os
import platform
import shutil
import sys
from importlib.util import find_spec
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent
_BOOTSTRAP_ENV = "CORRIDORKEY_QT_GUI_BOOTSTRAPPED"


def _configure_local_caches(env: dict[str, str] | None = None) -> dict[str, str]:
    target = env if env is not None else os.environ

    cache_map = {
        "UV_CACHE_DIR": _PROJECT_ROOT / ".uv_cache",
        "TORCH_HOME": _PROJECT_ROOT / ".torch_cache",
        "HF_HOME": _PROJECT_ROOT / ".hf_cache",
    }
    for key, path in cache_map.items():
        target.setdefault(key, str(path))
        os.makedirs(target[key], exist_ok=True)
    target.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return target


def _is_direct_script_invocation() -> bool:
    try:
        return Path(sys.argv[0]).resolve() == _THIS_FILE
    except Exception:
        return False


def _project_venv_python() -> Path:
    if os.name == "nt":
        return _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return _PROJECT_ROOT / ".venv" / "bin" / "python"


def _missing_runtime_modules() -> list[str]:
    required = ["PySide6", "cv2", "numpy", "torch", "transformers"]
    if sys.platform == "darwin" and platform.machine() == "arm64":
        required.extend(["mlx", "corridorkey_mlx"])
    return [name for name in required if find_spec(name) is None]


def _apple_silicon_uv_args() -> list[str]:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return []
    return [
        "--with",
        "mlx>=0.29",
        "--with",
        "mlx-metal>=0.29",
        "--with",
        "corridorkey-mlx@git+https://github.com/nikopueringer/corridorkey-mlx.git@04503e797060e091f991bc88b85ec61b0b9b862b",
    ]


def _relaunch_with_runtime() -> None:
    if not _is_direct_script_invocation():
        return

    missing = _missing_runtime_modules()
    if not missing:
        return

    if os.environ.get(_BOOTSTRAP_ENV) == "1":
        missing_text = ", ".join(missing)
        raise SystemExit(f"Failed to bootstrap CorridorKey Qt GUI with {missing_text}.")

    env = os.environ.copy()
    env[_BOOTSTRAP_ENV] = "1"
    env = _configure_local_caches(env)

    uv = shutil.which("uv")
    if uv:
        print("Bootstrapping CorridorKey Qt GUI with the project uv environment...", file=sys.stderr)
        cmd = [uv, "run", "--group", "gui", *_apple_silicon_uv_args(), "python", str(_THIS_FILE), *sys.argv[1:]]
        os.execvpe(uv, cmd, env)

    venv_python = _project_venv_python()
    if venv_python.exists():
        print("Launching CorridorKey Qt GUI with the project virtual environment...", file=sys.stderr)
        cmd = [str(venv_python), str(_THIS_FILE), *sys.argv[1:]]
        os.execve(str(venv_python), cmd, env)

    missing_text = ", ".join(missing) if missing else "the GUI runtime"
    raise SystemExit(
        "Could not locate a runnable project environment for CorridorKey Qt GUI. "
        f"Missing: {missing_text}. Install `uv`, then rerun the script."
    )


_configure_local_caches()
_relaunch_with_runtime()

import glob
import gc
import logging
import threading
import traceback
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import torch

from backend.clip_state import ClipEntry
from backend.errors import JobCancelledError
from backend.job_queue import GPUJob, JobType
from backend.project import create_project, get_clip_dirs, is_image_file, is_video_file
from backend.sam2_runtime import (
    apply_torch_inference_optimizations,
    merge_sam2_masks,
    sam2_preprocess_batch_size,
    sam2_session_devices,
)
from backend.service import CorridorKeyService, InferenceParams, OutputConfig

try:
    from PySide6.QtCore import QThread, Qt, Signal
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QProgressBar,
        QSizePolicy,
        QSlider,
        QSplitter,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - runtime dependency, not exercised in tests
    raise SystemExit(
        "PySide6 is still unavailable after bootstrap. "
        "Run `uv run --group gui python corridorkey_qt_gui.py` to inspect the environment directly."
    ) from exc


logger = logging.getLogger(__name__)

_MASK_PALETTE = [
    (245, 92, 71),
    (255, 191, 73),
    (64, 168, 255),
    (78, 205, 196),
    (139, 92, 246),
    (255, 123, 172),
    (127, 219, 255),
    (107, 203, 119),
]

_SAM2_VARIANTS = {
    "SAM2 Tiny (fast)": {
        "label": "SAM2.1 Tiny",
        "repos": ("facebook/sam2.1-hiera-tiny", "facebook/sam2-hiera-tiny"),
    },
    "SAM2 Small (balanced)": {
        "label": "SAM2.1 Small",
        "repos": ("facebook/sam2.1-hiera-small", "facebook/sam2-hiera-small"),
    },
}
_DEFAULT_SAM2_VARIANT = "SAM2 Tiny (fast)"
_FAST_PNG_WRITE_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]


@dataclass
class SegmentationCandidate:
    label_id: int
    label_name: str
    score: float
    box: tuple[int, int, int, int]
    mask: np.ndarray
    prompt_point: tuple[int, int] | None = None


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    arr = image.astype(np.float32)
    if arr.size == 0:
        return np.zeros((8, 8), dtype=np.uint8)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    max_value = float(arr.max()) if arr.size else 1.0
    if max_value > 1.5:
        arr = arr / max_value
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _to_rgb_preview(image: np.ndarray, grayscale: bool = False) -> np.ndarray:
    if image.ndim == 2:
        gray = _normalize_to_uint8(image)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 4:
        image = image[:, :, :3]

    if grayscale:
        if image.dtype != np.uint8:
            image = _normalize_to_uint8(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if image.dtype != np.uint8:
        image = _normalize_to_uint8(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _qt_pixmap_from_rgb(image: np.ndarray) -> QPixmap:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape [H, W, 3]")
    contiguous = np.ascontiguousarray(image)
    h, w, _ = contiguous.shape
    qimage = QImage(contiguous.data, w, h, contiguous.strides[0], QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


def _mask_to_preview(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = _normalize_to_uint8(mask)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def _clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def _merge_candidate_masks(candidates: list[SegmentationCandidate]) -> np.ndarray | None:
    if not candidates:
        return None
    merged = np.zeros_like(candidates[0].mask, dtype=np.float32)
    for candidate in candidates:
        merged = np.maximum(merged, candidate.mask.astype(np.float32))
    return merged


def _coarse_hint_from_mask(mask: np.ndarray, blur_size: int, erode_size: int) -> np.ndarray:
    binary = (mask >= 0.5).astype(np.uint8) * 255
    if erode_size > 0:
        k = erode_size if erode_size % 2 == 1 else erode_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.erode(binary, kernel, iterations=1)
    if blur_size > 0:
        k = blur_size if blur_size % 2 == 1 else blur_size + 1
        binary = cv2.GaussianBlur(binary, (k, k), 0)
    return binary


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    binary = mask >= 0.5
    if not binary.any():
        return None
    ys, xs = np.nonzero(binary)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_binary = a >= 0.5
    b_binary = b >= 0.5
    intersection = int(np.logical_and(a_binary, b_binary).sum())
    union = int(np.logical_or(a_binary, b_binary).sum())
    return intersection / union if union else 0.0


def _make_candidate(mask: np.ndarray, score: float, label_index: int, prompt_point: tuple[int, int]) -> SegmentationCandidate:
    box = _mask_box(mask)
    if box is None:
        raise ValueError("The selected point did not produce a valid object mask.")
    return SegmentationCandidate(
        label_id=label_index,
        label_name=f"Object {label_index + 1}",
        score=score,
        box=box,
        mask=mask.astype(np.float32),
        prompt_point=prompt_point,
    )


def _overlay_candidates(
    frame_rgb: np.ndarray,
    candidates: list[SegmentationCandidate],
    selected_indices: set[int] | None,
) -> np.ndarray:
    canvas = frame_rgb.copy()
    selected_indices = selected_indices or set()
    for index, candidate in enumerate(candidates):
        color = np.array(_MASK_PALETTE[index % len(_MASK_PALETTE)], dtype=np.uint8)
        mask = candidate.mask >= 0.5
        if mask.any():
            canvas[mask] = ((canvas[mask].astype(np.float32) * 0.55) + (color.astype(np.float32) * 0.45)).astype(
                np.uint8
            )

        x1, y1, x2, y2 = candidate.box
        is_selected = index in selected_indices
        border_color = (50, 255, 120) if is_selected else tuple(int(v) for v in color.tolist())
        thickness = 4 if is_selected else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, thickness)
        label = f"{index + 1}. {candidate.label_name} {candidate.score:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            border_color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def _build_output_config(mode: str) -> OutputConfig:
    if mode == "Preview PNG":
        return OutputConfig(
            fg_enabled=False,
            matte_enabled=False,
            comp_enabled=True,
            processed_enabled=False,
            comp_format="png",
        )

    if mode == "Matte + Preview":
        return OutputConfig(
            fg_enabled=False,
            matte_enabled=True,
            matte_format="exr",
            comp_enabled=True,
            processed_enabled=False,
            comp_format="png",
        )

    return OutputConfig(
        fg_enabled=True,
        fg_format="exr",
        matte_enabled=True,
        matte_format="exr",
        comp_enabled=True,
        comp_format="png",
        processed_enabled=True,
        processed_format="exr",
    )


def _clear_alpha_assets(clip_root: str) -> None:
    alpha_dir = os.path.join(clip_root, "AlphaHint")
    if os.path.isdir(alpha_dir):
        shutil.rmtree(alpha_dir)
    for candidate in glob.glob(os.path.join(clip_root, "AlphaHint.*")):
        if os.path.isfile(candidate):
            os.remove(candidate)


def _copy_alpha_asset(source_path: str, clip_root: str) -> str:
    source_path = os.path.abspath(source_path)
    if not os.path.exists(source_path):
        raise FileNotFoundError(source_path)

    _clear_alpha_assets(clip_root)

    if os.path.isdir(source_path):
        target_dir = os.path.join(clip_root, "AlphaHint")
        shutil.copytree(source_path, target_dir)
        return target_dir

    if is_video_file(source_path):
        ext = os.path.splitext(source_path)[1]
        target_path = os.path.join(clip_root, f"AlphaHint{ext}")
        shutil.copy2(source_path, target_path)
        return target_path

    if is_image_file(source_path):
        target_dir = os.path.join(clip_root, "AlphaHint")
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(source_path, os.path.join(target_dir, os.path.basename(source_path)))
        return target_dir

    raise ValueError("Mask hint must be a video file, image, or image-sequence directory")


def _format_exception_details(context: str, exc: BaseException) -> str:
    details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
    return f"{context}\n{details}" if details else f"{context}\n{exc}"


def _summarize_error_for_dialog(message: str) -> str:
    if "Traceback (most recent call last):" not in message:
        return message
    lines = [line for line in message.strip().splitlines() if line.strip()]
    summary_lines = lines[-3:] if len(lines) >= 3 else lines
    summary = "\n".join(summary_lines)
    return f"{summary}\n\nFull traceback is in the Session log."


class MediaReader:
    def __init__(self, path: str, *, grayscale: bool = False):
        self.path = os.path.abspath(path)
        self.grayscale = grayscale
        self.kind = "directory" if os.path.isdir(self.path) else "file"
        self.files: list[str] = []
        self.frame_count = 0

        if os.path.isdir(self.path):
            self.files = sorted(f for f in os.listdir(self.path) if is_image_file(f))
            self.frame_count = len(self.files)
        elif is_video_file(self.path):
            cap = cv2.VideoCapture(self.path)
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        elif is_image_file(self.path):
            self.frame_count = 1
        else:
            raise ValueError(f"Unsupported media path: {self.path}")

    def read_rgb(self, index: int) -> np.ndarray | None:
        if self.frame_count <= 0:
            return None

        if os.path.isdir(self.path):
            index = max(0, min(index, len(self.files) - 1))
            image = cv2.imread(os.path.join(self.path, self.files[index]), cv2.IMREAD_UNCHANGED)
            if image is None:
                return None
            return _to_rgb_preview(image, grayscale=self.grayscale)

        if is_image_file(self.path):
            image = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            if image is None:
                return None
            return _to_rgb_preview(image, grayscale=self.grayscale)

        cap = cv2.VideoCapture(self.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        if self.grayscale:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class DropPreviewLabel(QLabel):
    file_dropped = Signal(str)
    image_clicked = Signal(int, int, bool)

    def __init__(self, title: str):
        super().__init__()
        self._title = title
        self._current_pixmap: QPixmap | None = None
        self._image_size: tuple[int, int] | None = None
        self._placeholder = f"{title}\nDrop a file here"
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(220)
        self.setStyleSheet(
            "QLabel { background: #171a20; color: #d9dee7; border: 1px dashed #4e5868; border-radius: 10px; }"
        )
        self.set_placeholder(self._placeholder)

    def set_placeholder(self, text: str) -> None:
        self._current_pixmap = None
        self._image_size = None
        self.setText(text)
        self.setPixmap(QPixmap())

    def set_rgb_image(self, image: np.ndarray) -> None:
        self._image_size = (image.shape[1], image.shape[0])
        self._current_pixmap = _qt_pixmap_from_rgb(image)
        self._refresh_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._refresh_pixmap()

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            self.file_dropped.emit(urls[0].toLocalFile())
            event.acceptProposedAction()
        else:
            event.ignore()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._current_pixmap is None or self._image_size is None:
            return super().mousePressEvent(event)

        pixmap_rect = self._pixmap_rect()
        if pixmap_rect is None or not pixmap_rect.contains(int(event.position().x()), int(event.position().y())):
            return super().mousePressEvent(event)

        image_width, image_height = self._image_size
        scale_x = image_width / pixmap_rect.width()
        scale_y = image_height / pixmap_rect.height()
        image_x = int((event.position().x() - pixmap_rect.x()) * scale_x)
        image_y = int((event.position().y() - pixmap_rect.y()) * scale_y)
        shift_held = bool(event.modifiers() & Qt.ShiftModifier)
        self.image_clicked.emit(image_x, image_y, shift_held)
        super().mousePressEvent(event)

    def _refresh_pixmap(self) -> None:
        if self._current_pixmap is None:
            return
        scaled = self._current_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
        self.setText("")

    def _pixmap_rect(self):
        if self.pixmap() is None or self.pixmap().isNull():
            return None
        pix = self.pixmap()
        x = (self.width() - pix.width()) // 2
        y = (self.height() - pix.height()) // 2
        from PySide6.QtCore import QRect

        return QRect(x, y, pix.width(), pix.height())


class SegmentationModelManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._image_model = None
        self._image_processor = None
        self._video_model = None
        self._video_processor = None
        self._variant_key: str | None = None
        self._variant_label = "SAM2"
        self._repo_id: str | None = None
        self._device = "cpu"
        self._dtype = torch.float32

    def runtime_label(self) -> str:
        return f"{self._variant_label} on {self._device}"

    def segment_point(
        self,
        frame_rgb: np.ndarray,
        point_xy: tuple[int, int],
        preferred_device: str,
        model_variant: str,
    ) -> tuple[SegmentationCandidate, str]:
        with self._lock:
            try:
                candidate = self._segment_point_once(frame_rgb, point_xy, preferred_device, model_variant)
            except Exception:
                if self._device == "cpu":
                    raise
                logger.warning("SAM2 image segmentation failed on %s, retrying on CPU", self._device)
                self._reset_runtime()
                candidate = self._segment_point_once(frame_rgb, point_xy, "cpu", model_variant, force_device="cpu")
        return candidate, self.runtime_label()

    def generate_mask_hint(
        self,
        source_path: str,
        clip_root: str,
        selections: list[SegmentationCandidate],
        preferred_device: str,
        model_variant: str,
        blur_size: int,
        erode_size: int,
        cache_size: int,
        cancel_event: threading.Event,
        on_frame: Callable[[int, int, np.ndarray, np.ndarray], None],
    ) -> tuple[str, str]:
        with self._lock:
            try:
                alpha_dir = self._generate_mask_hint_once(
                    source_path,
                    clip_root,
                    selections,
                    preferred_device,
                    model_variant,
                    blur_size,
                    erode_size,
                    cache_size,
                    cancel_event,
                    on_frame,
                )
            except Exception:
                if self._device == "cpu":
                    raise
                logger.warning("SAM2 video propagation failed on %s, retrying on CPU", self._device)
                self._reset_runtime()
                alpha_dir = self._generate_mask_hint_once(
                    source_path,
                    clip_root,
                    selections,
                    "cpu",
                    model_variant,
                    blur_size,
                    erode_size,
                    cache_size,
                    cancel_event,
                    on_frame,
                    force_device="cpu",
                )
        return alpha_dir, self.runtime_label()

    def _segment_point_once(
        self,
        frame_rgb: np.ndarray,
        point_xy: tuple[int, int],
        preferred_device: str,
        model_variant: str,
        *,
        force_device: str | None = None,
    ) -> SegmentationCandidate:
        self._ensure_image_runtime(preferred_device, model_variant, force_device=force_device)
        inputs = self._image_processor(
            images=frame_rgb,
            input_points=[[[[float(point_xy[0]), float(point_xy[1])]]]],
            input_labels=[[[1]]],
            return_tensors="pt",
        )
        model_inputs = self._move_inputs_to_device(inputs, pixel_dtype=self._dtype)
        with torch.inference_mode():
            outputs = self._image_model(**model_inputs, multimask_output=False)
        masks = self._image_processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"], binarize=True)
        mask = masks[0][0, 0].detach().cpu().numpy().astype(np.float32)
        if float(mask.sum()) < 128.0:
            raise RuntimeError("The selected point did not produce a stable object mask.")
        score = 1.0
        if outputs.iou_scores is not None:
            score = float(outputs.iou_scores.reshape(-1)[0].detach().cpu().item())
        if outputs.object_score_logits is not None:
            object_score = torch.sigmoid(outputs.object_score_logits.reshape(-1)[0]).detach().cpu().item()
            score = max(score, float(object_score))
        return _make_candidate(mask, score, 0, point_xy)

    def _generate_mask_hint_once(
        self,
        source_path: str,
        clip_root: str,
        selections: list[SegmentationCandidate],
        preferred_device: str,
        model_variant: str,
        blur_size: int,
        erode_size: int,
        cache_size: int,
        cancel_event: threading.Event,
        on_frame: Callable[[int, int, np.ndarray, np.ndarray], None],
        *,
        force_device: str | None = None,
    ) -> str:
        self._ensure_video_runtime(preferred_device, model_variant, force_device=force_device)
        preprocess_batch_size = sam2_preprocess_batch_size(self._device, cache_size)

        alpha_dir = os.path.join(clip_root, "AlphaHint")
        _clear_alpha_assets(clip_root)
        os.makedirs(alpha_dir, exist_ok=True)

        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source video: {source_path}")

        total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        try:
            session_devices = sam2_session_devices(self._device)
            session = self._video_processor.init_video_session(
                inference_device=session_devices["inference_device"],
                inference_state_device=session_devices["inference_state_device"],
                processing_device=session_devices["processing_device"],
                video_storage_device=session_devices["video_storage_device"],
                max_vision_features_cache_size=max(1, cache_size),
                dtype=self._dtype,
            )

            frame_index = 0
            while True:
                frame_batch_rgb: list[np.ndarray] = []
                while len(frame_batch_rgb) < preprocess_batch_size:
                    if cancel_event.is_set():
                        raise JobCancelledError(Path(clip_root).name, frame_index)

                    ok, frame_bgr = cap.read()
                    if not ok or frame_bgr is None:
                        break
                    frame_batch_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

                if not frame_batch_rgb:
                    break

                frame_inputs = self._video_processor(images=frame_batch_rgb, return_tensors="pt")
                model_inputs = self._move_inputs_to_device(frame_inputs, pixel_dtype=self._dtype)
                pixel_values_batch = model_inputs["pixel_values"]
                original_sizes = frame_inputs["original_sizes"]

                if frame_index == 0:
                    self._video_processor.add_inputs_to_inference_session(
                        session,
                        frame_idx=0,
                        obj_ids=[candidate.label_id for candidate in selections],
                        input_masks=[candidate.mask >= 0.5 for candidate in selections],
                    )

                pred_masks_batch: list[torch.Tensor] = []
                with torch.inference_mode():
                    for batch_offset in range(len(frame_batch_rgb)):
                        current_frame_index = frame_index + batch_offset
                        outputs = self._video_model(
                            session,
                            frame_idx=current_frame_index,
                            frame=pixel_values_batch[batch_offset : batch_offset + 1],
                        )
                        pred_masks_batch.append(outputs.pred_masks)
                        self._trim_processed_frames(session, current_frame_index, max(2, cache_size))

                processed_masks = self._video_processor.post_process_masks(
                    pred_masks_batch,
                    original_sizes,
                    binarize=True,
                )

                for batch_offset, (frame_rgb, object_masks) in enumerate(
                    zip(frame_batch_rgb, processed_masks, strict=True)
                ):
                    current_frame_index = frame_index + batch_offset
                    merged_mask = merge_sam2_masks(object_masks)
                    hint = _coarse_hint_from_mask(merged_mask, blur_size, erode_size)
                    out_path = os.path.join(alpha_dir, f"frame_{current_frame_index:06d}.png")
                    cv2.imwrite(out_path, hint, _FAST_PNG_WRITE_PARAMS)
                    on_frame(current_frame_index, total_frames, frame_rgb, hint)

                frame_index += len(frame_batch_rgb)

            if frame_index == 0:
                raise RuntimeError("No frames were read from the source video")
            return alpha_dir
        finally:
            cap.release()

    def _ensure_image_runtime(self, preferred_device: str, model_variant: str, *, force_device: str | None = None) -> None:
        if self._image_model is not None and self._variant_key == model_variant and (force_device is None or self._device == force_device):
            return
        self._unload_image_runtime()
        self._unload_video_runtime()
        self._load_image_runtime(preferred_device, model_variant, force_device=force_device)

    def _ensure_video_runtime(self, preferred_device: str, model_variant: str, *, force_device: str | None = None) -> None:
        if self._video_model is not None and self._variant_key == model_variant and (force_device is None or self._device == force_device):
            return
        self._unload_video_runtime()
        self._unload_image_runtime()
        self._load_video_runtime(preferred_device, model_variant, force_device=force_device)

    def _load_image_runtime(self, preferred_device: str, model_variant: str, *, force_device: str | None = None) -> None:
        from transformers import Sam2Model, Sam2Processor

        self._load_runtime(
            model_variant,
            preferred_device,
            force_device,
            Sam2Processor,
            Sam2Model,
            "image",
        )

    def _load_video_runtime(self, preferred_device: str, model_variant: str, *, force_device: str | None = None) -> None:
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        self._load_runtime(
            model_variant,
            preferred_device,
            force_device,
            Sam2VideoProcessor,
            Sam2VideoModel,
            "video",
        )

    def _load_runtime(
        self,
        model_variant: str,
        preferred_device: str,
        force_device: str | None,
        processor_cls,
        model_cls,
        mode: str,
    ) -> None:
        variant = _SAM2_VARIANTS.get(model_variant, _SAM2_VARIANTS[_DEFAULT_SAM2_VARIANT])
        self._variant_key = model_variant if model_variant in _SAM2_VARIANTS else _DEFAULT_SAM2_VARIANT
        self._variant_label = variant["label"]
        device_order = [force_device] if force_device else self._preferred_devices(preferred_device)

        last_error = None
        for repo_id in variant["repos"]:
            try:
                processor = processor_cls.from_pretrained(repo_id)
            except Exception as exc:
                last_error = exc
                continue

            for device_name in device_order:
                dtype = self._preferred_dtype(device_name)
                model = None
                try:
                    model = model_cls.from_pretrained(repo_id, torch_dtype=dtype)
                    model.eval()
                    apply_torch_inference_optimizations(device_name)
                    model.to(device_name)
                    self._repo_id = repo_id
                    self._device = device_name
                    self._dtype = dtype
                    if mode == "image":
                        self._image_model = model
                        self._image_processor = processor
                    else:
                        self._video_model = model
                        self._video_processor = processor
                    return
                except Exception as exc:
                    last_error = exc
                    if model is not None:
                        del model
                    self._clear_device_cache(device_name)

        raise RuntimeError(f"Unable to load the {self._variant_label} {mode} runtime: {last_error}")

    def _move_inputs_to_device(self, inputs, *, pixel_dtype: torch.dtype) -> dict[str, torch.Tensor]:
        model_inputs: dict[str, torch.Tensor] = {}
        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key == "pixel_values":
                model_inputs[key] = value.to(self._device, dtype=pixel_dtype)
            else:
                model_inputs[key] = value.to(self._device)
        return model_inputs

    def _preferred_devices(self, preferred_device: str) -> list[str]:
        device_order: list[str] = []
        if preferred_device:
            device_order.append(preferred_device)
        if torch.cuda.is_available() and "cuda" not in device_order:
            device_order.append("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and "mps" not in device_order:
            device_order.append("mps")
        if "cpu" not in device_order:
            device_order.append("cpu")
        return device_order

    @staticmethod
    def _preferred_dtype(device_name: str) -> torch.dtype:
        if device_name == "cuda":
            return torch.bfloat16
        if device_name == "mps":
            return torch.float16
        return torch.float32

    @staticmethod
    def _trim_processed_frames(session, frame_index: int, keep_frame_count: int) -> None:
        processed_frames = getattr(session, "processed_frames", None)
        if not isinstance(processed_frames, dict):
            return
        floor = max(0, frame_index - keep_frame_count)
        for stale_index in list(processed_frames.keys()):
            if stale_index not in (0, frame_index) and stale_index < floor:
                processed_frames.pop(stale_index, None)

    def _unload_image_runtime(self) -> None:
        if self._image_model is not None:
            del self._image_model
            self._image_model = None
        self._image_processor = None
        self._clear_device_cache(self._device)

    def _unload_video_runtime(self) -> None:
        if self._video_model is not None:
            del self._video_model
            self._video_model = None
        self._video_processor = None
        self._clear_device_cache(self._device)

    def _reset_runtime(self) -> None:
        self._unload_image_runtime()
        self._unload_video_runtime()
        self._repo_id = None
        self._variant_key = None
        self._variant_label = "SAM2"
        self._device = "cpu"
        self._dtype = torch.float32

    @staticmethod
    def _clear_device_cache(device_name: str) -> None:
        try:
            from device_utils import clear_device_cache

            clear_device_cache(device_name)
        except Exception:
            pass
        gc.collect()


class SegmentObjectWorker(QThread):
    status = Signal(str)
    candidate_ready = Signal(object, str)
    failed = Signal(str)

    def __init__(
        self,
        frame_rgb: np.ndarray,
        point_xy: tuple[int, int],
        manager: SegmentationModelManager,
        preferred_device: str,
        model_variant: str,
    ):
        super().__init__()
        self._frame_rgb = frame_rgb
        self._point_xy = point_xy
        self._manager = manager
        self._preferred_device = preferred_device
        self._model_variant = model_variant

    def run(self) -> None:
        self.status.emit("Segmenting the clicked object with SAM2...")
        try:
            candidate, device_name = self._manager.segment_point(
                self._frame_rgb,
                self._point_xy,
                self._preferred_device,
                self._model_variant,
            )
            self.candidate_ready.emit(candidate, device_name)
        except Exception as exc:  # pragma: no cover - depends on local model/runtime
            self.failed.emit(_format_exception_details("Segment object worker failed.", exc))


class GenerateMaskHintWorker(QThread):
    status = Signal(str)
    progress = Signal(int, int)
    preview = Signal(int, object, object)
    completed = Signal(str)
    cancelled = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        source_path: str,
        clip_root: str,
        selections: list[SegmentationCandidate],
        manager: SegmentationModelManager,
        preferred_device: str,
        model_variant: str,
        blur_size: int,
        erode_size: int,
        batch_size: int,
    ):
        super().__init__()
        self._source_path = source_path
        self._clip_root = clip_root
        self._selections = selections
        self._manager = manager
        self._preferred_device = preferred_device
        self._model_variant = model_variant
        self._blur_size = blur_size
        self._erode_size = erode_size
        self._batch_size = max(1, batch_size)
        self._preview_stride = max(1, min(4, self._batch_size))
        self._cancel_requested = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def run(self) -> None:
        try:
            self.status.emit(
                f"Generating mask hints with {self._model_variant} video propagation. "
                f"Cache window: {self._batch_size} frames. Press Esc or use Cancel to stop."
            )
            alpha_dir, runtime_label = self._manager.generate_mask_hint(
                source_path=self._source_path,
                clip_root=self._clip_root,
                selections=self._selections,
                preferred_device=self._preferred_device,
                model_variant=self._model_variant,
                blur_size=self._blur_size,
                erode_size=self._erode_size,
                cache_size=self._batch_size,
                cancel_event=self._cancel_requested,
                on_frame=self._on_frame_processed,
            )
            self.status.emit(f"Mask hint generation finished with {runtime_label}.")
            self.completed.emit(alpha_dir)
        except JobCancelledError as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:  # pragma: no cover - depends on local model/runtime
            self.failed.emit(_format_exception_details("Mask hint generation worker failed.", exc))

    def _on_frame_processed(self, frame_index: int, total_frames: int, frame_rgb: np.ndarray, hint: np.ndarray) -> None:
        self._check_cancel(frame_index)
        self.progress.emit(frame_index + 1, total_frames)
        if frame_index == 0 or frame_index + 1 >= total_frames or ((frame_index + 1) % self._preview_stride) == 0:
            self.preview.emit(frame_index, frame_rgb, hint)

    def _check_cancel(self, frame_index: int) -> None:
        if self._cancel_requested.is_set():
            raise JobCancelledError(Path(self._clip_root).name, frame_index)


class ExportWorker(QThread):
    status = Signal(str)
    progress = Signal(int, int)
    preview = Signal(int, object)
    completed = Signal(str)
    cancelled = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        service: CorridorKeyService,
        clip: ClipEntry,
        params: InferenceParams,
        output_config: OutputConfig,
        batch_size: int,
    ):
        super().__init__()
        self._service = service
        self._clip = clip
        self._params = params
        self._output_config = output_config
        self._batch_size = max(1, batch_size)
        self._job = GPUJob(JobType.INFERENCE, clip.name)

    def request_cancel(self) -> None:
        self._job.request_cancel()

    def run(self) -> None:
        try:
            self.status.emit(
                f"Exporting CorridorKey outputs in batches of {self._batch_size}. "
                "Press Esc or use Cancel to stop after the current batch."
            )

            def on_progress(_clip_name: str, current: int, total: int) -> None:
                self.progress.emit(current, total)
                comp_dir = os.path.join(self._clip.root_path, "Output", "Comp")
                if not os.path.isdir(comp_dir):
                    return
                comp_files = sorted(f for f in os.listdir(comp_dir) if is_image_file(f))
                if not comp_files:
                    return
                latest = os.path.join(comp_dir, comp_files[-1])
                image = cv2.imread(latest, cv2.IMREAD_COLOR)
                if image is None:
                    return
                preview_index = max(0, min(current - 1, total - 1)) if total > 0 else 0
                self.preview.emit(preview_index, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            self._service.run_inference(
                self._clip,
                self._params,
                job=self._job,
                on_progress=on_progress,
                output_config=self._output_config,
                batch_size=self._batch_size,
            )
            self.completed.emit(os.path.join(self._clip.root_path, "Output"))
        except JobCancelledError as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:  # pragma: no cover - depends on local engine/runtime
            self.failed.emit(_format_exception_details("Export worker failed.", exc))


class CorridorKeyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CorridorKey Qt Preview")
        self.resize(1450, 920)

        self.service = CorridorKeyService()
        self.device = self.service.detect_device()
        self.segmentation_manager = SegmentationModelManager()

        self.project_dir: str | None = None
        self.clip: ClipEntry | None = None
        self.source_reader: MediaReader | None = None
        self.alpha_reader: MediaReader | None = None
        self.loaded_source_path: str | None = None
        self.loaded_alpha_path: str | None = None
        self.current_frame_index = 0
        self.candidates: list[SegmentationCandidate] = []
        self.selected_candidate_indices: set[int] = set()
        self.segment_worker: SegmentObjectWorker | None = None
        self.mask_worker: GenerateMaskHintWorker | None = None
        self.export_worker: ExportWorker | None = None
        self._pending_click_shift = False

        self._build_ui()
        self._set_busy(False)
        self._refresh_runtime_label()
        self._refresh_actions()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._cancel_active_job()
        self.service.unload_engines()
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.key() == Qt.Key_Escape and self._has_cancelable_job():
            self._cancel_active_job()
            event.accept()
            return
        super().keyPressEvent(event)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        left_panel = self._build_left_panel()
        main_layout.addWidget(left_panel, 0)

        right_panel = self._build_right_panel()
        main_layout.addWidget(right_panel, 1)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(380)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        inputs_group = QGroupBox("Inputs")
        inputs_layout = QGridLayout(inputs_group)
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Source video path")
        self.source_edit.editingFinished.connect(self._load_source_from_edit)
        self.source_browse_button = QPushButton("Browse")
        self.source_browse_button.clicked.connect(self._browse_source)

        self.alpha_edit = QLineEdit()
        self.alpha_edit.setPlaceholderText("Mask hint video / image / folder")
        self.alpha_edit.editingFinished.connect(self._load_alpha_from_edit)
        self.alpha_browse_button = QPushButton("Browse")
        self.alpha_browse_button.clicked.connect(self._browse_alpha)

        self.project_label = QLabel("No project loaded")
        self.project_label.setWordWrap(True)
        self.device_label = QLabel(self._runtime_summary_text())
        self.segment_device_label = QLabel("Segmentation backend: pending")
        self.project_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        inputs_layout.addWidget(QLabel("Video"), 0, 0)
        inputs_layout.addWidget(self.source_edit, 0, 1)
        inputs_layout.addWidget(self.source_browse_button, 0, 2)
        inputs_layout.addWidget(QLabel("Mask hint"), 1, 0)
        inputs_layout.addWidget(self.alpha_edit, 1, 1)
        inputs_layout.addWidget(self.alpha_browse_button, 1, 2)
        inputs_layout.addWidget(QLabel("Project"), 2, 0)
        inputs_layout.addWidget(self.project_label, 2, 1, 1, 2)
        inputs_layout.addWidget(self.device_label, 3, 0, 1, 3)
        inputs_layout.addWidget(self.segment_device_label, 4, 0, 1, 3)
        layout.addWidget(inputs_group)

        segment_group = QGroupBox("Mask Hint Generation")
        segment_layout = QFormLayout(segment_group)
        self.sam2_model_combo = QComboBox()
        self.sam2_model_combo.addItems(list(_SAM2_VARIANTS.keys()))
        self.sam2_model_combo.setCurrentText(_DEFAULT_SAM2_VARIANT)

        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(0, 99)
        self.blur_spin.setSingleStep(2)
        self.blur_spin.setValue(19)

        self.erode_spin = QSpinBox()
        self.erode_spin.setRange(0, 31)
        self.erode_spin.setValue(5)

        self.clear_objects_button = QPushButton("Clear Objects")
        self.clear_objects_button.clicked.connect(self._clear_segmented_objects)
        self.generate_button = QPushButton("Generate Mask Hint")
        self.generate_button.clicked.connect(self._generate_mask_hint)

        segment_layout.addRow("SAM2 model", self.sam2_model_combo)
        segment_layout.addRow("Blur size", self.blur_spin)
        segment_layout.addRow("Erode size", self.erode_spin)
        segment_layout.addRow(self.clear_objects_button)
        segment_layout.addRow(self.generate_button)
        layout.addWidget(segment_group)

        inference_group = QGroupBox("CorridorKey Settings")
        inference_layout = QFormLayout(inference_group)
        self.gamma_combo = QComboBox()
        self.gamma_combo.addItems(["sRGB", "Linear"])

        self.despill_spin = QSpinBox()
        self.despill_spin.setRange(0, 10)
        self.despill_spin.setValue(5)

        self.auto_despeckle_check = QCheckBox("Enable auto-despeckle")
        self.auto_despeckle_check.setChecked(True)

        self.despeckle_size_spin = QSpinBox()
        self.despeckle_size_spin.setRange(0, 5000)
        self.despeckle_size_spin.setValue(400)

        self.refiner_spin = QDoubleSpinBox()
        self.refiner_spin.setRange(0.1, 3.0)
        self.refiner_spin.setSingleStep(0.1)
        self.refiner_spin.setDecimals(2)
        self.refiner_spin.setValue(1.0)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 64)
        self.batch_size_spin.setValue(8)

        inference_layout.addRow("Input gamma", self.gamma_combo)
        inference_layout.addRow("Despill (0-10)", self.despill_spin)
        inference_layout.addRow(self.auto_despeckle_check)
        inference_layout.addRow("Despeckle size", self.despeckle_size_spin)
        inference_layout.addRow("Refiner scale", self.refiner_spin)
        inference_layout.addRow("Batch / cache size", self.batch_size_spin)
        layout.addWidget(inference_group)

        export_group = QGroupBox("Export")
        export_layout = QFormLayout(export_group)
        self.export_mode_combo = QComboBox()
        self.export_mode_combo.addItems(["Preview PNG", "Matte + Preview", "Full Package"])
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self._export_outputs)
        self.cancel_button = QPushButton("Cancel (Esc)")
        self.cancel_button.clicked.connect(self._cancel_active_job)
        self.cancel_button.setEnabled(False)
        action_row = QWidget()
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.addWidget(self.export_button)
        action_layout.addWidget(self.cancel_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label = QLabel("0 / 0")
        export_layout.addRow("Export mode", self.export_mode_combo)
        export_layout.addRow(action_row)
        export_layout.addRow(self.progress_bar)
        export_layout.addRow(self.progress_label)
        layout.addWidget(export_group)

        self.status_label = QLabel(
            "Load a source video to begin. Click objects on frame 0, hold Shift to add/remove more, and press Esc to cancel long jobs."
        )
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Session log")
        layout.addWidget(self.log_view, 1)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        splitter = QSplitter(Qt.Vertical)

        video_group = QGroupBox("Source Video")
        video_layout = QVBoxLayout(video_group)
        self.video_view = DropPreviewLabel("Source Video")
        self.video_view.file_dropped.connect(self._load_source_from_path)
        self.video_view.image_clicked.connect(self._handle_video_click)
        video_layout.addWidget(self.video_view)

        mask_group = QGroupBox("Mask Hint / Export Preview")
        mask_layout = QVBoxLayout(mask_group)
        self.mask_view = DropPreviewLabel("Mask Hint")
        self.mask_view.file_dropped.connect(self._load_alpha_from_path)
        mask_layout.addWidget(self.mask_view)

        splitter.addWidget(video_group)
        splitter.addWidget(mask_group)
        splitter.setSizes([520, 320])

        layout.addWidget(splitter, 1)

        slider_row = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        self.frame_label = QLabel("Frame 0 / 0")
        slider_row.addWidget(self.frame_slider, 1)
        slider_row.addWidget(self.frame_label)
        layout.addLayout(slider_row)
        return panel

    def _runtime_summary_text(self) -> str:
        backend = self.service.inference_backend
        if backend == "mlx":
            backend_text = "MLX"
        elif backend == "torch":
            backend_text = "Torch"
        else:
            backend_text = "auto"
        return f"Compute device: {self.device} | Inference backend: {backend_text}"

    def _refresh_runtime_label(self) -> None:
        self.device_label.setText(self._runtime_summary_text())

    def _browse_source(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self,
            "Choose a source video",
            str(Path.home()),
            "Video Files (*.mp4 *.mov *.avi *.mkv *.mxf *.webm *.m4v);;All Files (*)",
        )
        if path:
            self._load_source_from_path(path)

    def _browse_alpha(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self,
            "Choose a mask hint asset",
            str(Path.home()),
            "Media Files (*.mp4 *.mov *.avi *.mkv *.mxf *.webm *.m4v *.png *.jpg *.jpeg *.tif *.tiff *.bmp *.exr);;All Files (*)",
        )
        if path:
            self._load_alpha_from_path(path)

    def _load_source_from_edit(self) -> None:
        path = self.source_edit.text().strip()
        if path:
            self._load_source_from_path(path)

    def _load_alpha_from_edit(self) -> None:
        path = self.alpha_edit.text().strip()
        if path:
            self._load_alpha_from_path(path)

    def _load_source_from_path(self, path: str) -> None:
        path = os.path.abspath(path)
        if self.loaded_source_path == path:
            return
        if not os.path.isfile(path) or not is_video_file(path):
            self._show_error("Source video must be a valid video file.")
            return

        try:
            project_dir = create_project(path, copy_source=False, display_name=Path(path).stem.replace("_", " "))
            clip_paths = get_clip_dirs(project_dir)
            if not clip_paths:
                raise RuntimeError(f"No clip folders were created in {project_dir}")
            clip = ClipEntry(name=os.path.basename(clip_paths[0]), root_path=clip_paths[0])
            clip.find_assets()

            self.project_dir = project_dir
            self.clip = clip
            self.source_reader = MediaReader(clip.input_asset.path)
            self.alpha_reader = None
            self.loaded_source_path = path
            self.loaded_alpha_path = None
            self.current_frame_index = 0
            self.candidates = []
            self.selected_candidate_indices.clear()

            self.source_edit.setText(path)
            self.alpha_edit.clear()
            self.project_label.setText(project_dir)
            self._reset_progress()
            self._configure_slider(self.source_reader.frame_count)
            self._update_previews()
            self._log(f"Loaded source video: {path}")
            self.segment_device_label.setText("Segmentation backend: pending")
            self._set_status("Source video loaded. Click objects on frame 0 or drop an existing mask hint.")
        except Exception as exc:
            self._show_error(str(exc))
            return

        self._refresh_actions()

    def _load_alpha_from_path(self, path: str) -> None:
        if self.clip is None:
            self._show_error("Load the source video first.")
            return

        path = os.path.abspath(path)
        if self.loaded_alpha_path == path:
            return
        try:
            target = _copy_alpha_asset(path, self.clip.root_path)
            self.clip.find_assets()
            if self.clip.alpha_asset is None:
                raise RuntimeError("Mask hint import did not produce a readable AlphaHint asset.")
            self.alpha_reader = MediaReader(self.clip.alpha_asset.path, grayscale=True)
            self.loaded_alpha_path = path
            self.alpha_edit.setText(path)
            self.candidates = []
            self.selected_candidate_indices.clear()
            self._update_previews()
            self._log(f"Imported mask hint: {target}")
            self._set_status("Mask hint loaded. Export is now available. Click frame 0 if you want to replace it.")
        except Exception as exc:
            self._show_error(str(exc))
            return

        self._refresh_actions()

    def _clear_segmented_objects(self) -> None:
        if self._is_busy():
            return
        self.candidates = []
        self.selected_candidate_indices.clear()
        self._update_previews()
        self._set_status("Cleared the interactive SAM2 selection.")
        self._refresh_actions()

    def _segment_object_from_click(self, image_x: int, image_y: int, shift_held: bool) -> None:
        if self.source_reader is None:
            self._show_error("Load a source video first.")
            return

        frame_rgb = self.source_reader.read_rgb(0)
        if frame_rgb is None:
            self._show_error("Could not read frame 0 from the source video.")
            return

        self.current_frame_index = 0
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)

        self._pending_click_shift = shift_held
        self.segment_worker = SegmentObjectWorker(
            frame_rgb=frame_rgb,
            point_xy=(image_x, image_y),
            manager=self.segmentation_manager,
            preferred_device=self.device,
            model_variant=self.sam2_model_combo.currentText(),
        )
        self.segment_worker.status.connect(self._set_status)
        self.segment_worker.candidate_ready.connect(self._on_candidate_ready)
        self.segment_worker.failed.connect(self._show_error)
        self.segment_worker.finished.connect(self._on_worker_finished)
        self.segment_worker.start()
        self._set_busy(True)

    def _on_candidate_ready(self, candidate: SegmentationCandidate, runtime_label: str) -> None:
        self.segment_device_label.setText(f"Segmentation backend: {runtime_label}")
        existing_index = None
        for index, existing in enumerate(self.candidates):
            if _mask_iou(existing.mask, candidate.mask) >= 0.85:
                existing_index = index
                break

        if existing_index is None:
            label_index = len(self.candidates)
            prompt_point = candidate.prompt_point or (0, 0)
            candidate = _make_candidate(candidate.mask, candidate.score, label_index, prompt_point)
            self.candidates.append(candidate)
            target_index = len(self.candidates) - 1
            self._log(f"Added {candidate.label_name} with {runtime_label}.")
        else:
            target_index = existing_index
            candidate = self.candidates[target_index]

        if self._pending_click_shift:
            if target_index in self.selected_candidate_indices:
                self.selected_candidate_indices.remove(target_index)
            else:
                self.selected_candidate_indices.add(target_index)
        else:
            self.selected_candidate_indices = {target_index}

        self._update_previews()
        selection_count = len(self.selected_candidate_indices)
        if selection_count == 0:
            self._set_status("Selection cleared.")
        elif selection_count == 1:
            self._set_status(f"Selected '{candidate.label_name}'. Hold Shift and click to add or remove more objects.")
        else:
            self._set_status(f"Selected {selection_count} objects. Generate the mask hint when ready.")
        self._refresh_actions()

    def _handle_video_click(self, image_x: int, image_y: int, shift_held: bool) -> None:
        if self.current_frame_index != 0 or self.source_reader is None or self._is_busy():
            return

        hit_index = None
        for index, candidate in enumerate(self.candidates):
            h, w = candidate.mask.shape
            x = max(0, min(w - 1, image_x))
            y = max(0, min(h - 1, image_y))
            if candidate.mask[y, x] >= 0.5:
                hit_index = index
                break

        if hit_index is None:
            self._segment_object_from_click(image_x, image_y, shift_held)
            return

        if shift_held:
            if hit_index in self.selected_candidate_indices:
                self.selected_candidate_indices.remove(hit_index)
            else:
                self.selected_candidate_indices.add(hit_index)
        else:
            self.selected_candidate_indices = {hit_index}

        selected_candidates = [self.candidates[index] for index in sorted(self.selected_candidate_indices)]
        merged_mask = _merge_candidate_masks(selected_candidates)
        if merged_mask is not None:
            self.mask_view.set_rgb_image(_mask_to_preview(_coarse_hint_from_mask(merged_mask, 0, 0)))
        self._update_previews()
        selection_count = len(self.selected_candidate_indices)
        if selection_count == 0:
            self._set_status("Selection cleared.")
        elif selection_count == 1:
            candidate = selected_candidates[0]
            self._set_status(f"Selected '{candidate.label_name}'. Hold Shift and click to add more objects.")
        else:
            self._set_status(f"Selected {selection_count} objects. Generate the mask hint when ready.")
        self._refresh_actions()

    def _generate_mask_hint(self) -> None:
        if self.clip is None or self.source_reader is None:
            self._show_error("Load a source video first.")
            return
        if not self.selected_candidate_indices:
            self._show_error("Click one or more objects on frame 0 first.")
            return

        selections = [self.candidates[index] for index in sorted(self.selected_candidate_indices)]
        self._log(
            "DEBUG mask hint request: "
            f"clip_root={self.clip.root_path}, "
            f"source={self.clip.input_asset.path}, "
            f"objects={[candidate.label_name for candidate in selections]}, "
            f"sam2_model={self.sam2_model_combo.currentText()}, "
            f"blur={self.blur_spin.value()}, "
            f"erode={self.erode_spin.value()}, "
            f"cache={self.batch_size_spin.value()}"
        )
        self.mask_worker = GenerateMaskHintWorker(
            source_path=self.clip.input_asset.path,
            clip_root=self.clip.root_path,
            selections=selections,
            manager=self.segmentation_manager,
            preferred_device=self.device,
            model_variant=self.sam2_model_combo.currentText(),
            blur_size=self.blur_spin.value(),
            erode_size=self.erode_spin.value(),
            batch_size=self.batch_size_spin.value(),
        )
        self.mask_worker.status.connect(self._set_status)
        self.mask_worker.progress.connect(self._set_progress)
        self.mask_worker.preview.connect(self._on_mask_preview)
        self.mask_worker.completed.connect(self._on_mask_generation_completed)
        self.mask_worker.cancelled.connect(self._on_worker_cancelled)
        self.mask_worker.failed.connect(self._show_error)
        self.mask_worker.finished.connect(self._on_worker_finished)
        self.mask_worker.start()
        self._set_busy(True)

    def _on_mask_preview(self, frame_index: int, frame_rgb: np.ndarray, hint: np.ndarray) -> None:
        self.video_view.set_rgb_image(frame_rgb)
        self.mask_view.set_rgb_image(_mask_to_preview(hint))
        self.frame_label.setText(f"Frame {frame_index + 1} / {max(1, self.source_reader.frame_count if self.source_reader else 1)}")

    def _on_mask_generation_completed(self, alpha_dir: str) -> None:
        if self.clip is None:
            return
        self.clip.find_assets()
        if self.clip.alpha_asset is None:
            self._show_error("Mask generation completed, but AlphaHint could not be reloaded.")
            return
        self.alpha_reader = MediaReader(self.clip.alpha_asset.path, grayscale=True)
        self.alpha_edit.setText(alpha_dir)
        self.loaded_alpha_path = os.path.abspath(alpha_dir)
        self.current_frame_index = 0
        self._update_previews()
        self._set_status("Mask hint generation finished. Export is ready.")
        self._log(f"Generated AlphaHint sequence at {alpha_dir}")
        self._refresh_actions()

    def _export_outputs(self) -> None:
        if self.clip is None or self.clip.alpha_asset is None:
            self._show_error("Load or generate a mask hint before exporting.")
            return

        params = InferenceParams(
            input_is_linear=self.gamma_combo.currentText() == "Linear",
            despill_strength=self.despill_spin.value() / 10.0,
            auto_despeckle=self.auto_despeckle_check.isChecked(),
            despeckle_size=self.despeckle_size_spin.value(),
            refiner_scale=self.refiner_spin.value(),
        )
        output_config = _build_output_config(self.export_mode_combo.currentText())
        self._log(
            "DEBUG export request: "
            f"clip_root={self.clip.root_path}, "
            f"input={self.clip.input_asset.path if self.clip.input_asset else 'missing'}, "
            f"alpha={self.clip.alpha_asset.path if self.clip.alpha_asset else 'missing'}, "
            f"mode={self.export_mode_combo.currentText()}, "
            f"batch_size={self.batch_size_spin.value()}, "
            f"gamma={self.gamma_combo.currentText()}, "
            f"despill={params.despill_strength}, "
            f"auto_despeckle={params.auto_despeckle}, "
            f"despeckle_size={params.despeckle_size}, "
            f"refiner_scale={params.refiner_scale}"
        )

        self.export_worker = ExportWorker(
            service=self.service,
            clip=self.clip,
            params=params,
            output_config=output_config,
            batch_size=self.batch_size_spin.value(),
        )
        self.export_worker.status.connect(self._set_status)
        self.export_worker.progress.connect(self._set_progress)
        self.export_worker.preview.connect(self._on_export_preview)
        self.export_worker.completed.connect(self._on_export_completed)
        self.export_worker.cancelled.connect(self._on_worker_cancelled)
        self.export_worker.failed.connect(self._show_error)
        self.export_worker.finished.connect(self._on_worker_finished)
        self.export_worker.start()
        self._set_busy(True)

    def _on_export_preview(self, frame_index: int, comp_rgb: np.ndarray) -> None:
        self.current_frame_index = frame_index
        if self.source_reader is not None:
            source_frame = self.source_reader.read_rgb(frame_index)
            if source_frame is not None:
                self.video_view.set_rgb_image(source_frame)
        self.mask_view.set_rgb_image(comp_rgb)
        total = max(1, self.source_reader.frame_count if self.source_reader else 1)
        self.frame_label.setText(f"Frame {frame_index + 1} / {total}")

    def _on_export_completed(self, output_dir: str) -> None:
        if self.clip is not None:
            self.clip.find_assets()
        self._set_status("Export completed.")
        self._log(f"Outputs written to {output_dir}")
        self._update_previews()
        self._refresh_actions()

    def _on_worker_finished(self) -> None:
        self._set_busy(False)
        self._refresh_runtime_label()
        self.segment_worker = None
        self.mask_worker = None
        self.export_worker = None
        self._refresh_actions()

    def _on_worker_cancelled(self, message: str) -> None:
        self._log(f"CANCELLED: {message}")
        self.status_label.setText(message)

    def _configure_slider(self, frame_count: int) -> None:
        enabled = frame_count > 0
        self.frame_slider.setEnabled(enabled)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(0, frame_count - 1))
        self.frame_slider.setValue(0)
        self.frame_label.setText(f"Frame 1 / {max(frame_count, 1)}")

    def _on_frame_changed(self, value: int) -> None:
        self.current_frame_index = value
        self._update_previews()

    def _update_previews(self) -> None:
        if self.source_reader is None:
            self.video_view.set_placeholder("Source Video\nDrop a video file here")
            self.mask_view.set_placeholder("Mask Hint\nDrop a video or sequence here")
            self.frame_label.setText("Frame 0 / 0")
            return

        source_frame = self.source_reader.read_rgb(self.current_frame_index)
        if source_frame is not None:
            selected_indices = self.selected_candidate_indices if self.current_frame_index == 0 else set()
            if self.current_frame_index == 0 and self.candidates:
                self.video_view.set_rgb_image(_overlay_candidates(source_frame, self.candidates, selected_indices))
            else:
                self.video_view.set_rgb_image(source_frame)

        if self.current_frame_index == 0 and self.selected_candidate_indices:
            selected = [self.candidates[index] for index in sorted(self.selected_candidate_indices)]
            merged_mask = _merge_candidate_masks(selected)
            if merged_mask is not None:
                self.mask_view.set_rgb_image(_mask_to_preview(_coarse_hint_from_mask(merged_mask, 0, 0)))
        elif self.alpha_reader is not None:
            alpha_frame = self.alpha_reader.read_rgb(self.current_frame_index)
            if alpha_frame is not None:
                self.mask_view.set_rgb_image(alpha_frame)
        else:
            self.mask_view.set_placeholder("Mask Hint\nDrop a mask asset or click frame 0 to generate one")

        total = max(self.source_reader.frame_count, 1)
        self.frame_label.setText(f"Frame {self.current_frame_index + 1} / {total}")

    def _refresh_actions(self) -> None:
        busy = self.segment_worker is not None or self.mask_worker is not None or self.export_worker is not None
        has_source = self.clip is not None and self.source_reader is not None
        has_segmented_objects = bool(self.candidates)
        has_selection = bool(self.selected_candidate_indices)
        has_alpha = self.clip is not None and self.clip.alpha_asset is not None

        self.clear_objects_button.setEnabled(has_source and has_segmented_objects and not busy)
        self.generate_button.setEnabled(has_source and has_selection and not busy)
        self.export_button.setEnabled(has_source and has_alpha and not busy)
        self.cancel_button.setEnabled(self._has_cancelable_job())
        self.frame_slider.setEnabled(has_source and not busy)
        self.source_edit.setEnabled(not busy)
        self.alpha_edit.setEnabled(not busy)
        self.source_browse_button.setEnabled(not busy)
        self.alpha_browse_button.setEnabled(not busy)
        self.batch_size_spin.setEnabled(not busy)
        self.sam2_model_combo.setEnabled(not busy)

    def _set_busy(self, busy: bool) -> None:
        if busy:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
        self._refresh_actions()

    def _set_progress(self, current: int, total: int) -> None:
        total = max(total, 1)
        if self.progress_bar.maximum() != total:
            self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(min(current, total))
        self.progress_label.setText(f"{current} / {total}")

    def _reset_progress(self) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0 / 0")

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self._log(text)

    def _show_error(self, message: str) -> None:
        self._set_busy(False)
        self._refresh_actions()
        if "Traceback (most recent call last):" in message:
            self._log("ERROR TRACEBACK BEGIN")
            for line in message.splitlines():
                self._log(line)
            self._log("ERROR TRACEBACK END")
            dialog_message = _summarize_error_for_dialog(message)
        else:
            self._log(f"ERROR: {message}")
            dialog_message = message
        QMessageBox.critical(self, "CorridorKey Qt GUI", dialog_message)

    def _log(self, message: str) -> None:
        if not message:
            return
        self.log_view.appendPlainText(message)

    def _is_busy(self) -> bool:
        return self.segment_worker is not None or self.mask_worker is not None or self.export_worker is not None

    def _has_cancelable_job(self) -> bool:
        return self.mask_worker is not None or self.export_worker is not None

    def _cancel_active_job(self) -> None:
        if self.mask_worker is not None:
            self.mask_worker.request_cancel()
            self._set_status("Cancelling mask hint generation...")
            return
        if self.export_worker is not None:
            self.export_worker.request_cancel()
            self._set_status("Cancelling export after the current batch...")
            return


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app = QApplication([])
    app.setApplicationName("CorridorKey Qt Preview")
    window = CorridorKeyWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import os
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


def _running_inside_project_venv() -> bool:
    venv_root = _project_venv_python().parent.parent.resolve()
    try:
        return Path(sys.prefix).resolve() == venv_root
    except Exception:
        return False


def _missing_runtime_modules() -> list[str]:
    required = ("PySide6", "cv2", "numpy", "torch")
    return [name for name in required if find_spec(name) is None]


def _relaunch_with_runtime() -> None:
    if not _is_direct_script_invocation():
        return

    missing = _missing_runtime_modules()
    if _running_inside_project_venv() and not missing:
        return

    if os.environ.get(_BOOTSTRAP_ENV) == "1":
        missing_text = ", ".join(missing) if missing else "the project runtime"
        raise SystemExit(f"Failed to bootstrap CorridorKey Qt GUI with {missing_text}.")

    env = os.environ.copy()
    env[_BOOTSTRAP_ENV] = "1"
    env = _configure_local_caches(env)

    uv = shutil.which("uv")
    if uv:
        print("Bootstrapping CorridorKey Qt GUI with the project uv environment...", file=sys.stderr)
        cmd = [uv, "run", "--group", "gui", "python", str(_THIS_FILE), *sys.argv[1:]]
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
import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from backend.clip_state import ClipEntry
from backend.project import create_project, get_clip_dirs, is_image_file, is_video_file
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


@dataclass
class SegmentationCandidate:
    label_id: int
    label_name: str
    score: float
    box: tuple[int, int, int, int]
    mask: np.ndarray


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


def _xyxy_to_xywh(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _xywh_to_xyxy(box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    return x1, y1, x2, y2


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union else 0.0


def _rectangle_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = _clip_box(box, width, height)
    mask = np.zeros((height, width), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    return mask


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


def _create_tracker():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    return None


def _overlay_candidates(
    frame_rgb: np.ndarray,
    candidates: list[SegmentationCandidate],
    selected_index: int | None,
) -> np.ndarray:
    canvas = frame_rgb.copy()
    for index, candidate in enumerate(candidates):
        color = np.array(_MASK_PALETTE[index % len(_MASK_PALETTE)], dtype=np.uint8)
        mask = candidate.mask >= 0.5
        if mask.any():
            canvas[mask] = ((canvas[mask].astype(np.float32) * 0.55) + (color.astype(np.float32) * 0.45)).astype(
                np.uint8
            )

        x1, y1, x2, y2 = candidate.box
        border_color = (50, 255, 120) if selected_index == index else tuple(int(v) for v in color.tolist())
        thickness = 4 if selected_index == index else 2
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
    image_clicked = Signal(int, int)

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
        self.image_clicked.emit(image_x, image_y)
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
        self._model = None
        self._categories: list[str] = []
        self._device = "cpu"

    def detect_candidates(
        self,
        frame_rgb: np.ndarray,
        preferred_device: str,
        score_threshold: float,
    ) -> tuple[list[SegmentationCandidate], str]:
        with self._lock:
            output = self._predict(frame_rgb, preferred_device)

        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        boxes = output["boxes"].detach().cpu().numpy()
        masks = output["masks"].detach().cpu().numpy()[:, 0]

        candidates: list[SegmentationCandidate] = []
        height, width = frame_rgb.shape[:2]
        for score, label_id, box_array, mask in zip(scores, labels, boxes, masks):
            if float(score) < score_threshold:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box_array.tolist()]
            clipped_box = _clip_box((x1, y1, x2, y2), width, height)
            label_index = int(label_id)
            label_name = self._categories[label_index] if label_index < len(self._categories) else f"class_{label_id}"
            candidates.append(
                SegmentationCandidate(
                    label_id=label_index,
                    label_name=label_name,
                    score=float(score),
                    box=clipped_box,
                    mask=mask.astype(np.float32),
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:12], self._device

    def _predict(self, frame_rgb: np.ndarray, preferred_device: str):
        if self._model is None:
            self._load_model(preferred_device)

        tensor = torch.from_numpy(frame_rgb.transpose(2, 0, 1)).float().div(255.0).to(self._device)
        try:
            with torch.inference_mode():
                return self._model([tensor])[0]
        except Exception:
            if self._device == "cpu":
                raise
            self._load_model("cpu")
            tensor = tensor.to("cpu")
            with torch.inference_mode():
                return self._model([tensor])[0]

    def _load_model(self, preferred_device: str) -> None:
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn

        device_order = [preferred_device] if preferred_device else []
        if torch.cuda.is_available() and "cuda" not in device_order:
            device_order.append("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and "mps" not in device_order:
            device_order.append("mps")
        if "cpu" not in device_order:
            device_order.append("cpu")

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self._categories = list(weights.meta.get("categories", []))

        last_error = None
        for device_name in device_order:
            try:
                model = maskrcnn_resnet50_fpn(weights=weights)
                model.eval()
                model.to(device_name)
                self._model = model
                self._device = device_name
                return
            except Exception as exc:  # pragma: no cover - depends on local torch backend support
                last_error = exc

        raise RuntimeError(f"Unable to load the segmentation model: {last_error}")


class DetectObjectsWorker(QThread):
    status = Signal(str)
    candidates_ready = Signal(object, str)
    failed = Signal(str)

    def __init__(
        self,
        frame_rgb: np.ndarray,
        manager: SegmentationModelManager,
        preferred_device: str,
        score_threshold: float,
    ):
        super().__init__()
        self._frame_rgb = frame_rgb
        self._manager = manager
        self._preferred_device = preferred_device
        self._score_threshold = score_threshold

    def run(self) -> None:
        self.status.emit("Running instance segmentation on frame 0...")
        try:
            candidates, device_name = self._manager.detect_candidates(
                self._frame_rgb,
                self._preferred_device,
                self._score_threshold,
            )
            self.candidates_ready.emit(candidates, device_name)
        except Exception as exc:  # pragma: no cover - depends on local model/runtime
            self.failed.emit(str(exc))


class GenerateMaskHintWorker(QThread):
    status = Signal(str)
    progress = Signal(int, int)
    preview = Signal(int, object, object)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        source_path: str,
        clip_root: str,
        selection: SegmentationCandidate,
        manager: SegmentationModelManager,
        preferred_device: str,
        score_threshold: float,
        blur_size: int,
        erode_size: int,
    ):
        super().__init__()
        self._source_path = source_path
        self._clip_root = clip_root
        self._selection = selection
        self._manager = manager
        self._preferred_device = preferred_device
        self._score_threshold = score_threshold
        self._blur_size = blur_size
        self._erode_size = erode_size

    def run(self) -> None:
        alpha_dir = os.path.join(self._clip_root, "AlphaHint")
        cap = None
        try:
            self.status.emit("Generating mask hints with Mask R-CNN...")
            _clear_alpha_assets(self._clip_root)
            os.makedirs(alpha_dir, exist_ok=True)

            cap = cv2.VideoCapture(self._source_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open source video: {self._source_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            tracker = _create_tracker()
            previous_box = self._selection.box

            frame_index = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if frame_index == 0:
                    chosen_mask = self._selection.mask
                    if tracker is not None:
                        tracker.init(frame_bgr, _xyxy_to_xywh(previous_box))
                else:
                    tracked_box = None
                    if tracker is not None:
                        tracker_ok, tracker_box = tracker.update(frame_bgr)
                        if tracker_ok:
                            tracked_box = _xywh_to_xyxy(tracker_box)

                    candidates, _device_name = self._manager.detect_candidates(
                        frame_rgb,
                        self._preferred_device,
                        self._score_threshold,
                    )
                    best = self._choose_candidate(candidates, tracked_box or previous_box)

                    if best is not None:
                        chosen_mask = best.mask
                        previous_box = best.box
                        if tracker is not None:
                            tracker = _create_tracker()
                            if tracker is not None:
                                tracker.init(frame_bgr, _xyxy_to_xywh(previous_box))
                    elif tracked_box is not None:
                        previous_box = _clip_box(tracked_box, frame_rgb.shape[1], frame_rgb.shape[0])
                        chosen_mask = _rectangle_mask(frame_rgb.shape[:2], previous_box)
                    else:
                        chosen_mask = np.zeros(frame_rgb.shape[:2], dtype=np.float32)

                hint = _coarse_hint_from_mask(chosen_mask, self._blur_size, self._erode_size)
                out_path = os.path.join(alpha_dir, f"frame_{frame_index:06d}.png")
                cv2.imwrite(out_path, hint)
                self.progress.emit(frame_index + 1, total_frames)
                self.preview.emit(frame_index, frame_rgb, hint)
                frame_index += 1

            if frame_index == 0:
                raise RuntimeError("No frames were read from the source video")

            self.completed.emit(alpha_dir)
        except Exception as exc:  # pragma: no cover - depends on local model/runtime
            self.failed.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()

    def _choose_candidate(
        self,
        candidates: list[SegmentationCandidate],
        reference_box: tuple[int, int, int, int],
    ) -> SegmentationCandidate | None:
        filtered = [item for item in candidates if item.label_id == self._selection.label_id]
        if not filtered:
            filtered = candidates
        if not filtered:
            return None

        scored: list[tuple[float, SegmentationCandidate]] = []
        for candidate in filtered:
            overlap = _box_iou(candidate.box, reference_box)
            weighted_score = candidate.score + (overlap * 2.0)
            scored.append((weighted_score, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]


class ExportWorker(QThread):
    status = Signal(str)
    progress = Signal(int, int)
    preview = Signal(int, object)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        service: CorridorKeyService,
        clip: ClipEntry,
        params: InferenceParams,
        output_config: OutputConfig,
    ):
        super().__init__()
        self._service = service
        self._clip = clip
        self._params = params
        self._output_config = output_config

    def run(self) -> None:
        try:
            self.status.emit("Exporting CorridorKey outputs...")

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
                on_progress=on_progress,
                output_config=self._output_config,
            )
            self.completed.emit(os.path.join(self._clip.root_path, "Output"))
        except Exception as exc:  # pragma: no cover - depends on local engine/runtime
            self.failed.emit(str(exc))


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
        self.selected_candidate_index: int | None = None
        self.detect_worker: DetectObjectsWorker | None = None
        self.mask_worker: GenerateMaskHintWorker | None = None
        self.export_worker: ExportWorker | None = None

        self._build_ui()
        self._set_busy(False)
        self._refresh_actions()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.service.unload_engines()
        super().closeEvent(event)

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
        self.device_label = QLabel(f"Inference device: {self.device}")
        self.segment_device_label = QLabel("Segmentation device: pending")
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

        segment_group = QGroupBox("Mask Generation")
        segment_layout = QFormLayout(segment_group)
        self.score_spin = QDoubleSpinBox()
        self.score_spin.setRange(0.05, 0.99)
        self.score_spin.setSingleStep(0.05)
        self.score_spin.setDecimals(2)
        self.score_spin.setValue(0.55)

        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(0, 99)
        self.blur_spin.setSingleStep(2)
        self.blur_spin.setValue(19)

        self.erode_spin = QSpinBox()
        self.erode_spin.setRange(0, 31)
        self.erode_spin.setValue(5)

        self.detect_button = QPushButton("Detect Objects")
        self.detect_button.clicked.connect(self._detect_objects)
        self.generate_button = QPushButton("Generate Mask Hint")
        self.generate_button.clicked.connect(self._generate_mask_hint)

        segment_layout.addRow("Score threshold", self.score_spin)
        segment_layout.addRow("Blur size", self.blur_spin)
        segment_layout.addRow("Erode size", self.erode_spin)
        segment_layout.addRow(self.detect_button)
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

        inference_layout.addRow("Input gamma", self.gamma_combo)
        inference_layout.addRow("Despill (0-10)", self.despill_spin)
        inference_layout.addRow(self.auto_despeckle_check)
        inference_layout.addRow("Despeckle size", self.despeckle_size_spin)
        inference_layout.addRow("Refiner scale", self.refiner_spin)
        layout.addWidget(inference_group)

        export_group = QGroupBox("Export")
        export_layout = QFormLayout(export_group)
        self.export_mode_combo = QComboBox()
        self.export_mode_combo.addItems(["Preview PNG", "Matte + Preview", "Full Package"])
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self._export_outputs)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label = QLabel("0 / 0")
        export_layout.addRow("Export mode", self.export_mode_combo)
        export_layout.addRow(self.export_button)
        export_layout.addRow(self.progress_bar)
        export_layout.addRow(self.progress_label)
        layout.addWidget(export_group)

        self.status_label = QLabel("Load a source video to begin.")
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
            self.selected_candidate_index = None

            self.source_edit.setText(path)
            self.alpha_edit.clear()
            self.project_label.setText(project_dir)
            self._reset_progress()
            self._configure_slider(self.source_reader.frame_count)
            self._update_previews()
            self._log(f"Loaded source video: {path}")
            self._set_status("Source video loaded. Drop a mask hint or detect objects on frame 0.")
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
            self.selected_candidate_index = None
            self._update_previews()
            self._log(f"Imported mask hint: {target}")
            self._set_status("Mask hint loaded. Export is now available.")
        except Exception as exc:
            self._show_error(str(exc))
            return

        self._refresh_actions()

    def _detect_objects(self) -> None:
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

        self.detect_worker = DetectObjectsWorker(
            frame_rgb=frame_rgb,
            manager=self.segmentation_manager,
            preferred_device=self.device,
            score_threshold=self.score_spin.value(),
        )
        self.detect_worker.status.connect(self._set_status)
        self.detect_worker.candidates_ready.connect(self._on_candidates_ready)
        self.detect_worker.failed.connect(self._show_error)
        self.detect_worker.finished.connect(self._on_worker_finished)
        self.detect_worker.start()
        self._set_busy(True)

    def _on_candidates_ready(self, candidates: list[SegmentationCandidate], device_name: str) -> None:
        self.candidates = candidates
        self.selected_candidate_index = None
        self.segment_device_label.setText(f"Segmentation device: {device_name}")
        self._update_previews()
        if candidates:
            self._set_status("Click the object you want in the top preview, then generate the mask hint.")
            self._log(f"Detected {len(candidates)} candidate objects on frame 0.")
        else:
            self._set_status("No candidate objects were found on frame 0.")
            self._log("Segmentation model found no objects on frame 0.")
        self._refresh_actions()

    def _handle_video_click(self, image_x: int, image_y: int) -> None:
        if self.current_frame_index != 0 or not self.candidates:
            return
        if self.source_reader is None:
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
            self._set_status("That click did not land on a detected object.")
            return

        self.selected_candidate_index = hit_index
        candidate = self.candidates[hit_index]
        self.mask_view.set_rgb_image(_mask_to_preview(_coarse_hint_from_mask(candidate.mask, 0, 0)))
        self._update_previews()
        self._set_status(f"Selected '{candidate.label_name}'. Generate the mask hint when ready.")
        self._refresh_actions()

    def _generate_mask_hint(self) -> None:
        if self.clip is None or self.source_reader is None:
            self._show_error("Load a source video first.")
            return
        if self.selected_candidate_index is None:
            self._show_error("Detect objects first, then click the one you want.")
            return

        selection = self.candidates[self.selected_candidate_index]
        self.mask_worker = GenerateMaskHintWorker(
            source_path=self.clip.input_asset.path,
            clip_root=self.clip.root_path,
            selection=selection,
            manager=self.segmentation_manager,
            preferred_device=self.device,
            score_threshold=self.score_spin.value(),
            blur_size=self.blur_spin.value(),
            erode_size=self.erode_spin.value(),
        )
        self.mask_worker.status.connect(self._set_status)
        self.mask_worker.progress.connect(self._set_progress)
        self.mask_worker.preview.connect(self._on_mask_preview)
        self.mask_worker.completed.connect(self._on_mask_generation_completed)
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

        self.export_worker = ExportWorker(
            service=self.service,
            clip=self.clip,
            params=params,
            output_config=output_config,
        )
        self.export_worker.status.connect(self._set_status)
        self.export_worker.progress.connect(self._set_progress)
        self.export_worker.preview.connect(self._on_export_preview)
        self.export_worker.completed.connect(self._on_export_completed)
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
        self.detect_worker = None
        self.mask_worker = None
        self.export_worker = None
        self._refresh_actions()

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
            selected_index = self.selected_candidate_index if self.current_frame_index == 0 else None
            if self.current_frame_index == 0 and self.candidates:
                self.video_view.set_rgb_image(_overlay_candidates(source_frame, self.candidates, selected_index))
            else:
                self.video_view.set_rgb_image(source_frame)

        if self.alpha_reader is not None:
            alpha_frame = self.alpha_reader.read_rgb(self.current_frame_index)
            if alpha_frame is not None:
                self.mask_view.set_rgb_image(alpha_frame)
        elif self.selected_candidate_index is not None:
            selected = self.candidates[self.selected_candidate_index]
            self.mask_view.set_rgb_image(_mask_to_preview(_coarse_hint_from_mask(selected.mask, 0, 0)))
        else:
            self.mask_view.set_placeholder("Mask Hint\nDrop a mask asset or generate one")

        total = max(self.source_reader.frame_count, 1)
        self.frame_label.setText(f"Frame {self.current_frame_index + 1} / {total}")

    def _refresh_actions(self) -> None:
        busy = self.detect_worker is not None or self.mask_worker is not None or self.export_worker is not None
        has_source = self.clip is not None and self.source_reader is not None
        has_selection = self.selected_candidate_index is not None
        has_alpha = self.clip is not None and self.clip.alpha_asset is not None

        self.detect_button.setEnabled(has_source and not busy)
        self.generate_button.setEnabled(has_source and has_selection and not busy)
        self.export_button.setEnabled(has_source and has_alpha and not busy)
        self.frame_slider.setEnabled(has_source and not busy)
        self.source_edit.setEnabled(not busy)
        self.alpha_edit.setEnabled(not busy)
        self.source_browse_button.setEnabled(not busy)
        self.alpha_browse_button.setEnabled(not busy)

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
        self._log(f"ERROR: {message}")
        QMessageBox.critical(self, "CorridorKey Qt GUI", message)

    def _log(self, message: str) -> None:
        if not message:
            return
        self.log_view.appendPlainText(message)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app = QApplication([])
    app.setApplicationName("CorridorKey Qt Preview")
    window = CorridorKeyWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

"""python detectAll/medium/run_overlay.py

Screen capture + YOLO inference with click-through transparent overlay.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import mss
import numpy as np

import torch
from ultralytics import YOLO

from common import load_config, resolve_path
from tracker import Detection, TrackedObject, MultiObjectTracker, create_tracker_from_config


def _load_qt():
    from PyQt5.QtCore import Qt, QTimer, QRectF, QMetaObject, Q_ARG
    from PyQt5.QtGui import QColor, QFont, QPainter, QPen
    from PyQt5.QtWidgets import QApplication, QWidget
    return Qt, QTimer, QRectF, QColor, QFont, QPainter, QPen, QApplication, QWidget, QMetaObject


class OverlayApp:
    def __init__(self, config_path: Path) -> None:
        self.ensure_venv_requirements()
        self.config = load_config(config_path)
        self.infer_cfg = self.config["infer"]
        self.overlay_cfg = self.config["overlay"]
        self.data_yaml = resolve_path(self.config["paths"]["data_yaml"])

        try:
            Qt, QTimer, QRectF, QColor, QFont, QPainter, QPen, QApplication, QWidget, QMetaObject = _load_qt()
        except Exception as exc:
            raise RuntimeError("PyQt5 is required for the overlay. Please install it and retry.") from exc
        
        self._QMetaObject = QMetaObject

        class OverlayWidget(QWidget):
            def __init__(
                self,
                screen_geometry,
                box_color: Tuple[int, int, int],
                text_color: Tuple[int, int, int],
                text_bg_color: Tuple[int, int, int],
                thickness: int,
                font_scale: float,
                show_track_id: bool = False,
                show_velocity: bool = False,
                capture_exclusion_method: str = "api",
            ) -> None:
                super().__init__()
                self._lock = threading.Lock()
                self._tracks: List[TrackedObject] = []
                self._box_color = QColor(*box_color)
                self._text_color = QColor(*text_color)
                self._text_bg = QColor(*text_bg_color)
                self._thickness = thickness
                self._font_scale = font_scale
                self._show_track_id = show_track_id
                self._show_velocity = show_velocity
                self._capture_exclusion_method = capture_exclusion_method
                self._hwnd = None

                self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
                self.setAttribute(Qt.WA_TranslucentBackground, True)
                self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
                self.setAttribute(Qt.WA_NoSystemBackground, True)
                self.setGeometry(screen_geometry)
                self.show()
                
                self._apply_capture_exclusion()
            
            def _apply_capture_exclusion(self) -> None:
                method = self._capture_exclusion_method.lower()
                print(f"[Overlay] Using capture exclusion method: {method}")
                
                if method == "none":
                    print("[Overlay] Capture exclusion disabled")
                    return
                
                try:
                    import ctypes
                    self._hwnd = int(self.winId())
                    user32 = ctypes.windll.user32
                    
                    if method == "api":
                        self._try_api_exclusion(user32)
                    elif method == "layered":
                        self._try_layered_exclusion(user32)
                    elif method == "click_through":
                        self._try_clickthrough_exclusion(user32)
                    else:
                        print(f"[Overlay] Unknown method '{method}', trying 'api'")
                        self._try_api_exclusion(user32)
                        
                except Exception as e:
                    print(f"[Overlay] Capture exclusion failed: {e}")
            
            def _try_api_exclusion(self, user32) -> bool:
                WDA_EXCLUDEFROMCAPTURE = 0x00000011
                result = user32.SetWindowDisplayAffinity(self._hwnd, WDA_EXCLUDEFROMCAPTURE)
                
                if result:
                    print("[Overlay] [OK] API exclusion active (WDA_EXCLUDEFROMCAPTURE)")
                    return True
                
                WDA_MONITOR = 0x00000001
                result = user32.SetWindowDisplayAffinity(self._hwnd, WDA_MONITOR)
                if result:
                    print("[Overlay] [OK] API exclusion active (WDA_MONITOR fallback)")
                    return True
                
                import ctypes
                error = ctypes.get_last_error()
                print(f"[Overlay] [FAIL] API exclusion failed (error {error})")
                return False
            
            def _try_layered_exclusion(self, user32) -> bool:
                import ctypes
                
                GWL_EXSTYLE = -20
                WS_EX_LAYERED = 0x00080000
                LWA_COLORKEY = 0x00000001
                
                ex_style = user32.GetWindowLongW(self._hwnd, GWL_EXSTYLE)
                
                user32.SetWindowLongW(self._hwnd, GWL_EXSTYLE, ex_style | WS_EX_LAYERED)
                
                result = user32.SetLayeredWindowAttributes(
                    self._hwnd, 
                    0x00FF00FF,
                    255,
                    LWA_COLORKEY
                )
                
                if result:
                    print("[Overlay] [OK] Layered window exclusion applied")
                    return True
                
                error = ctypes.get_last_error()
                print(f"[Overlay] [FAIL] Layered exclusion failed (error {error})")
                return False
            
            def _try_clickthrough_exclusion(self, user32) -> bool:
                import ctypes
                
                GWL_EXSTYLE = -20
                WS_EX_TRANSPARENT = 0x00000020
                WS_EX_LAYERED = 0x00080000
                
                ex_style = user32.GetWindowLongW(self._hwnd, GWL_EXSTYLE)
                
                new_style = ex_style | WS_EX_TRANSPARENT | WS_EX_LAYERED
                result = user32.SetWindowLongW(self._hwnd, GWL_EXSTYLE, new_style)
                
                if result or user32.GetWindowLongW(self._hwnd, GWL_EXSTYLE) == new_style:
                    print("[Overlay] [OK] Click-through exclusion applied")
                    return True
                
                error = ctypes.get_last_error()
                print(f"[Overlay] [FAIL] Click-through exclusion failed (error {error})")
                return False

            def set_tracks(self, tracks: List[TrackedObject]) -> None:
                with self._lock:
                    self._tracks = tracks

            def paintEvent(self, _event) -> None:
                painter = QPainter(self)
                painter.setRenderHint(QPainter.Antialiasing)

                with self._lock:
                    tracks = list(self._tracks)

                font_size = max(8, int(12 * self._font_scale))
                font = QFont("Arial", font_size, QFont.Bold)
                painter.setFont(font)
                fm = painter.fontMetrics()

                for track in tracks:
                    x1, y1, x2, y2 = track.get_display_bbox()
                    
                    conf = track.smoothed_confidence
                    
                    pen = QPen(self._box_color)
                    pen.setWidth(self._thickness)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    rect = QRectF(x1, y1, x2 - x1, y2 - y1)
                    painter.drawRect(rect)

                    if self._show_track_id:
                        label_text = f"[{track.track_id}] {track.label} {conf:.2f}"
                    else:
                        label_text = f"{track.label} {conf:.2f}"
                    
                    text_w = fm.horizontalAdvance(label_text)
                    text_h = fm.height()
                    text_x = x1 + 4
                    text_y = max(0.0, y1 - text_h - 4)

                    painter.setPen(Qt.NoPen)
                    painter.setBrush(self._text_bg)
                    painter.drawRect(QRectF(text_x - 2, text_y - 2, text_w + 6, text_h + 4))

                    painter.setPen(self._text_color)
                    painter.drawText(QRectF(text_x, text_y, text_w + 2, text_h + 2), Qt.AlignLeft | Qt.AlignVCenter, label_text)
                    
                    if self._show_velocity:
                        vx, vy = track.get_velocity()
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        scale = 5.0
                        pen = QPen(QColor(255, 255, 0))
                        pen.setWidth(2)
                        painter.setPen(pen)
                        painter.drawLine(int(cx), int(cy), int(cx + vx * scale), int(cy + vy * scale))

        self._Qt = Qt
        self._QTimer = QTimer
        self._QApplication = QApplication
        self._OverlayWidget = OverlayWidget

        self.app = self._QApplication(sys.argv)
        self.device_setting = self.resolve_device_setting()
        self.model = YOLO(self.infer_cfg["model_path"])
        self.model.fuse()

        with mss.mss() as mss_instance:
            self.monitor = mss_instance.monitors[int(self.overlay_cfg["monitor_index"])]
        self.mss_instance = None
        self.screen_width = int(self.monitor["width"])
        self.screen_height = int(self.monitor["height"])
        self.capture_scale = float(self.overlay_cfg["capture_scale"])
        self.target_fps = float(self.overlay_cfg["target_fps"])

        screen = self._resolve_qt_screen()
        screen_geometry = screen.geometry()

        self.latest_tracks: List[TrackedObject] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
        self.tracker = create_tracker_from_config(self.overlay_cfg)
        print(f"[Tracker] Initialized with IoU threshold={self.tracker.config.iou_threshold}, "
              f"max coasting={self.tracker.config.max_age_coasting} frames")

        self.overlay = self._OverlayWidget(
            screen_geometry,
            box_color=tuple(self.overlay_cfg["box_color"]),
            text_color=tuple(self.overlay_cfg["text_color"]),
            text_bg_color=tuple(self.overlay_cfg["text_bg_color"]),
            thickness=int(self.overlay_cfg["line_thickness"]),
            font_scale=float(self.overlay_cfg["font_scale"]),
            show_track_id=bool(self.overlay_cfg.get("show_track_id", False)),
            show_velocity=bool(self.overlay_cfg.get("show_velocity", False)),
            capture_exclusion_method=str(self.overlay_cfg.get("capture_exclusion_method", "api")),
        )

        self.start_hotkey_monitor()

    def _resolve_qt_screen(self):
        screens = self.app.screens()
        monitor_index = int(self.overlay_cfg.get("monitor_index", 1))
        if 1 <= monitor_index <= len(screens):
            return screens[monitor_index - 1]
        return self.app.primaryScreen()

    def start(self) -> None:
        inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        inference_thread.start()

        interval_ms = int(1000 / max(1.0, self.target_fps))
        timer = self._QTimer()
        timer.timeout.connect(self._tick)
        timer.start(interval_ms)

        sys.exit(self.app.exec_())

    def _tick(self) -> None:
        with self.lock:
            tracks = list(self.latest_tracks)
        self.overlay.set_tracks(tracks)
        self.overlay.update()

    def start_hotkey_monitor(self) -> None:
        try:
            import ctypes
            user32 = ctypes.windll.user32
        except Exception:
            return

        poll_ms = int(self.overlay_cfg.get("exit_hotkey_poll_ms", 100))

        def watcher():
            while not self.stop_event.is_set():
                if user32.GetAsyncKeyState(0x1B) & 1:  # VK_ESCAPE
                    self.stop()
                    break
                if user32.GetAsyncKeyState(0x7B) & 1:  # VK_F12
                    self.stop()
                    break
                time.sleep(max(0.02, poll_ms / 1000.0))

        threading.Thread(target=watcher, daemon=True).start()

    def inference_loop(self) -> None:
        self.mss_instance = mss.mss()
        self.monitor = self.mss_instance.monitors[int(self.overlay_cfg["monitor_index"])]
        self.screen_width = int(self.monitor["width"])
        self.screen_height = int(self.monitor["height"])

        frame_delay = max(0.001, 1.0 / max(1.0, self.target_fps))
        imgsz = int(self.infer_cfg["imgsz"])
        conf = float(self.infer_cfg["conf"])
        iou = float(self.infer_cfg["iou"])
        max_det = int(self.infer_cfg.get("max_det", 300))
        agnostic_nms = bool(self.infer_cfg.get("agnostic_nms", False))
        half = bool(self.infer_cfg["half"]) and self.device_setting != "cpu"
        device = self.device_setting

        frame_count = 0
        
        while not self.stop_event.is_set():
            start = time.time()
            
            frame = np.array(self.mss_instance.grab(self.monitor))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if self.capture_scale != 1.0:
                scaled_width = max(1, int(self.screen_width * self.capture_scale))
                scaled_height = max(1, int(self.screen_height * self.capture_scale))
                frame_bgr = cv2.resize(frame_bgr, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
            else:
                scaled_width = self.screen_width
                scaled_height = self.screen_height

            results = self.model.predict(
                source=frame_bgr,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=max_det,
                agnostic_nms=agnostic_nms,
                device=device,
                half=half,
                verbose=False,
            )

            detections: List[Detection] = []
            if results:
                result = results[0]
                names = result.names
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    label = names.get(class_id, str(class_id))

                    scale_x = self.screen_width / float(scaled_width)
                    scale_y = self.screen_height / float(scaled_height)

                    detections.append(
                        Detection(
                            x1=x1 * scale_x,
                            y1=y1 * scale_y,
                            x2=x2 * scale_x,
                            y2=y2 * scale_y,
                            label=label,
                            confidence=confidence,
                        )
                    )

            active_tracks = self.tracker.update(detections)

            with self.lock:
                self.latest_tracks = active_tracks

            frame_count += 1
            elapsed = time.time() - start
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self._QTimer.singleShot(0, self.overlay.close)
        self._QTimer.singleShot(0, self.app.quit)

    def resolve_device_setting(self):
        device_setting = self.infer_cfg.get("device", "auto")
        if isinstance(device_setting, str) and device_setting.lower() in {"auto", "cuda"}:
            return 0 if torch.cuda.is_available() else "cpu"
        if str(device_setting) != "cpu" and not torch.cuda.is_available():
            print("CUDA not available for overlay. Falling back to CPU.")
            return "cpu"
        return device_setting

    def ensure_venv_requirements(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        venv_python = project_root / ".venv" / "Scripts" / "python.exe"
        if (
            venv_python.exists()
            and sys.prefix == sys.base_prefix
            and os.environ.get("POLLINATOR_REEXEC") != "1"
        ):
            os.environ["POLLINATOR_REEXEC"] = "1"
            os.execv(str(venv_python), [str(venv_python), *sys.argv])

        if sys.prefix != sys.base_prefix:
            requirements_path = Path(__file__).parent / "requirements.txt"
            try:
                import PyQt5  # noqa: F401
                import mss  # noqa: F401
                import cv2  # noqa: F401
                import ultralytics  # noqa: F401
            except Exception:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
                    check=True,
                )


def main() -> None:
    app = OverlayApp(Path(__file__).parent / "config.yaml")
    app.start()


if __name__ == "__main__":
    main()

"""Interface gráfica: câmera ou vídeo, caixas coloridas e a frase falada."""

from __future__ import annotations

import logging
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image

from obstaclesense.config import save_config
from obstaclesense.pipeline import ObstaclePipeline

logger = logging.getLogger(__name__)

PREDEFINED_MODELS = {
    "YOLO11n (recomendado)": "yolo11n.pt",
    "YOLO11s": "yolo11s.pt",
    "YOLOv8n": "yolov8n.pt",
    "YOLOv8s": "yolov8s.pt",
    "YOLOv8m": "yolov8m.pt",
}


class NavigationApp:
    def __init__(self, config: dict):
        self.config = config
        self.pipeline: ObstaclePipeline | None = None
        self.capture = None
        self.running = False
        self.thread: threading.Thread | None = None
        self.video_path = ""
        self._photo = None
        self._closing = False
        self._lock = threading.Lock()
        self._pending = None
        self._loaded = None
        self._load_error = None
        self._loading = False
        self._speak_enabled = bool(config.get("audio_enabled", True))
        self._loop_flag = bool(config.get("video_loop", False))

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.window = ctk.CTk()
        self.window.title("ObstacleSense AI")
        self.window.geometry(
            f"{int(config.get('window_width', 1280))}x{int(config.get('window_height', 920))}"
        )
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.window.bind("<space>", lambda _event: self.toggle())
        self.window.bind("<Escape>", lambda _event: self.on_close())

        self.camera_index = tk.IntVar(value=int(config.get("camera_index", 0)))
        self.source_mode = tk.StringVar(value="camera")
        initial_model = str(config.get("yolo_model_path", "yolo11n.pt"))
        preset = next((name for name, path in PREDEFINED_MODELS.items() if path == initial_model), None)
        self.model_choice = tk.StringVar(value=preset or "YOLO11n (recomendado)")
        self.custom_model = tk.StringVar(value="" if preset else initial_model)
        self.use_gpu = tk.BooleanVar(value=bool(config.get("enable_gpu", False)))
        self.confidence = tk.DoubleVar(value=float(config.get("confidence_threshold", 0.45)))
        self.audio_on = tk.BooleanVar(value=bool(config.get("audio_enabled", True)))
        self.loop_video = tk.BooleanVar(value=bool(config.get("video_loop", False)))

        self._build()
        self.status = ctk.StringVar(value="Carregando modelo...")
        ctk.CTkLabel(self.window, textvariable=self.status, height=28).pack(fill="x", padx=10, pady=(0, 8))
        self.window.after(50, self._kickoff_load)
        self.window.after(30, self._poll_ui)

    def _build(self) -> None:
        body = ctk.CTkFrame(self.window)
        body.pack(fill="both", expand=True, padx=10, pady=10)

        left = ctk.CTkFrame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.video_label = ctk.CTkLabel(left, text="Aguardando o modelo...")
        self.video_label.pack(fill="both", expand=True, padx=8, pady=8)

        bar = ctk.CTkFrame(left)
        bar.pack(fill="x", padx=8, pady=(0, 8))
        self.toggle_button = ctk.CTkButton(bar, text="Iniciar", command=self.toggle, state="disabled")
        self.toggle_button.pack(side="left", padx=6, pady=6)
        self.metrics = ctk.StringVar(value="FPS 0   0 ms   0 objetos")
        ctk.CTkLabel(bar, textvariable=self.metrics).pack(side="left", padx=8)

        self.guidance = ctk.StringVar(value="A orientação falada aparece aqui.")
        ctk.CTkLabel(
            left,
            textvariable=self.guidance,
            wraplength=700,
            justify="left",
            font=ctk.CTkFont(size=16),
        ).pack(fill="x", padx=8, pady=(0, 8))

        right = ctk.CTkFrame(body, width=360)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        tabs = ctk.CTkTabview(right)
        tabs.pack(fill="both", expand=True, padx=6, pady=6)
        for name in ("Entrada", "Modelo", "Detecção"):
            tabs.add(name)
        self._input_tab(tabs.tab("Entrada"))
        self._model_tab(tabs.tab("Modelo"))
        self._detect_tab(tabs.tab("Detecção"))
        ctk.CTkButton(right, text="Salvar configurações", command=self._save).pack(fill="x", padx=8, pady=8)

    def _input_tab(self, tab) -> None:
        ctk.CTkRadioButton(tab, text="Câmera", variable=self.source_mode, value="camera").pack(anchor="w", padx=10, pady=6)
        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.pack(fill="x", padx=10)
        ctk.CTkLabel(row, text="Índice").pack(side="left")
        ctk.CTkEntry(row, textvariable=self.camera_index, width=60).pack(side="left", padx=6)
        ctk.CTkRadioButton(tab, text="Arquivo de vídeo", variable=self.source_mode, value="file").pack(
            anchor="w", padx=10, pady=6
        )
        ctk.CTkButton(tab, text="Escolher vídeo", command=self._browse).pack(anchor="w", padx=10)
        self.file_label = ctk.CTkLabel(tab, text="Nenhum arquivo.", wraplength=300)
        self.file_label.pack(anchor="w", padx=10, pady=4)
        ctk.CTkCheckBox(tab, text="Repetir vídeo", variable=self.loop_video).pack(anchor="w", padx=10, pady=4)

    def _model_tab(self, tab) -> None:
        ctk.CTkLabel(tab, text="Modelo").pack(anchor="w", padx=10, pady=(8, 2))
        ctk.CTkComboBox(tab, values=list(PREDEFINED_MODELS), variable=self.model_choice).pack(fill="x", padx=10)
        ctk.CTkLabel(tab, text="Ou um arquivo .pt / .onnx").pack(anchor="w", padx=10, pady=(10, 2))
        ctk.CTkEntry(tab, textvariable=self.custom_model).pack(fill="x", padx=10)
        ctk.CTkButton(tab, text="Buscar modelo", command=self._browse_model).pack(anchor="w", padx=10, pady=6)
        ctk.CTkCheckBox(tab, text="Usar GPU se existir", variable=self.use_gpu).pack(anchor="w", padx=10, pady=4)
        ctk.CTkButton(tab, text="Recarregar modelo", command=self._reload).pack(fill="x", padx=10, pady=10)

    def _detect_tab(self, tab) -> None:
        ctk.CTkLabel(tab, text="Confiança").pack(anchor="w", padx=10, pady=(8, 2))
        ctk.CTkSlider(tab, from_=0.05, to=0.95, variable=self.confidence, command=self._on_conf).pack(
            fill="x", padx=10
        )
        self.conf_label = ctk.CTkLabel(tab, text=f"{self.confidence.get():.2f}")
        self.conf_label.pack(anchor="e", padx=10)
        ctk.CTkCheckBox(tab, text="Falar orientação", variable=self.audio_on).pack(anchor="w", padx=10, pady=8)
        ctk.CTkLabel(
            tab,
            text=(
                f"Intervalo de fala: {self.config.get('audio_interval', 2)} s\n"
                f"Alerta de proximidade: {self.config.get('proximity_warning_distance', 2)} m\n"
                f"Limiar de colisão: {self.config.get('collision_warning_threshold', 0.7)}"
            ),
            justify="left",
        ).pack(anchor="w", padx=10, pady=6)

    def _kickoff_load(self) -> None:
        if self._loading or self._closing:
            return
        self._apply_model_choice()
        self._loading = True
        self._load_error = None
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self) -> None:
        try:
            pipeline = ObstaclePipeline(self.config)
            with self._lock:
                self._loaded = pipeline
        except Exception as exc:
            logger.exception("Falha ao carregar o modelo.")
            with self._lock:
                self._load_error = exc
        finally:
            self._loading = False

    def _poll_ui(self) -> None:
        if self._closing:
            return
        loaded = None
        error = None
        result = None
        with self._lock:
            if self._loaded is not None:
                loaded = self._loaded
                self._loaded = None
            error = self._load_error
            self._load_error = None
            if self._pending is not None:
                result = self._pending
                self._pending = None
        if loaded is not None:
            if self.pipeline is not None:
                self.pipeline.close()
            self.pipeline = loaded
            self.toggle_button.configure(state="normal")
            self.status.set(f"Modelo {loaded.model_name} em {loaded.device}.")
        elif error is not None and self.pipeline is None:
            self.status.set(f"Falha ao carregar o modelo: {error}")
            messagebox.showerror("Modelo", str(error))
        if result is not None:
            self._show(result)
        self._speak_enabled = bool(self.audio_on.get())
        self._loop_flag = bool(self.loop_video.get())
        if self.pipeline and self.pipeline.speaker is not None and self.pipeline.speaker.ready:
            self.pipeline.speaker.enabled = self._speak_enabled
        if (
            not self.running
            and self.thread is not None
            and not self.thread.is_alive()
            and self.capture is not None
        ):
            self.stop()
        self.window.after(30, self._poll_ui)

    def _apply_model_choice(self) -> None:
        custom = self.custom_model.get().strip()
        if custom and Path(custom).exists():
            self.config["yolo_model_path"] = custom
        else:
            self.config["yolo_model_path"] = PREDEFINED_MODELS.get(self.model_choice.get(), "yolo11n.pt")
        self.config["enable_gpu"] = bool(self.use_gpu.get())
        self.config["device"] = "cuda" if self.use_gpu.get() else "cpu"
        self.config["confidence_threshold"] = float(self.confidence.get())
        self.config["audio_enabled"] = bool(self.audio_on.get())
        self.config["camera_index"] = int(self.camera_index.get())
        self.config["video_loop"] = bool(self.loop_video.get())

    def _reload(self) -> None:
        if self.running:
            self.stop()
        self.toggle_button.configure(state="disabled")
        self.status.set("Recarregando modelo...")
        if self.pipeline:
            self.pipeline.close()
            self.pipeline = None
        self._kickoff_load()

    def _on_conf(self, _value=None) -> None:
        value = float(self.confidence.get())
        self.conf_label.configure(text=f"{value:.2f}")
        self.config["confidence_threshold"] = value
        if self.pipeline and hasattr(self.pipeline.detector, "set_thresholds"):
            self.pipeline.detector.set_thresholds(confidence=value)

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Vídeo",
            filetypes=(("Vídeo", "*.mp4 *.avi *.mov *.mkv"), ("Todos", "*.*")),
        )
        if path:
            self.video_path = path
            self.source_mode.set("file")
            self.file_label.configure(text=Path(path).name)

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Modelo",
            filetypes=(("Modelo", "*.pt *.onnx"), ("Todos", "*.*")),
        )
        if path:
            self.custom_model.set(path)

    def _save(self) -> None:
        self._apply_model_choice()
        path = save_config(self.config)
        self.status.set(f"Configuração salva em {path.name}.")

    def toggle(self) -> None:
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        if self.pipeline is None:
            self.status.set("O modelo ainda não está pronto.")
            return
        self.config["audio_enabled"] = bool(self.audio_on.get())
        if self.pipeline.speaker is not None:
            self.pipeline.speaker.enabled = bool(self.audio_on.get())
        try:
            if self.source_mode.get() == "file":
                if not self.video_path:
                    raise FileNotFoundError("Escolha um arquivo de vídeo.")
                self.capture = cv2.VideoCapture(self.video_path)
            else:
                index = int(self.camera_index.get())
                backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
                self.capture = cv2.VideoCapture(index, backend)
                if not self.capture.isOpened():
                    self.capture = cv2.VideoCapture(index)
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config.get("frame_width", 640)))
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config.get("frame_height", 480)))
            if not self.capture.isOpened():
                raise FileNotFoundError("Não foi possível abrir a câmera ou o vídeo.")
        except Exception as exc:
            self.status.set(str(exc))
            messagebox.showerror("Entrada", str(exc))
            return
        self._speak_enabled = bool(self.audio_on.get())
        self._loop_flag = bool(self.loop_video.get())
        self.running = True
        self.toggle_button.configure(text="Parar")
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        thread = self.thread
        if thread and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=2.0)
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.toggle_button.configure(text="Iniciar")
        self.status.set("Detecção parada.")

    def _loop(self) -> None:
        while self.running and self.capture is not None:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                if self._loop_flag and self.video_path:
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.running = False
                break
            result = self.pipeline.process(frame, speak=bool(self._speak_enabled), annotate=True)
            with self._lock:
                self._pending = result
        logger.info("Laço da interface encerrado.")

    def _show(self, result) -> None:
        if self._closing or result.annotated is None:
            return
        frame = result.annotated
        label_w = max(self.video_label.winfo_width(), 640)
        label_h = max(self.video_label.winfo_height(), 360)
        scale = min(label_w / frame.shape[1], label_h / frame.shape[0])
        size = (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale)))
        resized = cv2.resize(frame, size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self._photo = ctk.CTkImage(image, size=size)
        self.video_label.configure(image=self._photo, text="")
        self.metrics.set(
            f"FPS {result.fps:.1f}   {result.latency_ms:.0f} ms   {len(result.objects)} objetos"
        )
        self.guidance.set(result.guidance)
        self.status.set(time.strftime("%H:%M:%S") + "  detecção em execução")

    def on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.running = False
        thread = self.thread
        if thread and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=2.0)
        if self.pipeline:
            self.pipeline.close()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        try:
            self.window.destroy()
        except Exception:
            logger.exception("Falha ao fechar a janela.")

    def run(self) -> None:
        self.window.mainloop()

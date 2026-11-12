"""Câmera, vídeo ou imagem no terminal, com overlay e voz."""

from __future__ import annotations

import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

import cv2

from obstaclesense.pipeline import ObstaclePipeline

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEMO_URL = "https://ultralytics.com/images/bus.jpg"
DEMO_IMAGE = Path("outputs/bus.jpg")


def run_demo(config: dict, output: str | None = None, show: bool = False) -> dict:
    DEMO_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    if not DEMO_IMAGE.exists():
        logger.info("Baixando imagem de demonstração...")
        urllib.request.urlretrieve(DEMO_URL, DEMO_IMAGE)
    destination = output or "outputs/demo.jpg"
    return run_source(
        str(DEMO_IMAGE),
        config,
        show=show,
        output=destination,
        max_frames=1,
        loop=False,
    )


def run_source(
    source: str,
    config: dict,
    show: bool = True,
    output: str | None = None,
    max_frames: int = 0,
    loop: bool = False,
) -> dict:
    pipeline = ObstaclePipeline(config)
    suffix = Path(source).suffix.lower()
    last = {}
    try:
        if suffix in IMAGE_SUFFIXES:
            frame = cv2.imread(source)
            if frame is None:
                raise FileNotFoundError(f"Não foi possível ler a imagem {source}")
            result = pipeline.process(frame, speak=True, annotate=True)
            last = result.to_dict()
            _print_result(last)
            if output and result.annotated is not None:
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(output, result.annotated)
                logger.info("Imagem anotada salva em %s", output)
            if show:
                _show_frame(result.annotated, wait=True)
            return last

        capture = _open_capture(source, config)
        if not capture.isOpened():
            raise FileNotFoundError(f"Não foi possível abrir a fonte {source}")
        writer = None
        seen = 0
        previous_guidance = None
        last_log = 0.0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                if loop and not source.isdigit():
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            result = pipeline.process(frame, speak=True, annotate=True)
            last = result.to_dict()
            seen += 1
            now = time.monotonic()
            if last["guidance"] != previous_guidance or now - last_log >= 1.0:
                _print_result(last)
                previous_guidance = last["guidance"]
                last_log = now
            if output:
                writer = _write_video(writer, output, result.annotated)
            if show and not _show_frame(result.annotated, wait=False):
                show = False
            if max_frames and seen >= max_frames:
                break
        capture.release()
        if writer is not None:
            writer.release()
            logger.info("Vídeo anotado salvo em %s", output)
        if output and writer is None and last and Path(output).suffix.lower() in IMAGE_SUFFIXES:
            # max_frames em vídeo pedido como imagem: fica o último quadro.
            pass
        return last
    finally:
        pipeline.close()


def _open_capture(source: str, config: dict):
    if source.isdigit():
        index = int(source)
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            capture = cv2.VideoCapture(index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(config.get("frame_width", 640)))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config.get("frame_height", 480)))
        return capture
    return cv2.VideoCapture(source)


def _write_video(writer, output: str, frame):
    if frame is None:
        return writer
    if writer is None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, 12.0, (width, height))
    writer.write(frame)
    return writer


def _show_frame(frame, wait: bool) -> bool:
    if frame is None:
        return False
    try:
        cv2.imshow("ObstacleSense AI", frame)
        key = cv2.waitKey(0 if wait else 1) & 0xFF
        if key in (ord("q"), 27):
            return False
        return True
    except cv2.error:
        logger.info("Sem janela de vídeo. Seguindo só pelo terminal.")
        return False


def _print_result(payload: dict) -> None:
    print(
        f"[{payload.get('fps', 0):.1f} FPS | {payload.get('latency_ms', 0):.0f} ms] "
        f"{payload.get('guidance', '')}"
    )
    for item in payload.get("objects", []):
        print(
            f"  - {item['label']} {item['confidence']:.2f}  "
            f"{item['distance_meters']:.2f} m  {item['direction']}  {item['danger_level']}"
        )
    sys.stdout.flush()


def dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)

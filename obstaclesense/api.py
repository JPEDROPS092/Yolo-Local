"""Serviço HTTP para o aplicativo Android enviar um quadro e receber a orientação."""

from __future__ import annotations

import asyncio
import base64
import json

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from obstaclesense import __version__
from obstaclesense.config import load_config
from obstaclesense.pipeline import ObstaclePipeline


class ObstacleOut(BaseModel):
    label: str
    confidence: float
    distance_meters: float
    direction: str
    danger_level: str


class AnalyzeResponse(BaseModel):
    guidance: str = Field(description="Orientação em português do Brasil.")
    latency_ms: float
    fps: float
    objects: list[ObstacleOut]


def create_app(pipeline: ObstaclePipeline | None = None, config: dict | None = None) -> FastAPI:
    app = FastAPI(
        title="ObstacleSense AI",
        version=__version__,
        description=(
            "Detecção de obstáculos para navegação de pessoas com deficiência visual. "
            "POST /analyze recebe uma imagem e devolve objetos e uma frase em pt-BR."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if pipeline is None:
        pipeline = ObstaclePipeline(config if config is not None else load_config())
    app.state.pipeline = pipeline

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "service": "ObstacleSense AI",
            "version": __version__,
            "model": pipeline.model_name,
            "device": pipeline.device,
            "depth_enabled": pipeline.depth_enabled,
            "classes": pipeline.class_names,
        }

    @app.post("/analyze", response_model=AnalyzeResponse)
    async def analyze(
        file: UploadFile | None = File(None),
        image: UploadFile | None = File(None),
    ):
        upload = file or image
        if upload is None:
            raise HTTPException(status_code=400, detail="envie uma imagem no campo file")
        raw = await upload.read()
        frame = _decode_image(raw)
        if frame is None:
            raise HTTPException(status_code=400, detail="imagem inválida")
        result = await asyncio.to_thread(pipeline.process, frame, False, False)
        return result.to_dict()

    @app.websocket("/ws")
    async def stream(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                raw = message.get("bytes")
                if raw is None and message.get("text"):
                    try:
                        payload = json.loads(message["text"])
                        raw = base64.b64decode(payload["image_base64"])
                    except Exception:
                        await websocket.send_json({"detail": "imagem inválida"})
                        continue
                frame = _decode_image(raw or b"")
                if frame is None:
                    await websocket.send_json({"detail": "imagem inválida"})
                    continue
                result = await asyncio.to_thread(pipeline.process, frame, False, False)
                await websocket.send_json(result.to_dict())
        except WebSocketDisconnect:
            return

    return app


def main() -> None:
    import uvicorn

    from obstaclesense.config import load_config

    config = load_config()
    uvicorn.run(
        create_app(config=config),
        host=str(config.get("api_host", "0.0.0.0")),
        port=int(config.get("api_port", 8000)),
    )


if __name__ == "__main__":
    main()


def _decode_image(raw: bytes):
    if not raw:
        return None
    array = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)

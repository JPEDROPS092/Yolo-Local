#!/usr/bin/env python3
"""ObstacleSense AI — detecção, voz e API para navegação assistiva."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from obstaclesense.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ObstacleSense AI: obstáculos à frente, distância e orientação em português."
    )
    parser.add_argument("--gui", action="store_true", help="Abre a interface gráfica.")
    parser.add_argument("--cli", action="store_true", help="Roda no terminal com câmera, vídeo ou imagem.")
    parser.add_argument("--api", action="store_true", help="Sobe o serviço HTTP para o aplicativo.")
    parser.add_argument("--demo", action="store_true", help="Analisa uma imagem de exemplo e salva o resultado.")
    parser.add_argument("--source", help="Índice da câmera, vídeo ou imagem.")
    parser.add_argument("--model", help="Caminho do modelo, por exemplo yolo11n.pt ou yolov8n.pt.")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Força CPU ou GPU.")
    parser.add_argument("--imgsz", type=int, help="Tamanho de inferência. 320 é o padrão leve.")
    parser.add_argument("--no-audio", action="store_true", help="Não fala as orientações.")
    parser.add_argument("--no-display", action="store_true", help="Não abre janela do OpenCV.")
    parser.add_argument("--depth", action="store_true", help="Liga o Depth Anything V2 Small.")
    parser.add_argument("--output", help="Arquivo anotado de saída.")
    parser.add_argument("--max-frames", type=int, default=0, help="Para depois de N quadros. 0 não limita.")
    parser.add_argument("--loop", action="store_true", help="Repete o arquivo de vídeo.")
    parser.add_argument("--export", choices=["onnx", "tflite"], help="Exporta o modelo e encerra.")
    parser.add_argument("--host", help="Endereço do serviço.")
    parser.add_argument("--port", type=int, help="Porta do serviço.")
    parser.add_argument("--config", help="Arquivo YAML de configuração.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    _apply_args(config, args)

    if args.export:
        from obstaclesense.export import export_model

        export_model(config["yolo_model_path"], args.export, imgsz=int(config["imgsz"]))
        return 0
    if args.api:
        _run_api(config)
        return 0
    if args.demo:
        from obstaclesense.cli import dumps, run_demo

        payload = run_demo(config, output=args.output, show=not args.no_display and _has_display())
        print(dumps(payload))
        return 0
    if args.cli or args.source:
        from obstaclesense.cli import run_source

        source = args.source or str(config.get("camera_index", 0))
        run_source(
            source,
            config,
            show=not args.no_display and _has_display(),
            output=args.output,
            max_frames=args.max_frames,
            loop=args.loop or bool(config.get("video_loop", False)),
        )
        return 0
    if not _has_display():
        print(
            "Não há tela nesta sessão. Use uma destas opções:\n"
            "  python main.py --demo --no-display\n"
            "  python main.py --cli --source video.mp4 --no-display\n"
            "  python main.py --api\n"
        )
        return 1
    from obstaclesense.gui import NavigationApp

    NavigationApp(config).run()
    return 0


def _apply_args(config: dict, args) -> None:
    if args.model:
        config["yolo_model_path"] = args.model
    if args.device:
        config["device"] = args.device
        config["enable_gpu"] = args.device == "cuda"
    if args.imgsz:
        config["imgsz"] = args.imgsz
    if args.no_audio:
        config["audio_enabled"] = False
    if args.depth:
        config["depth_enabled"] = True
    if args.host:
        config["api_host"] = args.host
    if args.port:
        config["api_port"] = args.port


def _run_api(config: dict) -> None:
    import uvicorn

    from obstaclesense.api import create_app

    app = create_app(config=config)
    uvicorn.run(app, host=str(config.get("api_host", "0.0.0.0")), port=int(config.get("api_port", 8000)))


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


if __name__ == "__main__":
    sys.exit(main())

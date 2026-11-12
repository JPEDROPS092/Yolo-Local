"""Voz offline em português do Brasil: pyttsx3, Piper ou espeak-ng."""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class SpeechGate:
    """Limita a fala ao intervalo do config e deixa um alerta alto interromper."""

    def __init__(self, interval_s: float):
        self.interval_s = max(0.0, float(interval_s))
        self.last_time: float | None = None
        self.last_level: str | None = None

    def allow(self, now: float, danger_level: str) -> bool:
        if self.last_time is None:
            return True
        if now - self.last_time >= self.interval_s:
            return True
        if danger_level == "high" and self.last_level != "high":
            return True
        return False

    def mark(self, now: float, danger_level: str) -> None:
        self.last_time = now
        self.last_level = danger_level


class Speaker:
    """Fila de fala fora do laço de visão, para não travar o quadro."""

    def __init__(
        self,
        engine: str = "auto",
        rate: int = 175,
        volume: float = 0.8,
        interval_s: float = 2.0,
        piper_model: str = "",
        enabled: bool = True,
    ):
        self.engine = _resolve_engine(engine, piper_model)
        self.rate = int(rate)
        self.volume = float(volume)
        self.piper_model = piper_model
        self.enabled = enabled and self.engine != "none"
        self.gate = SpeechGate(interval_s)
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=2)
        self._thread: threading.Thread | None = None
        self._pyttsx = None
        if self.enabled:
            self._thread = threading.Thread(target=self._loop, name="obstaclesense-tts", daemon=True)
            self._thread.start()
            logger.info("Voz pronta com motor '%s'.", self.engine)
        else:
            logger.warning("Voz desligada ou nenhum motor pt-BR encontrado.")

    @property
    def ready(self) -> bool:
        return self._thread is not None

    def say(self, text: str, danger_level: str = "low", now: float | None = None) -> bool:
        if not self.enabled or not text or self._thread is None:
            return False
        moment = time.monotonic() if now is None else now
        if not self.gate.allow(moment, danger_level):
            return False
        self.gate.mark(moment, danger_level)
        self._replace(text)
        return True

    def close(self) -> None:
        if self._thread and self._thread.is_alive():
            self._replace(None)
            self._thread.join(timeout=2.0)

    def _replace(self, item: str | None) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            pass

    def _loop(self) -> None:
        while True:
            text = self._queue.get()
            if text is None:
                return
            try:
                speak_text(
                    text,
                    engine=self.engine,
                    rate=self.rate,
                    volume=self.volume,
                    piper_model=self.piper_model,
                    pyttsx_holder=self,
                )
            except Exception:
                logger.exception("Falha ao falar a orientação.")


def speak_text(
    text: str,
    engine: str = "auto",
    rate: int = 175,
    volume: float = 0.8,
    piper_model: str = "",
    pyttsx_holder: Speaker | None = None,
) -> str:
    """Fala ou sintetiza. Devolve o motor que de fato rodou."""
    chosen = _resolve_engine(engine, piper_model)
    try:
        if chosen == "piper":
            _speak_piper(text, piper_model, rate)
        elif chosen == "pyttsx3":
            _speak_pyttsx3(text, rate, volume, pyttsx_holder)
        elif chosen == "espeak":
            _speak_espeak(text, rate, volume)
        else:
            logger.info("Orientação (sem áudio): %s", text)
        return chosen
    except Exception:
        if chosen != "espeak" and _espeak_bin():
            logger.exception("Motor %s falhou. Usando espeak-ng com voz pt-BR.", chosen)
            _speak_espeak(text, rate, volume)
            return "espeak"
        raise


def synthesize_to_file(
    text: str,
    destination: str | Path,
    rate: int = 175,
    engine: str = "auto",
    piper_model: str = "",
) -> Path:
    """Grava um wav pt-BR para demonstrar a voz sem alto-falante."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    chosen = _resolve_engine(engine, piper_model)
    if chosen == "piper":
        _piper_to_wav(text, piper_model, path)
        return path
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if espeak and chosen in {"espeak", "pyttsx3", "auto"}:
        subprocess.run(
            [espeak, "-v", "pt-br", "-s", str(int(rate)), "-w", str(path), text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return path
    if chosen == "pyttsx3":
        import pyttsx3

        tts = pyttsx3.init()
        tts.setProperty("rate", int(rate))
        _select_pt_voice(tts)
        tts.save_to_file(text, str(path))
        tts.runAndWait()
        return path
    raise RuntimeError("Nenhum sintetizador offline disponível para gravar áudio.")


def _resolve_engine(engine: str, piper_model: str) -> str:
    requested = (engine or "auto").lower()
    if requested == "none":
        return "none"
    piper_ready = bool(piper_model) and Path(piper_model).exists() and shutil.which("piper")
    if requested == "piper":
        return "piper" if piper_ready else "none"
    if requested == "pyttsx3":
        return "pyttsx3" if _pyttsx3_available() else "none"
    if requested == "espeak":
        return "espeak" if _espeak_bin() else "none"
    if piper_ready:
        return "piper"
    # espeak-ng com -v pt-br é o caminho estável no Linux. pyttsx3 continua
    # disponível em tts_engine: pyttsx3 e entra no auto se o espeak não existir.
    if _espeak_bin():
        return "espeak"
    if _pyttsx3_available():
        return "pyttsx3"
    return "none"


def _pyttsx3_available() -> bool:
    try:
        import pyttsx3  # noqa: F401
    except Exception:
        return False
    return True


def _espeak_bin() -> str | None:
    return shutil.which("espeak-ng") or shutil.which("espeak")


def _select_pt_voice(engine) -> None:
    voices = engine.getProperty("voices") or []
    best = None
    for voice in voices:
        blob = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')} {getattr(voice, 'languages', '')}".lower()
        if "pt-br" in blob or "brazil" in blob or "brazilian" in blob:
            best = voice.id
            break
        if best is None and ("portuguese" in blob or "portugu" in blob):
            best = voice.id
    if best:
        engine.setProperty("voice", best)
        logger.info("Voz pyttsx3 selecionada: %s", best)


def _speak_pyttsx3(text: str, rate: int, volume: float, holder: Speaker | None) -> None:
    import pyttsx3

    engine = holder._pyttsx if holder is not None else None
    if engine is None:
        engine = pyttsx3.init()
        engine.setProperty("rate", int(rate))
        engine.setProperty("volume", max(0.0, min(1.0, volume)))
        _select_pt_voice(engine)
        if holder is not None:
            holder._pyttsx = engine
    engine.say(text)
    engine.runAndWait()


def _speak_espeak(text: str, rate: int, volume: float) -> None:
    binary = _espeak_bin()
    if not binary:
        raise RuntimeError("espeak-ng não está instalado.")
    amplitude = int(max(0.0, min(1.0, volume)) * 200)
    subprocess.run(
        [binary, "-v", "pt-br", "-s", str(int(rate)), "-a", str(amplitude), text],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _speak_piper(text: str, model: str, rate: int) -> None:
    wav = Path("/tmp/obstaclesense-piper.wav")
    _piper_to_wav(text, model, wav, rate)
    for player in ("aplay", "paplay", "ffplay"):
        binary = shutil.which(player)
        if not binary:
            continue
        cmd = [binary, str(wav)] if player != "ffplay" else [binary, "-nodisp", "-autoexit", str(wav)]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return


def _piper_to_wav(text: str, model: str, destination: Path, rate: int = 175) -> None:
    # 175 palavras/min é um pouco mais rápido que o padrão do Piper (~1.0).
    length_scale = str(max(0.6, min(1.4, 175 / max(80, rate))))
    subprocess.run(
        ["piper", "--model", model, "--length_scale", length_scale, "--output_file", str(destination)],
        input=text.encode("utf-8"),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

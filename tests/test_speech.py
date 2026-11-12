"""Limite entre falas e síntese offline em pt-BR."""

import shutil
import wave
from pathlib import Path

from obstaclesense.speech import SpeechGate, Speaker, synthesize_to_file


def test_interval_blocks_and_high_danger_interrupts():
    gate = SpeechGate(2.0)
    assert gate.allow(0.0, "medium")
    gate.mark(0.0, "medium")
    assert gate.allow(1.0, "medium") is False
    assert gate.allow(1.0, "low") is False
    assert gate.allow(1.0, "high") is True
    gate.mark(1.0, "high")
    assert gate.allow(1.5, "high") is False
    assert gate.allow(3.0, "low") is True


def test_disabled_speaker_says_nothing():
    speaker = Speaker(engine="none", enabled=False)
    assert speaker.say("Existe uma pessoa à sua frente.", "high", now=0.0) is False
    speaker.close()


def test_espeak_writes_a_portuguese_wav(tmp_path: Path):
    if not (shutil.which("espeak-ng") or shutil.which("espeak")):
        return
    destination = tmp_path / "aviso.wav"
    text = "Existe uma pessoa aproximadamente dois metros à sua frente. Desvie para a direita."
    synthesize_to_file(text, destination, rate=175, engine="espeak")
    assert destination.stat().st_size > 1000
    with wave.open(str(destination)) as audio:
        assert audio.getnchannels() >= 1
        assert audio.getframerate() > 0
        assert audio.getnframes() > 1000

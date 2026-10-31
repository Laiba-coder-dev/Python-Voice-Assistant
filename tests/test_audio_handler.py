"""Unit tests for the audio helpers and VAD logic (no real hardware)."""

from __future__ import annotations

import numpy as np

from voice_assistant.audio_handler import (
    MicrophoneRecorder,
    decode_audio,
    pcm_to_wav_bytes,
    resample,
    to_float32,
    to_int16,
    wav_bytes_to_pcm,
)


def _tone(seconds: float, rate: int, freq: float = 220.0, amp: float = 0.5):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp * 32767).astype(np.int16)


def test_wav_roundtrip():
    pcm = _tone(0.5, 16000)
    wav = pcm_to_wav_bytes(pcm, 16000)
    decoded, rate = wav_bytes_to_pcm(wav)
    assert rate == 16000
    assert np.array_equal(decoded, pcm)


def test_decode_audio_accepts_wav_bytes():
    pcm = _tone(0.2, 16000)
    decoded, rate = decode_audio(pcm_to_wav_bytes(pcm, 16000))
    assert rate == 16000
    assert decoded.dtype == np.int16


def test_decode_audio_rejects_empty():
    from voice_assistant.audio_handler import AudioError

    try:
        decode_audio(b"")
    except AudioError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected AudioError")


def test_resample_changes_length():
    pcm = _tone(0.5, 16000)
    out = resample(pcm, 16000, 24000)
    assert abs(len(out) - 12000) <= 1
    assert out.dtype == np.int16


def test_float_int16_roundtrip():
    pcm = _tone(0.1, 16000)
    back = to_int16(to_float32(pcm))
    assert np.max(np.abs(back.astype(int) - pcm.astype(int))) <= 1


class _FakeStream:
    def __init__(self, blocks):
        self._blocks = iter(blocks)

    def read(self, _frames):
        block = next(self._blocks)
        return block.reshape(-1, 1), False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_recorder(**kwargs):
    defaults = dict(
        sample_rate=16000,
        block_duration=0.03,
        threshold_factor=2.0,
        min_threshold=0.01,
        silence_duration=0.09,
        min_speech_duration=0.06,
        max_record_seconds=5.0,
        calibration_seconds=0.06,
        pre_roll_seconds=0.03,
    )
    defaults.update(kwargs)
    return MicrophoneRecorder(**defaults)


def test_record_utterance_detects_speech_and_silence():
    rec = _make_recorder()
    block = rec.block_size
    silence = np.zeros(block, dtype=np.int16)
    loud = np.full(block, 20000, dtype=np.int16)
    # calibration blocks + pre-roll silences, then speech, then trailing silence.
    blocks = [silence] * 4 + [loud] * 4 + [silence] * 4
    rec._open_stream = lambda: _FakeStream(blocks)  # type: ignore[assignment]

    audio = rec.record_utterance()
    assert audio is not None
    assert audio.dtype == np.int16
    assert len(audio) / 16000 >= 0.06


def test_record_utterance_returns_none_for_silence_timeout():
    rec = _make_recorder()
    block = rec.block_size
    silence = np.zeros(block, dtype=np.int16)
    blocks = [silence] * 200
    rec._open_stream = lambda: _FakeStream(blocks)  # type: ignore[assignment]

    audio = rec.record_utterance(timeout=0.001)
    assert audio is None


def test_record_utterance_respects_stop_event():
    import threading

    rec = _make_recorder()
    block = rec.block_size
    silence = np.zeros(block, dtype=np.int16)
    blocks = [silence] * 200
    rec._open_stream = lambda: _FakeStream(blocks)  # type: ignore[assignment]

    stop = threading.Event()
    stop.set()
    assert rec.record_utterance(stop) is None

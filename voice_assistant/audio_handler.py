"""Microphone capture, voice-activity detection and PCM/WAV playback.

The module wraps *sounddevice* so the rest of the application never has to deal
with PortAudio directly.  Everything works with raw 16-bit mono PCM captured at
``sample_rate`` (16 kHz by default, matching Scribe's preferred input format).

Two pieces are exposed:

* :class:`MicrophoneRecorder` - blocks until a full spoken utterance has been
  captured, using a simple, well-behaved energy/RMS voice-activity detector.
* :class:`PCMPlayer` - plays either a complete buffer or a *stream* of PCM
  chunks (as produced by ElevenLabs' streaming text-to-speech) and can be
  stopped at any time.
"""

from __future__ import annotations

import io
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np

from .logging_utils import get_logger

try:  # pragma: no cover - import guard for headless machines
    import sounddevice as sd
except Exception as exc:  # noqa: BLE001 - PortAudio may be missing
    sd = None  # type: ignore[assignment]
    _SOUNDDEVICE_ERROR: Exception | None = exc
else:  # pragma: no cover - trivial
    _SOUNDDEVICE_ERROR = None

log = get_logger(__name__)

#: sounddevice/PortAudio error type, if available.
PortAudioError = getattr(sd, "PortAudioError", RuntimeError) if sd else RuntimeError


class AudioError(RuntimeError):
    """Generic audio-layer error."""


class AudioDeviceError(AudioError):
    """Raised when a microphone/speaker cannot be opened."""


class NoMicrophoneError(AudioDeviceError):
    """Raised when no usable input device is present."""


def _require_sounddevice() -> None:
    if sd is None:
        raise AudioError(
            "The 'sounddevice' audio backend is unavailable "
            f"({_SOUNDDEVICE_ERROR}). Install sounddevice and make sure a "
            "working PortAudio build is present."
        )


@dataclass(slots=True)
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float | None


def list_devices() -> list[DeviceInfo]:
    """Return every audio device known to PortAudio."""
    _require_sounddevice()
    devices = sd.query_devices()
    result: list[DeviceInfo] = []
    for index, dev in enumerate(devices):
        result.append(
            DeviceInfo(
                index=index,
                name=str(dev["name"]),
                max_input_channels=int(dev["max_input_channels"]),
                max_output_channels=int(dev["max_output_channels"]),
                default_samplerate=float(dev["default_samplerate"])
                if dev.get("default_samplerate")
                else None,
            )
        )
    return result


def list_input_devices() -> list[DeviceInfo]:
    return [d for d in list_devices() if d.max_input_channels > 0]


def list_output_devices() -> list[DeviceInfo]:
    return [d for d in list_devices() if d.max_output_channels > 0]


def default_input_device() -> int | None:
    _require_sounddevice()
    try:
        idx = sd.default.device[0]
    except Exception:  # pragma: no cover - defensive
        return None
    return int(idx) if idx is not None and idx >= 0 else None


def default_output_device() -> int | None:
    _require_sounddevice()
    try:
        idx = sd.default.device[1]
    except Exception:  # pragma: no cover - defensive
        return None
    return int(idx) if idx is not None and idx >= 0 else None


# --------------------------------------------------------------------------- #
# PCM helpers
# --------------------------------------------------------------------------- #
def rms(samples: np.ndarray) -> float:
    """Root-mean-square of a float array already scaled to [-1, 1]."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def to_float32(pcm: np.ndarray) -> np.ndarray:
    """Convert int16 PCM to float32 in the range [-1, 1]."""
    return pcm.astype(np.float32) / 32768.0


def to_int16(samples: np.ndarray) -> np.ndarray:
    """Convert float samples in [-1, 1] to int16 PCM."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Small linear-interpolation resampler (adequate for speech)."""
    if src_rate == dst_rate or pcm.size == 0:
        return pcm
    duration = pcm.shape[0] / float(src_rate)
    target_len = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, 1.0, num=pcm.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    out = np.interp(dst_x, src_x, pcm.astype(np.float32))
    return np.asarray(np.round(out), dtype=np.int16)


def pcm_to_wav_bytes(pcm: np.ndarray, samplerate: int) -> bytes:
    """Encode mono int16 PCM into an in-memory WAV file."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(samplerate))
        wav.writeframes(np.ascontiguousarray(pcm, dtype=np.int16).tobytes())
    return buffer.getvalue()


def wav_bytes_to_pcm(data: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV byte string into mono int16 PCM."""
    with wave.open(io.BytesIO(data), "rb") as wav:
        channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        samplerate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sampwidth == 2:
        pcm = np.frombuffer(frames, dtype=np.int16)
    elif sampwidth == 1:
        # 8-bit WAV is unsigned.
        pcm = (np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128) * 256
    elif sampwidth == 4:
        pcm = (np.frombuffer(frames, dtype=np.int32) >> 16).astype(np.int16)
    else:
        raise AudioError(f"Unsupported WAV sample width: {sampwidth * 8}-bit")

    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return np.ascontiguousarray(pcm, dtype=np.int16), samplerate


def decode_audio(data: bytes) -> tuple[np.ndarray, int]:
    """Best-effort decode of uploaded audio into mono int16 PCM.

    WAV is handled natively; other formats use *soundfile* when installed.
    """
    if not data:
        raise AudioError("No audio data was provided.")

    try:
        return wav_bytes_to_pcm(data)
    except Exception:
        pass

    try:  # optional dependency
        import soundfile as sf  # type: ignore

        samples, samplerate = sf.read(io.BytesIO(data), dtype="int16", always_2d=True)
        mono = samples.mean(axis=1).astype(np.int16)
        return np.ascontiguousarray(mono, dtype=np.int16), int(samplerate)
    except Exception as exc:  # noqa: BLE001
        raise AudioError(
            "Unsupported audio format. Provide a WAV file, or install "
            "'soundfile' for additional formats."
        ) from exc


# --------------------------------------------------------------------------- #
# Playback
# --------------------------------------------------------------------------- #
class PCMPlayer:
    """Plays mono int16 PCM through the default (or configured) output device.

    The player is safe to call from multiple threads: starting new playback
    stops anything already playing, so two responses are never heard at once.
    """

    def __init__(
        self,
        sample_rate: int = 24_000,
        device: int | None = None,
        channels: int = 1,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.device = device
        self.channels = channels
        self._stop = threading.Event()
        self._stream = None
        self._play_lock = threading.Lock()

    # -- state ---------------------------------------------------------- #
    @property
    def is_playing(self) -> bool:
        return self._stream is not None

    def stop(self) -> None:
        """Request playback to stop and abort the active stream."""
        self._stop.set()
        stream = self._stream
        if stream is not None:
            try:
                stream.abort()
            except Exception:  # pragma: no cover - best effort
                pass

    # -- playback ------------------------------------------------------- #
    def play(self, pcm: bytes | np.ndarray, sample_rate: int | None = None) -> None:
        """Play a complete PCM buffer (blocking)."""
        data = self._as_bytes(pcm)
        if not data:
            return
        requested = int(sample_rate or self.sample_rate)
        self._begin()
        with self._play_lock:
            try:
                rate = self._negotiate_rate(requested)
                if rate != requested:
                    pcm_arr = np.frombuffer(data, dtype=np.int16)
                    data = resample(pcm_arr, requested, rate).tobytes()
                with self._open_stream(rate) as stream:
                    self._stream = stream
                    stream.write(data)
            except AudioError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise AudioDeviceError(f"Audio playback failed: {exc}") from exc
            finally:
                self._stream = None

    def play_stream(
        self, chunks: Iterable[bytes], sample_rate: int | None = None
    ) -> None:
        """Play an iterator of PCM byte chunks, starting as soon as data arrives."""
        requested = int(sample_rate or self.sample_rate)
        self._begin()
        with self._play_lock:
            try:
                rate = self._negotiate_rate(requested)
                source: Iterable[bytes] = (
                    chunks if rate == requested else _resample_chunks(chunks, requested, rate)
                )
                with self._open_stream(rate) as stream:
                    self._stream = stream
                    for chunk in source:
                        if self._stop.is_set():
                            break
                        if chunk:
                            stream.write(chunk)
            except AudioError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise AudioDeviceError(f"Audio playback failed: {exc}") from exc
            finally:
                self._stream = None

    # -- internals ------------------------------------------------------ #
    def _begin(self) -> None:
        """Stop current playback and mark a new session as active."""
        self.stop()
        self._stop.clear()

    @staticmethod
    def _as_bytes(pcm: bytes | np.ndarray) -> bytes:
        if isinstance(pcm, (bytes, bytearray)):
            return bytes(pcm)
        return np.ascontiguousarray(pcm, dtype=np.int16).tobytes()

    def _open_stream(self, sample_rate: int):  # noqa: ANN001
        _require_sounddevice()
        return sd.RawOutputStream(
            samplerate=int(sample_rate),
            channels=self.channels,
            dtype="int16",
            device=self.device,
        )

    def _negotiate_rate(self, requested: int) -> int:
        """Return a sample rate the output device actually supports."""
        candidates = [requested]
        default = self._device_default_rate()
        if default:
            candidates.append(default)
        candidates.extend([48_000, 44_100, 24_000, 22_050, 16_000])
        seen: set[int] = set()
        for rate in candidates:
            if rate in seen:
                continue
            seen.add(rate)
            if self._device_supports(rate):
                return rate
        return requested

    def _device_supports(self, rate: int) -> bool:
        try:
            sd.check_output_settings(
                device=self.device,
                channels=self.channels,
                dtype="int16",
                samplerate=int(rate),
            )
            return True
        except Exception:  # noqa: BLE001
            return False

    def _device_default_rate(self) -> int | None:
        try:
            info = sd.query_devices(self.device)
            rate = info.get("default_samplerate")
            return int(rate) if rate else None
        except Exception:  # pragma: no cover - defensive
            return None


def _resample_chunks(
    chunks: Iterable[bytes], src_rate: int, dst_rate: int
) -> Iterator[bytes]:
    """Resample a stream of PCM chunks, carrying one sample for continuity."""
    carry = np.empty(0, dtype=np.int16)
    for chunk in chunks:
        if not chunk:
            continue
        block = np.frombuffer(chunk, dtype=np.int16)
        combined = np.concatenate([carry, block]) if carry.size else block
        out = resample(combined, src_rate, dst_rate)
        # Carry the final sample into the next chunk for a smooth seam.
        carry = combined[-1:]
        yield out.tobytes()


# --------------------------------------------------------------------------- #
# Microphone capture with energy-based VAD
# --------------------------------------------------------------------------- #
class MicrophoneRecorder:
    """Captures spoken utterances from the microphone.

    :meth:`record_utterance` blocks until the speaker starts (speech onset),
    then stops after a configurable trailing silence, returning the complete
    utterance as mono int16 PCM.  ``None`` means "no speech detected".
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        device: int | None = None,
        *,
        block_duration: float = 0.03,
        threshold_factor: float = 2.5,
        min_threshold: float = 0.01,
        silence_duration: float = 0.8,
        min_speech_duration: float = 0.25,
        max_record_seconds: float = 20.0,
        calibration_seconds: float = 0.4,
        pre_roll_seconds: float = 0.3,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.device = device
        self.block_size = max(1, int(self.sample_rate * block_duration))
        self.threshold_factor = threshold_factor
        self.min_threshold = min_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.max_record_seconds = max_record_seconds
        self.calibration_blocks = max(
            1, int(round(calibration_seconds / block_duration))
        )
        self.pre_roll_blocks = max(1, int(round(pre_roll_seconds / block_duration)))

    def _open_stream(self):  # noqa: ANN001
        _require_sounddevice()
        try:
            return sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.block_size,
                device=self.device,
            )
        except PortAudioError as exc:
            raise AudioDeviceError(
                f"Could not open the microphone ({exc}). Check that a microphone "
                "is connected and that microphone access is allowed."
            ) from exc

    def measure_noise_floor(self, stream, blocks: int | None = None) -> float:
        """Return the ambient noise RMS (normalised to [-1, 1])."""
        blocks = blocks or self.calibration_blocks
        readings: list[float] = []
        for _ in range(blocks):
            block, _overflow = stream.read(self.block_size)
            readings.append(rms(to_float32(block[:, 0])))
        return float(np.median(readings)) if readings else 0.0

    def record_utterance(
        self,
        stop_event: Optional[threading.Event] = None,
        *,
        timeout: float | None = None,
    ) -> np.ndarray | None:
        """Capture one utterance, or return ``None``.

        ``timeout`` (seconds) bounds how long to wait for speech to *start*;
        ``None`` waits indefinitely (until ``stop_event`` is set).
        """
        _require_sounddevice()
        block_duration = self.block_size / float(self.sample_rate)
        silence_needed = max(1, int(round(self.silence_duration / block_duration)))
        onset_needed = 2

        with self._open_stream() as stream:
            noise_floor = self.measure_noise_floor(stream)
            threshold = max(noise_floor * self.threshold_factor, self.min_threshold)
            log.debug(
                "Mic ready (rate=%s Hz, noise=%.4f, threshold=%.4f)",
                self.sample_rate,
                noise_floor,
                threshold,
            )

            pre_roll: deque[np.ndarray] = deque(maxlen=self.pre_roll_blocks)
            collected: list[np.ndarray] = []
            triggered = False
            above_count = 0
            silence_blocks = 0
            speech_blocks = 0
            started_waiting = time.monotonic()

            while True:
                if stop_event is not None and stop_event.is_set():
                    return None
                block, _overflow = stream.read(self.block_size)
                mono = np.ascontiguousarray(block[:, 0], dtype=np.int16)
                level = rms(to_float32(mono))

                if not triggered:
                    pre_roll.append(mono)
                    if level > threshold:
                        above_count += 1
                        if above_count >= onset_needed:
                            triggered = True
                            collected.extend(pre_roll)
                            pre_roll.clear()
                            speech_blocks = above_count
                            silence_blocks = 0
                    else:
                        above_count = 0
                    if timeout and (time.monotonic() - started_waiting) > timeout:
                        return None
                    continue

                # -- already speaking: collect until trailing silence --
                collected.append(mono)
                if level > threshold:
                    silence_blocks = 0
                    speech_blocks += 1
                else:
                    silence_blocks += 1

                if silence_blocks >= silence_needed:
                    break
                if (speech_blocks + silence_blocks) * block_duration >= (
                    self.max_record_seconds
                ):
                    log.debug("Max utterance length reached; stopping capture.")
                    break

            if not triggered or not collected:
                return None

            audio = np.concatenate(collected).astype(np.int16)
            duration = audio.shape[0] / float(self.sample_rate)
            if duration < self.min_speech_duration:
                return None
            return audio

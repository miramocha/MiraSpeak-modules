"""Piper through piper-tts, in-process on the CPU. One ONNX file per speaker.

MiraSpeak creates the adapter and calls it on its own synthesis threads, not the module thread.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
from miraspeak.modules.engine_api import (
    NORMAL_ONLY_STYLES,
    PACE_SPEEDS,
    SPEED_PACES,
    Adapter,
    AdapterError,
    AudioClip,
    Capabilities,
    ProgressCallback,
    SpeechRequest,
    SynthesisResult,
    concat,
    download_to,
    engine_data_dir,
    g2p_lock,
    pcm16_to_float32,
    spans_from_texts,
    starts_of,
)

ENGINE_ID = "piper"
MISSING = "Piper voice is not downloaded. Open the Piper module."
RUNTIME = "The Piper packages are not installed. Remove the Piper module and install it again."
ENGLISH_ONLY = "Piper currently supports English voices only."
DEFAULT_VOICE_ID = "en_US-lessac-medium"
HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
INDEX_URL = f"{HF_BASE}/voices.json"
DOWNLOAD_TIMEOUT_S = 30.0  # per network read, not the whole voice file
MAX_CHARS = 300
LOCAL_CACHE_MB = 250


@dataclass(frozen=True)
class PiperVoiceInfo:
    """One speaker model from the public Piper voice index."""

    voice_id: str
    name: str
    quality: str
    language_english: str
    region: str
    size_bytes: int
    onnx_rel: str
    config_rel: str

    def label(self) -> str:
        """Row text: ``Lessac · English (US) · medium · 63 MB``."""
        name = self.name.replace("_", " ").replace("-", " ").title() or self.voice_id
        language = self.language_english or "Voice"
        place = f"{language} ({self.region})" if self.region else language
        quality = self.quality or "medium"
        if self.size_bytes > 0:
            return f"{name} · {place} · {quality} · {self.size_bytes >> 20} MB"
        return f"{name} · {place} · {quality}"


def piper_dir() -> Path:
    """Same folder the built-in Piper used, so downloaded voices carry over."""
    return engine_data_dir(ENGINE_ID)


def voices_dir() -> Path:
    return piper_dir() / "voices"


def index_path() -> Path:
    return piper_dir() / "voices.json"


def voice_paths(voice_id: str) -> tuple[Path, Path]:
    folder = voices_dir()
    return folder / f"{voice_id}.onnx", folder / f"{voice_id}.onnx.json"


def english_voice_id(voice_id: str) -> bool:
    """True for an English Piper id (``en_US-lessac-medium``). Other languages are out of scope."""
    return voice_id.startswith("en_")


def english_voices(raw: object) -> dict[str, str]:
    """Speak aliases whose Piper ids are English. Other languages are not offered."""
    if not isinstance(raw, dict):
        return {}
    return {
        str(alias): native
        for alias, native in raw.items()
        if str(alias) and isinstance(native, str) and english_voice_id(native)
    }


def voice_installed(voice_id: str) -> bool:
    """True when both the ONNX model and its JSON config are on disk."""
    onnx, config = voice_paths(voice_id)
    return onnx.is_file() and config.is_file()


def length_scale_for_pace(pace: str) -> float:
    """Piper length scale for a MiraSpeak pace. Higher is slower."""
    speed = PACE_SPEEDS.get(pace, PACE_SPEEDS["normal"])
    if speed <= 0:
        return 1.0
    return 1.0 / speed


def parse_voice_index(data: object) -> tuple[PiperVoiceInfo, ...]:
    """Voices from a Piper ``voices.json`` object. Entries without both files are skipped."""
    if not isinstance(data, dict):
        raise AdapterError("Piper voice index is not an object.")
    voices = [
        info
        for key, raw in data.items()
        if (info := _parse_voice(str(key), raw)) and english_voice_id(info.voice_id)
    ]
    voices.sort(key=lambda item: (item.language_english.lower(), item.name.lower(), item.quality))
    return tuple(voices)


def read_voice_index(path: Path | None = None) -> tuple[PiperVoiceInfo, ...]:
    """Load the saved index. Missing → empty. Unreadable or bad JSON raises AdapterError."""
    file = index_path() if path is None else path
    if not file.is_file():
        return ()
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        raise AdapterError(f"Piper voice index could not be read: {e}") from e
    return parse_voice_index(data)


def _index_readable(path: Path) -> bool:
    """True when *path* is present and ``read_voice_index`` can load it."""
    if not path.is_file():
        return False
    try:
        read_voice_index(path)
    except AdapterError:
        return False
    return True


def _drop_unreadable_index(path: Path) -> None:
    """Delete an index that cannot be read. A missing or valid file stays."""
    if _index_readable(path):
        return
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def ensure_voice_index(
    *,
    force: bool = False,
    on_progress: ProgressCallback | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """Download ``voices.json`` into the Piper folder. An unreadable file is replaced."""
    piper_dir().mkdir(parents=True, exist_ok=True)
    dest = index_path()
    if not force and _index_readable(dest):
        if on_progress is not None:
            on_progress(dest.name, 100, dest.stat().st_size)
        return dest
    try:
        download_to(
            INDEX_URL,
            dest,
            timeout_s=DOWNLOAD_TIMEOUT_S,
            on_progress=on_progress,
            should_stop=should_stop,
        )
    except Exception:
        _drop_unreadable_index(dest)
        raise
    try:
        read_voice_index(dest)
    except AdapterError:
        _drop_unreadable_index(dest)
        raise
    return dest


def download_voice(
    info: PiperVoiceInfo,
    *,
    on_progress: ProgressCallback | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Download one English voice's ONNX and JSON. A failed second file deletes the pair."""
    if not english_voice_id(info.voice_id):
        raise AdapterError(ENGLISH_ONLY)
    voices_dir().mkdir(parents=True, exist_ok=True)
    onnx, config = voice_paths(info.voice_id)
    try:
        _download_file(f"{HF_BASE}/{info.onnx_rel}", onnx, on_progress, should_stop)
        _download_file(f"{HF_BASE}/{info.config_rel}", config, on_progress, should_stop)
    except Exception:
        if not voice_installed(info.voice_id):
            onnx.unlink(missing_ok=True)
            config.unlink(missing_ok=True)
        raise


def remove_voice_files(voice_id: str) -> None:
    """Delete one voice's ONNX and JSON. Missing files are fine.

    A file MiraSpeak still has open raises OSError after the others are removed.
    """
    failed: OSError | None = None
    for path in voice_paths(voice_id):
        try:
            path.unlink(missing_ok=True)
        except OSError as e:
            failed = e
    if failed is not None:
        raise failed


def _download_file(
    url: str,
    dest: Path,
    on_progress: ProgressCallback | None,
    should_stop: Callable[[], bool] | None,
) -> None:
    if dest.is_file():
        if on_progress is not None:
            on_progress(dest.name, 100, dest.stat().st_size)
        return
    download_to(
        url,
        dest,
        timeout_s=DOWNLOAD_TIMEOUT_S,
        on_progress=on_progress,
        should_stop=should_stop,
    )


def _parse_voice(key: str, raw: object) -> PiperVoiceInfo | None:
    if not isinstance(raw, dict):
        return None
    onnx_rel, config_rel, size = _file_paths(raw.get("files"))
    if not onnx_rel or not config_rel:
        return None
    language, region = _language_fields(raw.get("language"))
    name = raw.get("name")
    quality = raw.get("quality")
    return PiperVoiceInfo(
        voice_id=key,
        name=name.strip() if isinstance(name, str) and name.strip() else key,
        quality=quality.strip() if isinstance(quality, str) else "",
        language_english=language,
        region=region,
        size_bytes=size,
        onnx_rel=onnx_rel,
        config_rel=config_rel,
    )


def _language_fields(raw: object) -> tuple[str, str]:
    if not isinstance(raw, dict):
        return "", ""
    english = raw.get("name_english")
    region = raw.get("region")
    code = raw.get("code")
    language = english.strip() if isinstance(english, str) else ""
    place = region.strip() if isinstance(region, str) else ""
    if not place and isinstance(code, str) and "_" in code:
        place = code.split("_", 1)[1]
    return language, place


def _file_paths(files: object) -> tuple[str, str, int]:
    if not isinstance(files, dict):
        return "", "", 0
    onnx = config = ""
    size = 0
    for path, meta in files.items():
        name = str(path)
        if name.endswith(".onnx.json"):
            config = name
        elif name.endswith(".onnx"):
            onnx = name
            size = _size_bytes(meta)
    return onnx, config, size


def _size_bytes(meta: object) -> int:
    if not isinstance(meta, dict):
        return 0
    raw = meta.get("size_bytes", 0)
    return raw if isinstance(raw, int) and raw > 0 else 0


def runtime_installed() -> bool:
    """True when piper-tts can be imported. Does not load the native library."""
    return find_spec("piper") is not None


class PiperAdapter(Adapter):
    id = ENGINE_ID
    label = "Piper"
    local = True
    # Same as the built-in Piper, so lines it cached still play from the cache.
    cache_version = "piper1"
    default_config = {
        "voices": {},
        "synth_ahead": 1,
        "cache": False,
        "cache_read": True,
        "cache_mb": LOCAL_CACHE_MB,
    }

    def __init__(self, config: Mapping[str, Any], secrets: Any = None) -> None:
        super().__init__(config, secrets)
        self._loaded: dict[str, Any] = {}

    @classmethod
    def unavailable_reason(cls, config: Mapping[str, Any], secrets: Any) -> str | None:
        if not runtime_installed():
            return RUNTIME
        natives = english_voices(config.get("voices")).values()
        # One missing voice is that voice's problem. Hide Piper only when none are installed.
        if not any(voice_installed(native) for native in natives):
            return MISSING
        return None

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(styles=NORMAL_ONLY_STYLES, paces=SPEED_PACES)

    def close(self) -> None:
        self._loaded.clear()

    def synthesize(self, request: SpeechRequest) -> SynthesisResult:
        voice_id = self.resolve_voice(request.voice)
        if not isinstance(voice_id, str) or not english_voice_id(voice_id):
            raise AdapterError(ENGLISH_ONLY)
        if not voice_installed(voice_id):
            raise AdapterError(MISSING)
        _styles, dropped_style = self.resolve_styles(request.styles)
        pace, dropped_pace = self.resolve_pace(request.pace)
        pieces = self.chunk_text(request.text, MAX_CHARS) or [request.text]
        clips = [self._speak(piece, voice_id, pace) for piece in pieces]
        clip = concat(clips)
        return SynthesisResult(
            clip, dropped_style | dropped_pace, spans_from_texts(pieces), starts_of(clips)
        )

    def _speak(self, text: str, voice_id: str, pace: str) -> AudioClip:
        model = self._model(voice_id)
        samples, rate = _synthesize_locked(model, text, length_scale_for_pace(pace))
        if rate <= 0 or samples.size == 0:
            raise AdapterError("Piper produced no audio.")
        return AudioClip(samples, rate)

    def _model(self, voice_id: str) -> Any:
        cached = self._loaded.get(voice_id)
        if cached is not None:
            return cached
        try:
            # Deferred: importing piper loads onnxruntime, only when Piper speaks.
            from piper import PiperVoice
        except ImportError as e:
            raise AdapterError(RUNTIME) from e
        onnx, config = voice_paths(voice_id)
        try:
            with g2p_lock:
                loaded = PiperVoice.load(onnx, config_path=config, use_cuda=False)
        except Exception as e:
            raise AdapterError(f"Piper failed to load: {e}") from e
        self._loaded[voice_id] = loaded
        return loaded


def _synthesize_locked(model: Any, text: str, length_scale: float) -> tuple[np.ndarray, int]:
    """Run Piper under the shared espeak lock. Returns float32 samples and the sample rate."""
    from piper.config import SynthesisConfig

    syn = SynthesisConfig(length_scale=length_scale)
    if getattr(model.config, "num_speakers", 1) > 1:
        syn.speaker_id = 0
    chunks: list[np.ndarray] = []
    rate = 0
    with g2p_lock:
        try:
            produced = model.synthesize(text, syn_config=syn)
            for chunk in produced:
                pcm = chunk.audio_int16_bytes
                if not pcm:
                    continue
                if rate == 0:
                    rate = int(chunk.sample_rate)
                chunks.append(pcm16_to_float32(pcm))
        except ModuleNotFoundError as e:
            missing = f"the {e.name} package" if e.name else "an extra phonemizer package"
            raise AdapterError(f"This Piper voice needs {missing}.") from e
        except (ValueError, RuntimeError) as e:
            raise AdapterError(f"Piper could not read that line: {e}") from e
    if not chunks:
        return np.zeros(0, dtype=np.float32), rate
    return np.concatenate(chunks).astype(np.float32), rate

"""Piper voices for MiraSpeak: the engine, plus a page that downloads and removes voices."""

from __future__ import annotations

import threading
from collections.abc import Callable

from miraspeak.modules.api import Module, ModuleContext, ModuleError, ui
from miraspeak.modules.engine_api import AdapterError, DownloadCanceledError, voice_alias

from .engine import (
    DEFAULT_VOICE_ID,
    ENGINE_ID,
    PiperAdapter,
    PiperVoiceInfo,
    download_voice,
    english_voices,
    ensure_voice_index,
    read_voice_index,
    remove_voice_files,
    voice_installed,
)

REMOVE = "Remove"
LICENSE_NOTE = (
    "CPU only. Each voice keeps the license published with its files. "
    "Piper itself is GPL-3.0-or-later."
)
NO_VOICES = ui.Empty(
    "No Piper voices yet",
    "Pick a voice above and press Download.",
    action="download",
    action_label="Download",
    art="list",
)


class PiperModule(Module):
    def __init__(self) -> None:
        self.ctx: ModuleContext | None = None
        self._index: tuple[PiperVoiceInfo, ...] = ()
        self._installed: list[tuple[str, str]] = []  # (alias, voice id), as the rows show them
        self._busy = False
        self._stop = threading.Event()

    def engines(self):
        return [PiperAdapter]

    def page(self) -> ui.Page:
        return ui.Page(
            sections=(
                ui.Section(
                    "Voices",
                    (
                        ui.Choice("voice", "Voice", default=DEFAULT_VOICE_ID),
                        ui.Button("download", "Download"),
                        ui.Button("refresh", "Refresh list"),
                        ui.Button("cancel", "Cancel", enabled=False),
                        ui.Status("status"),
                        ui.Rows(
                            "installed",
                            "Installed",
                            row_actions=(REMOVE,),
                            when_empty=NO_VOICES,
                        ),
                        ui.Status("license", LICENSE_NOTE),
                    ),
                ),
                ui.Section(
                    "Speech",
                    (
                        ui.Toggle("cache", "Cache lines so Replay skips generating them"),
                        ui.Number("synth_ahead", "Generate ahead", default=1, minimum=1, maximum=5),
                    ),
                ),
            )
        )

    def on_start(self, ctx: ModuleContext) -> None:
        self.ctx = ctx
        try:
            self._index = read_voice_index()
        except AdapterError as e:
            ctx.log.warning("%s", e)
        self._show_choices()
        if not self._index:
            self._refresh_index(force=False)

    def on_stop(self) -> None:
        self._stop.set()

    def on_voices_changed(self) -> None:
        """MiraSpeak added the engine, or its voices changed: redraw from the saved settings."""
        self._sync_from_settings()

    def on_action(self, action: str, row: int | None = None) -> None:
        if action == "download":
            self._download(str(self._ctx.values.get("voice", "")))
        elif action == "refresh":
            self._refresh_index(force=True)
        elif action == "cancel":
            self._stop.set()
        elif action == REMOVE and row is not None and 0 <= row < len(self._installed):
            self._remove(self._installed[row][0])

    def on_value(self, key: str, value: object) -> None:
        try:
            if key == "cache":
                self._ctx.set_engine_config(ENGINE_ID, cache=bool(value))
            elif key == "synth_ahead" and isinstance(value, int | float):
                self._ctx.set_engine_config(ENGINE_ID, synth_ahead=int(value))
        except ModuleError as e:
            self._status(str(e), ui.TONE_WARNING)

    # settings

    @property
    def _ctx(self) -> ModuleContext:
        assert self.ctx is not None
        return self.ctx

    def _voices(self) -> dict[str, str] | None:
        """Installed alias → Piper voice id, or None before MiraSpeak has added the engine."""
        try:
            table = self._ctx.engine_config(ENGINE_ID)
        except ModuleError:
            return None
        return english_voices(table.get("voices"))

    def _sync_from_settings(self) -> None:
        try:
            table = self._ctx.engine_config(ENGINE_ID)
        except ModuleError:
            return
        if bool(table.get("cache", False)) != bool(self._ctx.values.get("cache", False)):
            self._ctx.ui.set("cache", bool(table.get("cache", False)))
        ahead = table.get("synth_ahead")
        if isinstance(ahead, int) and ahead != self._ctx.values.get("synth_ahead"):
            self._ctx.ui.set("synth_ahead", ahead)
        self._show_installed(english_voices(table.get("voices")))
        self._show_choices()

    def _save_voices(self, voices: dict[str, str]) -> None:
        self._ctx.set_engine_config(ENGINE_ID, voices=voices)
        self._show_installed(voices)
        self._show_choices()

    # page

    def _status(self, text: str, tone: str = ui.TONE_NEUTRAL) -> None:
        self._ctx.ui.set_status("status", text, tone)

    def _show_installed(self, voices: dict[str, str]) -> None:
        by_id = {info.voice_id: info for info in self._index}
        rows: list[tuple[str, str, str]] = []
        for alias, voice_id in voices.items():
            info = by_id.get(voice_id)
            label = info.label() if info is not None else alias.replace("-", " ").title()
            if not voice_installed(voice_id):
                label = f"{label} · missing files"
            rows.append((label, alias, voice_id))
        rows.sort(key=lambda item: item[0].lower())
        self._installed = [(alias, voice_id) for _label, alias, voice_id in rows]
        self._ctx.ui.set_rows("installed", [label for label, _alias, _id in rows])

    def _show_choices(self) -> None:
        """Voices to download: not installed, or installed with a missing file."""
        voices = self._voices() or {}
        present = {voice_id for voice_id in voices.values() if voice_installed(voice_id)}
        listed = {info.voice_id for info in self._index}
        options = [(i.voice_id, i.label()) for i in self._index if i.voice_id not in present]
        for voice_id in voices.values():
            if voice_id not in present and voice_id not in listed:
                options.append((voice_id, voice_id))
        self._ctx.ui.set_options("voice", options)
        current = str(self._ctx.values.get("voice", ""))
        if options and current not in {voice_id for voice_id, _label in options}:
            self._ctx.ui.set("voice", options[0][0])

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for key in ("download", "refresh", "voice"):
            self._ctx.ui.set_enabled(key, not busy)
        self._ctx.ui.set_enabled("cancel", busy)

    # work off the module thread

    def _run(self, message: str, work: Callable[[], object], done: Callable[[object], None]):
        """Run *work* on a worker thread, then *done* back on the module thread."""
        if self._busy:
            return
        self._stop.clear()
        self._set_busy(True)
        self._status(message)
        ctx = self._ctx

        def thread() -> None:
            try:
                result: object = work()
            except DownloadCanceledError:
                ctx.call_later(0, lambda: self._finish("Download canceled."))
                return
            except Exception as e:
                ctx.log.warning("Piper download failed", exc_info=True)
                text = f"Download failed: {e}"
                ctx.call_later(0, lambda: self._finish(text, ui.TONE_WARNING))
                return
            ctx.call_later(0, lambda: done(result))

        threading.Thread(target=thread, name="piper-download", daemon=True).start()

    def _finish(self, text: str, tone: str = ui.TONE_NEUTRAL) -> None:
        self._set_busy(False)
        self._status(text, tone)

    def _progress(self):
        ctx = self._ctx
        last: dict[str, int] = {}

        def on_progress(name: str, pct: int, total: int) -> None:
            if last.get(name, -2) == pct:
                return
            last[name] = pct
            if pct < 0:
                text = f"{name}: downloading…"
            elif total:
                text = f"{name}: {pct}% of {total >> 20} MB"
            else:
                text = f"{name}: {pct}%"
            ctx.call_later(0, lambda: self._status(text))

        return on_progress

    def _refresh_index(self, *, force: bool) -> None:
        def work() -> tuple[PiperVoiceInfo, ...]:
            ensure_voice_index(
                force=force, on_progress=self._progress(), should_stop=self._stop.is_set
            )
            return read_voice_index()

        def done(result: object) -> None:
            if isinstance(result, tuple):
                self._index = result
            self._show_choices()
            self._sync_from_settings()
            self._finish(f"{len(self._index)} voices in the list.")

        self._run("Refreshing voices…", work, done)

    def _download(self, voice_id: str) -> None:
        if not voice_id:
            self._status("Pick a voice first.", ui.TONE_WARNING)
            return
        known = next((info for info in self._index if info.voice_id == voice_id), None)
        stop = self._stop

        def work() -> PiperVoiceInfo:
            info = known
            if info is None:
                ensure_voice_index(on_progress=self._progress(), should_stop=stop.is_set)
                info = next((i for i in read_voice_index() if i.voice_id == voice_id), None)
            if info is None:
                raise AdapterError(f"No Piper voice '{voice_id}'. Refresh the voice list.")
            download_voice(info, on_progress=self._progress(), should_stop=stop.is_set)
            return info

        def done(result: object) -> None:
            if not isinstance(result, PiperVoiceInfo):
                return
            voices = self._voices()
            if voices is None:
                self._finish("MiraSpeak has not added the Piper engine yet.", ui.TONE_WARNING)
                return
            alias = next((a for a, v in voices.items() if v == result.voice_id), None)
            if alias is None:
                voices[voice_alias(result.name, result.quality, voices)] = result.voice_id
            try:
                self._save_voices(voices)
            except ModuleError as e:
                self._finish(str(e), ui.TONE_WARNING)
                return
            self._finish(f"Installed {result.label()}.")

        self._run("Downloading…", work, done)

    def _remove(self, alias: str) -> None:
        voices = self._voices()
        if voices is None or alias not in voices:
            self._status("That voice is not installed.", ui.TONE_WARNING)
            return
        voice_id = voices.pop(alias)
        remove_voice_files(voice_id)
        try:
            self._save_voices(voices)
        except ModuleError as e:
            self._status(str(e), ui.TONE_WARNING)
            return
        self._status(f"Removed {alias}.")

"""Speak MiraSpeak's lines into a Discord voice channel through the user's own bot."""

from __future__ import annotations

import array
import base64
import binascii
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from miraspeak.modules.api import (
    AUDIO_CHANNELS,
    AUDIO_FRAME_BYTES,
    AUDIO_SAMPLE_RATE,
    AudioStream,
    LineEvent,
    LineState,
    Module,
    ModuleContext,
    ModuleError,
    ui,
)

IDLE_CHECK_S = 30.0
MAX_AVATAR_BYTES = 8 * 1024 * 1024  # Discord refuses larger images
AVATAR_TYPES = ("png", "jpg", "jpeg", "gif", "webp")
PROFILE_KEYS = ("avatar", "set_avatar", "username", "set_name")
PORTAL_URL = "https://discord.com/developers/applications"
NO_SERVERS = (
    "Connected as {}, but the bot is not in any server. "
    "Press Open invite link, add it, then press Refresh servers."
)
MISSING_LIBRARY = "The Discord packages are not installed. Remove the module and install it again."

# View Channels (1 << 10) + Connect (1 << 20) + Speak (1 << 21)
INVITE_PERMISSIONS = 3_146_752
INVITE_URL = "https://discord.com/oauth2/authorize?client_id={}&scope=bot&permissions={}"

GUIDE_URL = "https://github.com/miramocha/MiraSpeak-docs/blob/main/guides/discord.md"
INTRO = (
    "Each person uses their own Discord bot. Press Open setup guide for the steps, "
    "with screenshots."
)


def invite_url(token: str) -> str | None:
    """Invite link for the bot that owns *token*, or None when it doesn't look like a bot token.

    The first part of a bot token is the bot's user id in base64, which is also its client id.
    """
    head = token.strip().split(".", 1)[0]
    try:
        client_id = base64.urlsafe_b64decode(head + "=" * (-len(head) % 4)).decode("ascii")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if not client_id.isdigit():
        return None
    return INVITE_URL.format(client_id, INVITE_PERMISSIONS)


def _make_runner(report: Callable[[str, Any], None]) -> Any:
    from .bot import BotRunner  # discord.py is an optional extra; import only when used

    return BotRunner(report)


TONE_HZ = 440.0
TONE_SECONDS = 1.5
TONE_LEVEL = 0.3
TONE_FADE_S = 0.05


class ToneStream:
    """A short test tone as an ``AudioStream``, so a test never depends on speech."""

    line_id = 0
    mine = True

    def __init__(self) -> None:
        total = int(AUDIO_SAMPLE_RATE * TONE_SECONDS)
        fade = int(AUDIO_SAMPLE_RATE * TONE_FADE_S)
        mono = array.array("h")
        for i in range(total):
            edge = min(1.0, i / fade, (total - 1 - i) / fade)
            wave = math.sin(2 * math.pi * TONE_HZ * i / AUDIO_SAMPLE_RATE)
            mono.append(int(wave * edge * TONE_LEVEL * 32767))
        stereo = array.array("h")
        for sample in mono:
            stereo.extend([sample] * AUDIO_CHANNELS)
        self._pcm = stereo.tobytes()
        self._at = 0

    def read_frame(self) -> bytes:
        frame = self._pcm[self._at : self._at + AUDIO_FRAME_BYTES]
        self._at += AUDIO_FRAME_BYTES
        return frame

    def close(self) -> None:
        self._at = len(self._pcm)


class DiscordModule(Module):
    # Tests replace this to run without discord.py.
    runner_factory: Callable[[Callable[[str, Any], None]], Any] = staticmethod(_make_runner)

    def page(self) -> ui.Page:
        return ui.Page(
            intro=INTRO,
            sections=(
                ui.Section(
                    "Bot",
                    (
                        ui.Button("guide", "Open setup guide"),
                        ui.Button("portal", "Open developer portal"),
                        ui.Secret("token", "Bot token", placeholder="Paste the token here"),
                        ui.Button("invite", "Open invite link"),
                        ui.Button("connect", "Connect"),
                        ui.Status("status", "Not connected."),
                    ),
                ),
                ui.Section(
                    "Bot profile",
                    (
                        ui.File("avatar", "Avatar", extensions=AVATAR_TYPES, inline=True),
                        ui.Button("set_avatar", "Update avatar", enabled=False),
                        ui.Text(
                            "username", "Name", placeholder="Discord limits how often a bot renames"
                        ),
                        ui.Button("set_name", "Update name", enabled=False),
                        ui.Toggle(
                            "show_activity",
                            'Show "Speaking as <voice>" as the bot\'s activity',
                            True,
                        ),
                    ),
                ),
                ui.Section(
                    "Voice channel",
                    (
                        ui.Choice("guild", "Server", enabled=False),
                        ui.Choice("channel", "Voice channel", enabled=False),
                        ui.Button("refresh", "Refresh servers", enabled=False),
                        ui.Button("test_sound", "Test sound", enabled=False),
                        ui.Button("join", "Join", enabled=False),
                        ui.Button("leave", "Leave", enabled=False),
                    ),
                ),
                ui.Section(
                    "Behavior",
                    (
                        ui.Toggle("auto_join", "Connect and join when MiraSpeak starts", True),
                        ui.Number(
                            "idle_minutes",
                            "Leave when idle",
                            default=10,
                            minimum=0,
                            maximum=240,
                            suffix=" min (0 = never)",
                        ),
                    ),
                ),
            ),
        )

    def on_start(self, ctx: ModuleContext) -> None:
        self.ctx = ctx
        self._runner: Any = None
        self._ready = False
        self._joined = False
        self._guilds: dict[str, tuple[tuple[str, str], ...]] = {}
        self._pending: AudioStream | None = None
        self._leave_reason = ""
        self._refresh_asked = False
        self._last_audio = time.monotonic()
        self._activity_voice = ""  # the last voice a line played with
        self._enabled = dict.fromkeys(  # matches the page's first paint
            ("guild", "channel", "refresh", "join", "test_sound", "leave", *PROFILE_KEYS[1::2]),
            False,
        )
        self._idle_timer = ctx.every(IDLE_CHECK_S, self._check_idle)
        if ctx.values.get("auto_join") and ctx.secrets.get("token"):
            self._connect()

    def on_stop(self) -> None:
        self._idle_timer.cancel()
        self._drop_pending()
        self._close_runner()

    def on_action(self, action: str, row: int | None = None) -> None:
        if action == "guide":
            self._open(GUIDE_URL)
        elif action == "portal":
            self._open(PORTAL_URL)
        elif action == "invite":
            self._open_invite()
        elif action == "connect":
            self._connect()
        elif action == "refresh":
            self._refresh()
        elif action == "test_sound":
            self._test_sound()
        elif action == "join":
            self._join()
        elif action == "leave" and self._runner is not None:
            self._runner.leave()
        elif action == "set_avatar":
            self._set_avatar()
        elif action == "set_name":
            self._set_name()

    def on_value(self, key: str, value: object) -> None:
        if key == "guild":
            self._show_channels(str(value))
        elif key == "show_activity":
            self._sync_activity()

    def on_secret(self, key: str) -> None:
        if key != "token":
            return
        self._close_runner()
        if self.ctx.secrets.get("token"):
            self._connect()
        else:
            self._status("Token cleared.")

    def on_audio(self, stream: AudioStream) -> None:
        self._last_audio = time.monotonic()
        if self._runner is None or not self._ready:
            stream.close()
        elif self._joined:
            self._runner.play(stream)
        elif self.ctx.values.get("auto_join") and self._channel_id() is not None:
            self._drop_pending()
            self._pending = stream
            self._join()
        else:
            stream.close()

    def on_line(self, event: LineEvent) -> None:
        if event.state is not LineState.PLAYING or event.silent or not event.voice_label:
            return
        if event.voice_label != self._activity_voice:
            self._activity_voice = event.voice_label
            self._sync_activity()

    def _open_invite(self) -> None:
        token = self.ctx.secrets.get("token")
        if not token:
            self._status("Paste your bot token first.", warn=True)
            return
        url = invite_url(token)
        if url is None:
            self._status(
                "That does not look like a bot token. Copy it from the Bot tab.", warn=True
            )
            return
        self._open(url)

    def _open(self, url: str) -> None:
        try:
            self.ctx.open_url(url)
        except ModuleError as e:
            self._status(f"Could not open the browser: {e}", warn=True)

    # connecting

    def _connect(self) -> None:
        token = self.ctx.secrets.get("token")
        if not token:
            self._status("Paste your bot token first.", warn=True)
            return
        self._close_runner()
        try:
            runner = type(self).runner_factory(self._report)
        except ImportError:
            self.ctx.log.exception("discord.py is missing")
            self._status(MISSING_LIBRARY, warn=True)
            return
        self._runner = runner
        self._status("Connecting…")
        runner.start(token)

    def _close_runner(self) -> None:
        runner, self._runner = self._runner, None
        self._ready = self._joined = False
        self._sync_controls()
        if runner is not None:
            runner.close()

    def _sync_controls(self) -> None:
        """Gray out the voice controls until connected, and Test sound and Leave until joined."""
        has_channels = any(self._guilds.values())
        wanted = {
            "guild": self._ready,
            "channel": self._ready,
            "refresh": self._ready,
            "join": self._ready and has_channels,
            "test_sound": self._joined,
            "leave": self._joined,
            "set_avatar": self._ready,
            "set_name": self._ready,
        }
        for key, value in wanted.items():
            if self._enabled.get(key) != value:
                self._enabled[key] = value
                self.ctx.ui.set_enabled(key, value)

    # bot profile

    def _set_avatar(self) -> None:
        if self._runner is None or not self._ready:
            self._status("Press Connect first.", warn=True)
            return
        raw = self.ctx.values.get("avatar")
        path = Path(raw) if isinstance(raw, str) and raw else None
        if path is None:
            self._status("Browse for an avatar image first.", warn=True)
            return
        try:
            if path.stat().st_size > MAX_AVATAR_BYTES:
                self._status("That image is over 8 MB. Pick a smaller one.", warn=True)
                return
            image = path.read_bytes()
        except OSError as e:
            self._status(f"Could not read the image: {e}", warn=True)
            return
        self._status("Updating avatar…")
        self._runner.set_avatar(image)

    def _set_name(self) -> None:
        if self._runner is None or not self._ready:
            self._status("Press Connect first.", warn=True)
            return
        name = str(self.ctx.values.get("username", "")).strip()
        if not 2 <= len(name) <= 32:
            self._status("A bot name needs 2 to 32 characters.", warn=True)
            return
        self._status("Renaming…")
        self._runner.set_username(name)

    def _sync_activity(self) -> None:
        """Show the last voice as the bot's activity, or clear it when the toggle is off."""
        if self._runner is None or not self._ready:
            return
        show = self.ctx.values.get("show_activity") and self._activity_voice
        self._runner.set_activity(f"Speaking as {self._activity_voice}" if show else None)

    def _test_sound(self) -> None:
        if self._runner is None or not self._joined:
            self._status("Join a voice channel first.", warn=True)
            return
        self._runner.play(ToneStream())

    def _refresh(self) -> None:
        if self._runner is None or not self._ready:
            self._status("Press Connect first.", warn=True)
            return
        self._refresh_asked = True
        self._runner.refresh()

    def _join(self) -> None:
        if self._runner is None or not self._ready:
            self._status("Press Connect first.", warn=True)
            return
        channel_id = self._channel_id()
        if channel_id is None:
            self._status("Pick a voice channel first.", warn=True)
            return
        self._status("Joining…")
        self._runner.join(channel_id)

    def _channel_id(self) -> int | None:
        raw = self.ctx.values.get("channel")
        return int(raw) if isinstance(raw, str) and raw.isdigit() else None

    # events from the bot thread, handed over to this thread

    def _report(self, kind: str, detail: Any) -> None:
        self.ctx.call_later(0, lambda: self._on_event(kind, detail))

    def _on_event(self, kind: str, detail: Any) -> None:
        if self._runner is None:
            return
        if kind == "ready":
            self._on_ready(detail)
        elif kind == "guilds":
            self._on_guilds(detail)
        elif kind == "joined":
            self._joined = True
            self._sync_controls()
            self._status(f"In #{detail}.")
            self._play_pending()
        elif kind == "left":
            self._joined = False
            self._sync_controls()
            self._drop_pending()
            self._status(self._leave_reason or "Left the voice channel.")
            self._leave_reason = ""
        elif kind == "kicked":
            self._joined = False
            self._sync_controls()
            self._drop_pending()
            self._status(f"Removed from #{detail}. Press Join to return.", warn=True)
        elif kind == "disconnected":  # login failed or the connection dropped for good
            self._ready = self._joined = False
            self._sync_controls()
            self._drop_pending()
        elif kind == "played":
            frames, peak, encrypted = detail
            loud = round(peak / 32767 * 100)
            text = f"Sent {frames * 0.02:.1f} s to Discord, peak {loud}%."
            if not encrypted:
                text += (
                    " Discord's voice encryption is not ready, so listeners may hear nothing. "
                    "Have someone in the channel before the bot joins, then press Leave and Join."
                )
            self._status(text, warn=loud == 0 or not encrypted)
        elif kind == "failed":
            self._drop_pending()
            self._status(str(detail), warn=True)
        elif kind in ("profile", "profile_failed"):
            self._status(str(detail), warn=kind == "profile_failed")

    def _on_guilds(self, info: Any) -> None:
        self._apply_guilds(info)
        if not self._refresh_asked:
            return
        self._refresh_asked = False
        count = len(info.guilds)
        if count:
            self._status(f"Found {count} server{'s' if count != 1 else ''}.")
        else:
            self._status(NO_SERVERS.format(info.name), warn=True)

    def _on_ready(self, info: Any) -> None:
        self._ready = True
        self._sync_activity()
        self._apply_guilds(info)
        if self._guilds:
            self._status(f"Connected as {info.name}.")
        else:
            self._status(NO_SERVERS.format(info.name), warn=True)
        if self.ctx.values.get("auto_join") and self._channel_id() is not None:
            self._join()

    def _apply_guilds(self, info: Any) -> None:
        self._guilds = {
            str(g.id): tuple((str(c.id), c.name) for c in g.channels) for g in info.guilds
        }
        self._sync_controls()
        self.ctx.ui.set_options("guild", [(str(g.id), g.name) for g in info.guilds])
        guild = str(self.ctx.values.get("guild", ""))
        if guild not in self._guilds and self._guilds:
            guild = next(iter(self._guilds))
            self.ctx.ui.set("guild", guild)
        self._show_channels(guild)

    def _show_channels(self, guild: str) -> None:
        channels = self._guilds.get(guild, ())
        self.ctx.ui.set_options("channel", channels)
        current = str(self.ctx.values.get("channel", ""))
        if channels and current not in {cid for cid, _name in channels}:
            self.ctx.ui.set("channel", channels[0][0])

    # audio

    def _play_pending(self) -> None:
        stream, self._pending = self._pending, None
        if stream is not None and self._runner is not None:
            self._runner.play(stream)

    def _drop_pending(self) -> None:
        stream, self._pending = getattr(self, "_pending", None), None
        if stream is not None:
            stream.close()

    def _check_idle(self) -> None:
        minutes = self.ctx.values.get("idle_minutes")
        if not self._joined or not isinstance(minutes, int | float) or minutes <= 0:
            return
        if time.monotonic() - self._last_audio >= minutes * 60:
            self._leave_reason = f"Left after {int(minutes)} min idle."
            self._runner.leave()

    def _status(self, text: str, *, warn: bool = False) -> None:
        tone = ui.TONE_WARNING if warn else ui.TONE_NEUTRAL
        self.ctx.ui.set_status("status", text, tone)

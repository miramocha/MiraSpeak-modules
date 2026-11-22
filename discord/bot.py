"""The Discord side: one discord.py client on its own thread and event loop."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

import aiohttp
import numpy as np
from miraspeak.modules.api import AudioStream

import discord

log = logging.getLogger("miraspeak.module.discord")

JOIN_TIMEOUT_S = 15.0
CLOSE_TIMEOUT_S = 4.0
ENCRYPT_WAIT_S = 4.0  # how long a line waits for Discord's voice encryption
ENCRYPT_POLL_S = 0.1
VOICE_UNREACHABLE = (
    "Couldn't reach Discord's voice server. Your network or firewall may be blocking it: "
    "voice needs outbound access beyond port 443 (UDP and the voice websocket ports)."
)
PROFILE_TIMEOUT_S = 20.0  # discord.py would otherwise sleep out a long rate limit
RATE_LIMITED_LONG = (
    "Discord says to try again later. Bots can only rename a couple of times an hour."
)
OPUS_HELP = "Discord voice needs libopus. On Linux install it (for example `apt install libopus0`)."

# (kind, detail): ready (BotInfo), failed (text), joined (channel name), left, kicked,
# disconnected (the client ended: login failed or the connection dropped for good),
# profile / profile_failed (status text for an avatar or name edit)
Report = Callable[[str, Any], None]


@dataclass(frozen=True)
class ChannelInfo:
    id: int
    name: str


@dataclass(frozen=True)
class GuildInfo:
    id: int
    name: str
    channels: tuple[ChannelInfo, ...]


@dataclass(frozen=True)
class BotInfo:
    name: str
    guilds: tuple[GuildInfo, ...]


class StreamSource(discord.AudioSource):
    """Feeds a MiraSpeak ``AudioStream`` (48 kHz stereo 16-bit PCM) to a voice client."""

    def __init__(
        self, stream: AudioStream, on_done: Callable[[int, int], None] | None = None
    ) -> None:
        self._stream = stream
        self._on_done = on_done
        self.frames = 0
        self.peak = 0  # loudest sample sent, out of 32767

    def read(self) -> bytes:
        frame = self._stream.read_frame()
        if frame:
            self.frames += 1
            self.peak = max(self.peak, int(np.abs(np.frombuffer(frame, dtype="<i2")).max()))
        return frame

    def is_opus(self) -> bool:
        return False

    def cleanup(self) -> None:
        self._stream.close()
        if self._on_done is not None and self.frames:
            self._on_done(self.frames, self.peak)


def profile_problem(error: discord.HTTPException) -> str:
    """A plain-words reason a profile edit was refused, with Discord's wait when it gives one."""
    if error.status == 429:
        retry = getattr(error.response, "headers", {}).get("Retry-After")
        try:
            wait = float(retry)
        except (TypeError, ValueError):
            return RATE_LIMITED_LONG
        minutes = max(1, round(wait / 60))
        return f"Discord says try again in {minutes} minute{'s' if minutes != 1 else ''}."
    if "too fast" in (error.text or "").lower():
        return RATE_LIMITED_LONG
    return f"Discord refused the change: {error.text or error.status}"


def join_problem(channel: Any) -> str | None:
    """Why the bot can't speak in *channel*, or None. Checks the permissions Discord enforces."""
    perms = channel.permissions_for(channel.guild.me)
    if not perms.view_channel:
        return f"The bot can't see #{channel.name}. Give it View Channel."
    if not perms.connect:
        return f"The bot lacks Connect in #{channel.name}. Change the channel's permissions."
    if not perms.speak:
        return f"The bot lacks Speak in #{channel.name}. Change the channel's permissions."
    limit = getattr(channel, "user_limit", 0)
    if limit and len(channel.members) >= limit and not perms.move_members:
        return f"#{channel.name} is full."
    return None


def silenced(voice: Any) -> str | None:
    """Why nobody can hear the bot in its current channel, or None."""
    me = voice.guild.me
    state = getattr(me, "voice", None)
    connection = getattr(voice, "_connection", None)
    log.debug(
        "voice state in #%s: server_mute=%s suppress=%s self_mute=%s dave=%s can_encrypt=%s",
        getattr(voice.channel, "name", "?"),
        getattr(state, "mute", None),
        getattr(state, "suppress", None),
        getattr(state, "self_mute", None),
        getattr(connection, "dave_protocol_version", None),
        getattr(connection, "can_encrypt", None),
    )
    name = getattr(voice.channel, "name", "the channel")
    if state is not None and getattr(state, "mute", False):
        return f"The bot is server-muted in #{name}. Right-click the bot and turn off Server Mute."
    if state is not None and getattr(state, "suppress", False):
        return f"The bot is suppressed in Stage #{name}. A moderator must invite it to speak."
    if state is not None and getattr(state, "self_mute", False):
        return f"The bot muted itself in #{name}. Press Leave, then Join again."
    return None


def encryption_ready(voice: Any) -> bool:
    """True when Discord's voice encryption (DAVE) is ready, or the channel doesn't use it."""
    connection = getattr(voice, "_connection", None)
    if getattr(connection, "dave_protocol_version", 0) == 0:
        return True
    return bool(getattr(connection, "can_encrypt", False))


async def wait_for_encryption(voice: Any) -> bool:
    """Wait up to ``ENCRYPT_WAIT_S`` for encryption; False if it never became ready.

    The session forms once another person is in the channel, so a lone bot can wait in vain.
    """
    waited = 0.0
    while not encryption_ready(voice):
        if waited >= ENCRYPT_WAIT_S or not voice.is_connected():
            return False
        await asyncio.sleep(ENCRYPT_POLL_S)
        waited += ENCRYPT_POLL_S
    return True


def guild_info(guild: Any) -> GuildInfo:
    channels = (*guild.voice_channels, *guild.stage_channels)
    return GuildInfo(guild.id, guild.name, tuple(ChannelInfo(c.id, c.name) for c in channels))


class BotRunner:
    """Owns the bot's thread. Every method is safe to call from the module thread and returns
    at once, except ``close``; results come back through *report* (from the bot thread).
    """

    def __init__(self, report: Report) -> None:
        self._report = report
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: discord.Client | None = None
        self._leaving = False

    def start(self, token: str) -> None:
        self.close()
        loop = asyncio.new_event_loop()
        self._loop = loop
        self._client = self._make_client()
        self._thread = threading.Thread(
            target=self._run,
            args=(loop, self._client, token),
            name="miraspeak-discord",
            daemon=True,
        )
        self._thread.start()

    def join(self, channel_id: int) -> None:
        self._submit(self._join(channel_id))

    def leave(self) -> None:
        self._submit(self._leave())

    def refresh(self) -> None:
        """Send the current server and channel lists again (as a ``guilds`` report)."""
        self._submit(self._announce())

    def play(self, stream: AudioStream) -> None:
        loop = self._loop
        if loop is None:
            stream.close()
            return
        self._submit(self._play(stream))

    def set_avatar(self, image: bytes) -> None:
        self._submit(self._edit_profile("Avatar updated.", avatar=image))

    def set_username(self, name: str) -> None:
        self._submit(self._edit_profile(f"Name changed to {name}.", username=name))

    def set_activity(self, text: str | None) -> None:
        """Show *text* as the bot's activity, or clear it with ``None``."""
        self._submit(self._set_activity(text))

    def close(self) -> None:
        """Leave, log out and stop the thread. Waits a few seconds at most."""
        loop, thread, client = self._loop, self._thread, self._client
        self._loop = self._thread = self._client = None
        if loop is None or thread is None or client is None:
            return
        self._leaving = True
        try:
            asyncio.run_coroutine_threadsafe(client.close(), loop).result(CLOSE_TIMEOUT_S)
        except (concurrent.futures.TimeoutError, RuntimeError):
            log.debug("discord shutdown did not finish cleanly", exc_info=True)
        thread.join(CLOSE_TIMEOUT_S)

    # bot thread

    def _make_client(self) -> discord.Client:
        intents = discord.Intents.none()
        intents.guilds = True
        intents.voice_states = True
        client = discord.Client(intents=intents)
        for handler in (
            self.on_ready,
            self.on_voice_state_update,
            self.on_guild_join,
            self.on_guild_remove,
            self.on_guild_channel_create,
            self.on_guild_channel_delete,
            self.on_guild_channel_update,
        ):
            client.event(handler)  # registered under the handler's name
        return client

    def _run(self, loop: asyncio.AbstractEventLoop, client: discord.Client, token: str) -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(client.start(token))
        except asyncio.CancelledError:
            pass  # close() ended the login on purpose
        except discord.LoginFailure:
            self._report("failed", "Discord rejected the token. Paste it again.")
            self._report("disconnected", None)
        except Exception as e:  # network or gateway errors end the bot, never MiraSpeak
            log.exception("discord client stopped")
            self._report("failed", f"Connection failed: {e}")
            self._report("disconnected", None)
        else:
            if self._loop is loop:  # not closed on purpose
                self._report("failed", "Disconnected from Discord. Press Connect to retry.")
                self._report("disconnected", None)
        finally:
            try:
                loop.run_until_complete(client.close())
            finally:
                loop.close()

    def _snapshot(self) -> BotInfo | None:
        client = self._client
        if client is None or client.user is None:
            return None
        return BotInfo(client.user.name, tuple(guild_info(g) for g in client.guilds))

    async def on_ready(self) -> None:
        if (info := self._snapshot()) is not None:
            self._report("ready", info)

    async def _announce(self) -> None:
        if (info := self._snapshot()) is not None:
            self._report("guilds", info)

    async def on_guild_join(self, _guild: Any) -> None:
        await self._announce()

    async def on_guild_remove(self, _guild: Any) -> None:
        await self._announce()

    async def on_guild_channel_create(self, _channel: Any) -> None:
        await self._announce()

    async def on_guild_channel_delete(self, _channel: Any) -> None:
        await self._announce()

    async def on_guild_channel_update(self, _before: Any, _after: Any) -> None:
        await self._announce()

    async def on_voice_state_update(self, member: Any, before: Any, after: Any) -> None:
        client = self._client
        if client is None or client.user is None or member.id != client.user.id:
            return
        if before.channel is not None and after.channel is None and not self._leaving:
            self._report("kicked", before.channel.name)

    async def _join(self, channel_id: int) -> None:
        client = self._client
        channel = client.get_channel(channel_id) if client is not None else None
        if channel is None:
            self._report("failed", "That channel is gone. Press Connect to refresh the lists.")
            return
        if not discord.opus.is_loaded() and not discord.opus._load_default():
            self._report("failed", OPUS_HELP)
            return
        problem = join_problem(channel)
        if problem is not None:
            self._report("failed", problem)
            return
        for other in list(client.voice_clients):  # audio goes to one server only
            if other.guild.id != channel.guild.id:
                self._leaving = True
                await other.disconnect(force=True)
        voice = channel.guild.voice_client
        # A timed-out connect leaves the client registered. With no channel, reading
        # voice.channel.id crashes the join; still in this channel but not connected,
        # the next attempt would report joined and then play nothing.
        if voice is not None and (voice.channel is None or not voice.is_connected()):
            await self._drop_failed_voice(channel)
            voice = None
        try:
            if voice is None:
                await channel.connect(timeout=JOIN_TIMEOUT_S, reconnect=True)
            elif voice.channel.id != channel.id:
                await voice.move_to(channel)
        except (TimeoutError, discord.DiscordException) as e:  # before OSError: it is one
            await self._drop_failed_voice(channel)
            self._report("failed", f"Couldn't join #{channel.name}: {e or 'timed out'}")
            return
        except (OSError, aiohttp.ClientError) as e:  # includes ClientConnectorError
            log.warning("discord voice server unreachable: %s", e)
            await self._drop_failed_voice(channel)
            self._report("failed", VOICE_UNREACHABLE)
            return
        self._leaving = False
        self._report("joined", channel.name)

    async def _drop_failed_voice(self, channel: Any) -> None:
        """Disconnect a half-open voice client without reporting it as a kick."""
        voice = channel.guild.voice_client
        if voice is None:
            return
        self._leaving = True
        try:
            await voice.disconnect(force=True)
        except (OSError, discord.DiscordException):
            log.debug("could not drop the failed voice client", exc_info=True)

    async def _leave(self) -> None:
        client = self._client
        if client is None:
            return
        self._leaving = True
        for voice in list(client.voice_clients):
            await voice.disconnect(force=True)
        self._report("left", "")

    async def _play(self, stream: AudioStream) -> None:
        client = self._client
        voice = next(iter(client.voice_clients), None) if client is not None else None
        if voice is None or not voice.is_connected():
            stream.close()
            return
        problem = silenced(voice)
        if problem is not None:
            stream.close()
            self._report("failed", problem)
            return
        encrypted = await wait_for_encryption(voice)
        if not voice.is_connected():
            stream.close()
            return
        if voice.is_playing():
            voice.stop()
        try:
            source = StreamSource(
                stream,
                lambda frames, peak: self._report("played", (frames, peak, encrypted)),
            )
            voice.play(source, after=self._after_play)
        except discord.DiscordException as e:  # includes OpusNotLoaded
            stream.close()
            self._report("failed", f"Couldn't play audio: {e}")

    async def _edit_profile(self, done: str, **fields: Any) -> None:
        client = self._client
        if client is None or client.user is None:
            return
        try:
            await asyncio.wait_for(client.user.edit(**fields), PROFILE_TIMEOUT_S)
        except TimeoutError:
            self._report("profile_failed", RATE_LIMITED_LONG)
        except discord.HTTPException as e:
            self._report("profile_failed", profile_problem(e))
        except (OSError, aiohttp.ClientError) as e:
            log.warning("discord profile update failed: %s", e)
            self._report("profile_failed", f"Couldn't reach Discord: {e}")
        else:
            self._report("profile", done)

    async def _set_activity(self, text: str | None) -> None:
        client = self._client
        if client is None or client.user is None:
            return
        activity = discord.CustomActivity(name=text) if text else None
        try:
            await client.change_presence(activity=activity)
        except (OSError, discord.DiscordException):
            log.debug("could not change the bot's activity", exc_info=True)

    @staticmethod
    def _after_play(error: Exception | None) -> None:
        if error is not None:
            log.warning("discord playback ended with an error: %s", error)

    def _submit(self, coro: Coroutine[Any, Any, None]) -> None:
        loop = self._loop
        if loop is None:
            coro.close()
            return
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        future.add_done_callback(self._log_failure)

    @staticmethod
    def _log_failure(future: concurrent.futures.Future) -> None:
        if not future.cancelled() and (error := future.exception()) is not None:
            log.error("discord task failed: %s", error)

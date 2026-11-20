"""Send a typed turn to a webhook and play that connection's filler while the agent replies."""

from __future__ import annotations

import json
import threading
from dataclasses import replace

import httpx
from miraspeak.modules.api import (
    FillerResult,
    HostBusyError,
    Module,
    ModuleContext,
    ModuleError,
    ModuleStoppedError,
    Timer,
    ui,
    webhook_url_allowed,
)

from .connections import (
    CAP_MAX,
    NAME_MAX,
    PRESET_GROK,
    PRESETS,
    Connection,
    ConnectionDoc,
    ConnectionFormatError,
    local_day,
    new_connection_id,
    note_send,
    parse_connections,
    sent_on,
    serialize_connections,
)
from .post import post_message

FILE_NAME = "connections.json"
FILLER_TIMEOUT_S = 60.0
THINKING_AFTER_S = 10.0
_READY_RETRY_S = 0.25
_UNREADABLE = "This agent list isn't one MiraSpeak can read."
_URL = "Use an https URL, or http to this computer."

INTRO = (
    "Sends what you type to an agent and plays a filler while you wait. The agent speaks the reply."
)


class AgentChatModule(Module):
    """Bundled page. It stays under Speak so the page can turn the module off and on."""

    def __init__(self) -> None:
        self._ctx: ModuleContext | None = None
        self._doc = ConnectionDoc()
        self._editing_id: str | None = None
        self._busy = False
        self._opened = False
        self._timer: Timer | None = None
        self._ready: Timer | None = None

    def page(self) -> ui.Page:
        return ui.Page(
            intro=INTRO,
            sections=(
                ui.Section(
                    "",
                    (
                        ui.LineBox(
                            "message",
                            agent="agent",
                            action="send",
                            button="Send",
                            placeholder="Ask something short.",
                            status="say",
                        ),
                        ui.Choice("agent", "Agent"),
                        ui.Status("say"),
                    ),
                ),
                ui.Section(
                    "Connections",
                    (
                        ui.Rows(
                            "connections",
                            "Agents",
                            row_actions=("Up", "Down", "Remove"),
                            click="edit",
                            when_empty=ui.Empty(
                                "No agents yet",
                                "Add an agent to send a turn.",
                                action="add",
                                action_label="Add",
                            ),
                        ),
                        ui.Button("add", "Add"),
                    ),
                ),
                ui.Section(
                    "Connection",
                    (
                        ui.Text("name", "Name"),
                        ui.Text("url", "Webhook URL", placeholder="https://", mono=True),
                        ui.Secret("webhook_key", "Webhook key", "Paste the key"),
                        ui.Status("key_status", "No key yet."),
                        ui.Choice(
                            "preset",
                            "Request",
                            ((PRESET_GROK, "Grok Bot routine"), ("plain", "Plain HTTP POST")),
                            default=PRESET_GROK,
                        ),
                        ui.Choice("persona", "Persona"),
                        ui.Choice("cue", "Filler cue", (("", "Default"),)),
                        ui.Number("cap", "Daily cap", maximum=CAP_MAX),
                        ui.Toggle("conn_on", "Enabled", default=True),
                        ui.Button("save", "Save"),
                        ui.Status("save_status"),
                    ),
                ),
                ui.Section(
                    "Settings",
                    (
                        ui.Choice("default_agent", "Default agent"),
                        ui.Toggle("filler", "Speak a filler", default=True),
                        ui.Text("thinking", "Still thinking", placeholder="thinking"),
                        ui.Status(
                            "thinking_hint",
                            "Cue name, played after 10 seconds. "
                            "A set without that cue keeps the first cue.",
                        ),
                    ),
                ),
            ),
        )

    def on_start(self, ctx: ModuleContext) -> None:
        self._ctx = ctx
        self._doc = self._read_doc()
        # The UI thread is still starting, so a host call here times out and the page stays off.
        self._ready = ctx.call_later(0, self._when_ready)

    def _when_ready(self) -> None:
        """Fill the page once the UI thread can answer. Retry while it is still starting."""
        ctx = self._ctx
        if ctx is None:
            return
        try:
            self._refresh()
            if not self._opened:
                self._open(0 if self._doc.connections else None)
                self._opened = True
        except HostBusyError:
            if self._ctx is not None:
                self._ready = self._ctx.call_later(_READY_RETRY_S, self._when_ready)
        except ModuleStoppedError:
            return

    def on_stop(self) -> None:
        self._cancel_timer()
        if self._ready is not None:
            self._ready.cancel()
            self._ready = None
        self._ctx = None

    def on_action(self, action: str, row: int | None = None) -> None:
        if action == "send":
            self._send()
        elif action == "save":
            self._save()
        elif action == "add":
            self._open(None)
        elif action == "edit" and row is not None:
            self._open(row)
        elif action == "Remove" and row is not None:
            self._remove(row)
        elif action == "Up" and row is not None:
            self._move(row, -1)
        elif action == "Down" and row is not None:
            self._move(row, 1)

    def on_value(self, key: str, value: object) -> None:
        if self._ctx is None:
            return
        if key == "persona" and isinstance(value, str):
            self._cues(value)
        elif key == "default_agent":
            self._show_rows()

    def on_secret(self, key: str) -> None:
        if key == "webhook_key":
            self._key_status()

    def _send(self) -> None:
        ctx = self._ctx
        if ctx is None or self._busy:
            return
        text = ctx.line("message").strip()
        conn = self._selected()
        blocked = _blocked(text, conn, self._doc)
        if blocked is not None or conn is None:
            self._status(blocked or "Turn on an agent first.")
            return
        self._busy = True
        self._set_composer(False)
        self._status("Sending…")
        try:
            session, note = self._begin_filler(conn)
        except ModuleError as exc:
            self._unlock_send(exc)
            return
        self._post_turn(conn, text, session, note)

    def _set_composer(self, enabled: bool) -> None:
        """Enable or disable the line and Send. No effect once the module has stopped."""
        ctx = self._ctx
        if ctx is None:
            return
        ctx.ui.set_enabled("message", enabled)
        ctx.ui.set_enabled("send", enabled)

    def _unlock_send(self, exc: ModuleError) -> None:
        """Release the send lock when the turn never started."""
        self._cancel_timer()
        self._busy = False
        self._set_composer(True)
        if isinstance(exc, HostBusyError):
            self._status(str(exc))

    def _begin_filler(self, conn: Connection) -> tuple[str, str]:
        ctx = self._ctx
        if ctx is None or not ctx.values.get("filler"):
            return "", ""
        started = ctx.start_filler(
            persona_id=conn.persona_id, cue=conn.cue, timeout_s=FILLER_TIMEOUT_S
        )
        if not started.ok:
            return "", f" {started.message}" if started.message else ""
        try:
            self._arm(started.session_id, conn.persona_id)
        except ModuleError:
            self._drop_filler(started.session_id)
            raise
        return started.session_id, " A filler is playing."

    def _drop_filler(self, session: str) -> None:
        """Stop a filler that started for a turn which will not be posted."""
        self._cancel_timer()
        ctx = self._ctx
        if ctx is None or not session:
            return
        try:
            ctx.stop_filler(session)
        except ModuleError:
            return

    def _arm(self, session_id: str, persona_id: str) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        cue = str(ctx.values.get("thinking") or "").strip()
        if not cue or not _has_cue(ctx, persona_id, cue):
            return
        self._timer = ctx.call_later(
            THINKING_AFTER_S, lambda: self._think(session_id, persona_id, cue)
        )

    def _think(self, session_id: str, persona_id: str, cue: str) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        ctx.cue_filler(
            persona_id=persona_id, cue=cue, session_id=session_id, timeout_s=FILLER_TIMEOUT_S
        )

    def _post_turn(self, conn: Connection, text: str, session: str, note: str) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        key = ctx.secrets.get(f"conn-{conn.id}") or ""
        url, preset, name, conn_id = conn.url, conn.preset, conn.name, conn.id

        def work() -> None:
            code = _post(ctx, url, key, preset, text)
            try:
                ctx.call_later(0, lambda: self._posted(code, name, conn_id, session, note))
            except ModuleError:
                return

        threading.Thread(target=work, name="miraspeak-agentchat", daemon=True).start()

    def _posted(self, code: int, name: str, conn_id: str, session: str, note: str) -> None:
        self._busy = False
        ctx = self._ctx
        if ctx is None:
            return
        self._set_composer(True)
        if code != 200:
            self._cancel_timer()
            if session:
                ctx.stop_filler(session)
            self._status("The agent didn't start.")
            return
        self._doc = note_send(self._doc, conn_id, local_day())
        self._write()
        ctx.ui.set_line("message", "")
        self._status(f"Sent to {name}.{note}")

    def _save(self) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        draft = self._draft()
        if isinstance(draft, str):
            ctx.ui.set_status("save_status", draft)
            return
        try:
            if ctx.secrets.get("webhook_key"):
                ctx.secrets.copy("webhook_key", f"conn-{draft.id}")
        except (ModuleError, ValueError) as exc:
            ctx.ui.set_status("save_status", str(exc))
            return
        self._doc = _replace(self._doc, draft)
        self._editing_id = draft.id
        self._write()
        self._refresh()
        self._key_status()
        ctx.ui.set_status("save_status", "Saved.")

    def _remove(self, row: int) -> None:
        ctx = self._ctx
        if ctx is None or not 0 <= row < len(self._doc.connections):
            return
        conn = self._doc.connections[row]
        ctx.secrets.clear(f"conn-{conn.id}")
        rows = self._doc.connections[:row] + self._doc.connections[row + 1 :]
        self._doc = replace(self._doc, connections=rows)
        self._write()
        self._refresh()
        if self._editing_id == conn.id:
            self._open(0 if rows else None)
        else:
            self._highlight()

    def _move(self, row: int, step: int) -> None:
        rows = list(self._doc.connections)
        other = row + step
        if other < 0 or other >= len(rows):
            return
        rows[row], rows[other] = rows[other], rows[row]
        self._doc = replace(self._doc, connections=tuple(rows))
        self._write()
        self._refresh()
        self._highlight()

    def _open(self, index: int | None) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        if index is None or not 0 <= index < len(self._doc.connections):
            self._editing_id = None
            self._blank()
            ctx.ui.highlight_row("connections", None)
            return
        conn = self._doc.connections[index]
        self._editing_id = conn.id
        self._personas()
        self._cues(conn.persona_id)
        ctx.ui.set("name", conn.name)
        ctx.ui.set("url", conn.url)
        ctx.ui.set("preset", conn.preset)
        ctx.ui.set("persona", conn.persona_id)
        ctx.ui.set("cue", conn.cue)
        ctx.ui.set("cap", conn.daily_cap)
        ctx.ui.set("conn_on", conn.enabled)
        ctx.secrets.clear("webhook_key")
        self._key_status()
        ctx.ui.highlight_row("connections", index)

    def _blank(self) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        self._personas()
        ctx.ui.set("name", "")
        ctx.ui.set("url", "")
        ctx.ui.set("preset", PRESET_GROK)
        ctx.ui.set("persona", "")
        self._cues("")
        ctx.ui.set("cue", "")
        ctx.ui.set("cap", 0)
        ctx.ui.set("conn_on", True)
        ctx.secrets.clear("webhook_key")
        self._key_status()

    def _draft(self) -> Connection | str:
        ctx = self._ctx
        if ctx is None:
            return "Turn Agent Chat on to save."
        name = str(ctx.values.get("name") or "").strip()
        if not name or len(name) > NAME_MAX:
            return "Name the connection."
        url = str(ctx.values.get("url") or "").strip()
        if not webhook_url_allowed(url):
            return _URL
        cap = ctx.values.get("cap")
        if isinstance(cap, bool) or not isinstance(cap, int) or not 0 <= cap <= CAP_MAX:
            return "Set the daily cap from 0 to 500."
        preset = ctx.values.get("preset")
        if preset not in PRESETS:
            preset = PRESET_GROK
        return Connection(
            self._editing_id or new_connection_id(),
            name,
            url,
            str(preset),
            str(ctx.values.get("persona") or ""),
            str(ctx.values.get("cue") or ""),
            cap,
            bool(ctx.values.get("conn_on")),
        )

    def _refresh(self) -> None:
        self._show_rows()
        self._agent_options()
        self._personas()
        self._highlight()

    def _show_rows(self) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        default = str(ctx.values.get("default_agent") or "")
        ctx.ui.set_rows(
            "connections", tuple(_row_label(conn, default) for conn in self._doc.connections)
        )

    def _agent_options(self) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        options = tuple((conn.id, conn.name) for conn in self._doc.connections if conn.enabled)
        shown = options or (("", "No agent is on"),)
        ctx.ui.set_options("agent", shown)
        ctx.ui.set_options("default_agent", shown)
        if not options:
            return
        ids = {option_id for option_id, _name in options}
        self._keep_choice("agent", ids, options[0][0])
        self._keep_choice("default_agent", ids, options[0][0])

    def _keep_choice(self, key: str, ids: set[str], fallback: str) -> None:
        """Leave a saved id that is still listed. An empty or missing one takes *fallback*."""
        ctx = self._ctx
        if ctx is None:
            return
        current = ctx.values.get(key)
        if isinstance(current, str) and current in ids:
            return
        ctx.ui.set(key, fallback)

    def _personas(self) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        people = ctx.list_personas()
        options = tuple(
            (persona.id, persona.name if not persona.vacant else f"{persona.name} (no voice)")
            for persona in people
        )
        ctx.ui.set_options("persona", options or (("", "No personas"),))

    def _cues(self, persona_id: str) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        listed = ctx.list_filler_cues(persona_id) if persona_id else ()
        names = () if isinstance(listed, FillerResult) else tuple(item.name for item in listed)
        ctx.ui.set_options("cue", (("", "Default"),) + tuple((name, name) for name in names))

    def _highlight(self) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        index = next(
            (i for i, conn in enumerate(self._doc.connections) if conn.id == self._editing_id),
            None,
        )
        ctx.ui.highlight_row("connections", index)

    def _key_status(self) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        if ctx.secrets.get("webhook_key"):
            ctx.ui.set_status("key_status", "A new key will replace the saved one.")
            return
        conn_id = self._editing_id
        if conn_id and ctx.secrets.get(f"conn-{conn_id}"):
            ctx.ui.set_status("key_status", "Key saved.")
            return
        ctx.ui.set_status("key_status", "No key yet.")

    def _selected(self) -> Connection | None:
        ctx = self._ctx
        if ctx is None:
            return None
        conn = self._doc.get(str(ctx.values.get("agent") or ""))
        if conn is None or not conn.enabled:
            return None
        return conn

    def _status(self, text: str) -> None:
        if self._ctx is not None:
            self._ctx.ui.set_status("say", text)

    def _read_doc(self) -> ConnectionDoc:
        ctx = self._ctx
        if ctx is None:
            return ConnectionDoc()
        raw = ctx.data.read_text(FILE_NAME)
        if not raw:
            return ConnectionDoc()
        try:
            return parse_connections(json.loads(raw))
        except (json.JSONDecodeError, ConnectionFormatError):
            self._status(_UNREADABLE)
            return ConnectionDoc()

    def _write(self) -> None:
        if self._ctx is not None:
            self._ctx.data.write_text(FILE_NAME, json.dumps(serialize_connections(self._doc)))

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


def _blocked(text: str, conn: Connection | None, doc: ConnectionDoc) -> str | None:
    """Why this turn cannot be sent, or None when it can."""
    if not text:
        return "Type a message."
    if conn is None:
        return "Turn on an agent first."
    if not webhook_url_allowed(conn.url):
        return _URL
    if conn.daily_cap and sent_on(doc, conn.id, local_day()) >= conn.daily_cap:
        return "This agent is at its daily limit."
    return None


def _has_cue(ctx: ModuleContext, persona_id: str, cue: str) -> bool:
    listed = ctx.list_filler_cues(persona_id)
    if isinstance(listed, FillerResult):
        return False
    return any(item.name == cue for item in listed)


def _replace(doc: ConnectionDoc, conn: Connection) -> ConnectionDoc:
    rows = list(doc.connections)
    for index, old in enumerate(rows):
        if old.id == conn.id:
            rows[index] = conn
            return replace(doc, connections=tuple(rows))
    rows.append(conn)
    return replace(doc, connections=tuple(rows))


def _row_label(conn: Connection, default_id: str) -> str:
    kind = "Grok Bot routine" if conn.preset == PRESET_GROK else "Plain HTTP POST"
    bits = [conn.name, kind]
    if not conn.enabled:
        bits.append("Off")
    if conn.id == default_id:
        bits.append("Default")
    return " · ".join(bits)


def _post(ctx: ModuleContext, url: str, key: str, preset: str, text: str) -> int:
    try:
        return post_message(url, key, preset=preset, text=text)
    except httpx.HTTPError as exc:
        ctx.log.warning("agent webhook failed: %s", type(exc).__name__)
    except ValueError as exc:
        ctx.log.warning("agent webhook refused: %s", exc)
    except Exception:
        ctx.log.exception("agent webhook failed")
    return 0

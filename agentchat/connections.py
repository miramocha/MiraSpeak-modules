"""Agent connections: the list and today's send counts. Keys stay in the keyring."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

PRESET_GROK = "grok"
PRESET_PLAIN = "plain"
PRESETS = frozenset({PRESET_GROK, PRESET_PLAIN})
DOC_VERSION = 1
NAME_MAX = 32
CAP_MAX = 500
_DAY = 10
_ID = re.compile(r"^[a-z][a-z0-9-]{0,24}$")


class ConnectionFormatError(ValueError):
    """The agent list is missing, malformed, or an unsupported version."""


@dataclass(frozen=True)
class Connection:
    """One webhook the page can send a turn to. The key is not stored here."""

    id: str
    name: str
    url: str
    preset: str
    persona_id: str
    cue: str
    daily_cap: int
    enabled: bool


@dataclass(frozen=True)
class DayCount:
    """How many turns *id* has sent on one local date."""

    id: str
    day: str
    count: int


@dataclass(frozen=True)
class ConnectionDoc:
    """The connection file. Order is the list order."""

    connections: tuple[Connection, ...] = ()
    counts: tuple[DayCount, ...] = ()

    def get(self, conn_id: str) -> Connection | None:
        for conn in self.connections:
            if conn.id == conn_id:
                return conn
        return None


def local_day() -> str:
    """Today's date in the local timezone, ``YYYY-MM-DD``."""
    return datetime.now().astimezone().date().isoformat()


def new_connection_id() -> str:
    """A slug that can name a keyring entry ``conn-<id>``."""
    return "c" + uuid.uuid4().hex[:8]


def sent_on(doc: ConnectionDoc, conn_id: str, day: str) -> int:
    """How many turns *conn_id* has counted on *day*."""
    for count in doc.counts:
        if count.id == conn_id and count.day == day:
            return count.count
    return 0


def note_send(doc: ConnectionDoc, conn_id: str, day: str) -> ConnectionDoc:
    """Add one send for *day* and drop every other day's counts."""
    current = sent_on(doc, conn_id, day)
    kept = tuple(count for count in doc.counts if count.day == day and count.id != conn_id)
    return ConnectionDoc(doc.connections, (*kept, DayCount(conn_id, day, current + 1)))


def serialize_connections(doc: ConnectionDoc) -> dict[str, Any]:
    """*doc* as a versioned JSON object."""
    return {
        "version": DOC_VERSION,
        "connections": [
            {
                "id": conn.id,
                "name": conn.name,
                "url": conn.url,
                "preset": conn.preset,
                "persona_id": conn.persona_id,
                "cue": conn.cue,
                "daily_cap": conn.daily_cap,
                "enabled": conn.enabled,
            }
            for conn in doc.connections
        ],
        "counts": [
            {"id": count.id, "day": count.day, "count": count.count} for count in doc.counts
        ],
    }


def parse_connections(data: Any) -> ConnectionDoc:
    """Validate one document. Raises ConnectionFormatError. Unknown keys are ignored."""
    if not isinstance(data, dict) or data.get("version") != DOC_VERSION:
        raise ConnectionFormatError("This agent list isn't one MiraSpeak can read.")
    raw = data.get("connections")
    counts = data.get("counts", [])
    if not isinstance(raw, list) or not isinstance(counts, list):
        raise ConnectionFormatError("This agent list isn't one MiraSpeak can read.")
    seen: set[str] = set()
    connections: list[Connection] = []
    for item in raw:
        conn = _connection(item)
        if conn.id in seen:
            raise ConnectionFormatError("This agent list isn't one MiraSpeak can read.")
        seen.add(conn.id)
        connections.append(conn)
    return ConnectionDoc(tuple(connections), tuple(_count(item) for item in counts))


def _connection(item: Any) -> Connection:
    if not isinstance(item, dict):
        raise ConnectionFormatError("This agent list isn't one MiraSpeak can read.")
    conn_id = item.get("id")
    name = item.get("name")
    url = item.get("url")
    preset = item.get("preset")
    persona = item.get("persona_id")
    cue = item.get("cue")
    cap = item.get("daily_cap")
    enabled = item.get("enabled")
    if (
        not isinstance(conn_id, str)
        or not isinstance(name, str)
        or not isinstance(url, str)
        or preset not in PRESETS
        or not isinstance(persona, str)
        or not isinstance(cue, str)
        or isinstance(cap, bool)
        or not isinstance(cap, int)
        or not isinstance(enabled, bool)
        or not _ID.fullmatch(conn_id)
        or not name.strip()
        or len(name.strip()) > NAME_MAX
        or cap < 0
        or cap > CAP_MAX
    ):
        raise ConnectionFormatError("This agent list isn't one MiraSpeak can read.")
    return Connection(conn_id, name.strip(), url.strip(), preset, persona, cue, cap, enabled)


def _count(item: Any) -> DayCount:
    if not isinstance(item, dict):
        raise ConnectionFormatError("This agent list isn't one MiraSpeak can read.")
    conn_id, day, count = item.get("id"), item.get("day"), item.get("count")
    if (
        not isinstance(conn_id, str)
        or not isinstance(day, str)
        or len(day) != _DAY
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
    ):
        raise ConnectionFormatError("This agent list isn't one MiraSpeak can read.")
    return DayCount(conn_id, day, count)

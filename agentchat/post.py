"""POST one turn to an agent webhook. The response body is ignored."""

from __future__ import annotations

import httpx
from miraspeak.modules.api import webhook_url_allowed

from .connections import PRESET_GROK, PRESET_PLAIN

POST_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


def post_message(
    url: str,
    key: str,
    *,
    preset: str,
    text: str,
    client: httpx.Client | None = None,
) -> int:
    """POST ``{"message": text}``. Returns the status code. Raises httpx.HTTPError.

    A saved key is sent as ``Authorization: Bearer``. Grok Bot routine and plain POST
    use the same body. The caller closes nothing; a client passed in stays open.
    """
    if preset not in {PRESET_GROK, PRESET_PLAIN} or not webhook_url_allowed(url):
        raise ValueError("That webhook URL is not one MiraSpeak will call.")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    own = client is None
    http = client or httpx.Client(timeout=POST_TIMEOUT, follow_redirects=False)
    try:
        response = http.post(url, json={"message": text}, headers=headers)
        return response.status_code
    finally:
        if own:
            http.close()

"""Alert channels: Telegram (free, works on a phone) and, if set up, email.

Telegram needs a bot token, TELEGRAM_BOT_TOKEN, as an environment variable or Streamlit secret, and a chat
to write to. That chat is either a fixed one, TELEGRAM_CHAT_ID (a team running its own copy), or one a
user links for their own session: they open the bot from a link with a one-time code and press Start, and
`find_chat` reads the code back from the bot's updates. So on a shared, public app every visitor gets
alerts on their own phone, and nobody can message a chat that did not ask for it.

Nothing here is sent unless something calls it: the live monitor on a new alert, the "Send alerts"
button on the evacuation plan, or a test button in Team and alerts. The token is never put in a message
or an error.
"""

from __future__ import annotations

import json
import os
import secrets
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.telegram.org"
TIMEOUT_S = 10
KEEP_UPDATES = 30  # recent /start messages kept, so links opened in the last few minutes are still found


def _setting(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def bot_ready() -> bool:
    """There is a bot, so users can link their own chat to it."""
    return bool(_setting("TELEGRAM_BOT_TOKEN"))


def _chat(chat_id: str | None) -> str:
    return str(chat_id or _setting("TELEGRAM_CHAT_ID"))


def telegram_configured(chat_id: str | None = None) -> bool:
    """A bot and a chat to write to: `chat_id` (a user's own) or the fixed TELEGRAM_CHAT_ID."""
    return bool(bot_ready() and _chat(chat_id))


def ready_channels(chat_id: str | None = None) -> list[str]:
    """The alert channels that are set up, as the app names them: "Telegram" and/or "email"."""
    from emailer import email_configured

    return [name for name, ready in (("Telegram", telegram_configured(chat_id)), ("email", email_configured()))
            if ready]


_bot_names: dict[str, str] = {}


def bot_username() -> str | None:
    """The bot's @name, for the link that opens it; None if Telegram can't be reached."""
    token = _setting("TELEGRAM_BOT_TOKEN")
    if token and token not in _bot_names:
        try:
            reply = _post("getMe", {})
        except Exception:
            return None
        if reply.get("ok"):
            _bot_names[token] = reply["result"]["username"]
    return _bot_names.get(token)


def new_link_code() -> str:
    """A one-time code for the link that opens the bot (letters and digits, as Telegram allows)."""
    return secrets.token_hex(8)


def connect_link(code: str, group: bool = False, via: str = "link") -> str | None:
    """The link that opens the bot with `code`: in a private chat, or (group=True) to add it to a group, such as
    a team of police officers or volunteers, so one alert reaches all of them.

    `via`: "link" is the t.me link (phone camera, or any browser); "app" opens the Telegram app installed on
    this computer directly, which also works where a network blocks t.me; "web" opens Telegram Web.
    """
    name = bot_username()
    if not name:
        return None
    start = "startgroup" if group else "start"
    app = f"tg://resolve?domain={name}&{start}={code}"
    if via == "app":
        return app
    if via == "web":
        return "https://web.telegram.org/k/#?tgaddr=" + urllib.parse.quote(app, safe="")
    return f"https://t.me/{name}?{start}={code}"


last_check: dict = {"ok": None, "problem": "", "seen": 0}  # how the last find_chat went, for the page to show


def find_chat(code: str) -> tuple[str, str] | None:
    """(chat id, name) of the chat that opened the bot with this code, if any has: a person who pressed Start
    ("/start <code>"), or a group the bot was just added to ("/start@<bot> <code>")."""
    try:
        reply = _post("getUpdates", {"allowed_updates": '["message"]', "limit": 100})
    except urllib.error.HTTPError as exc:
        last_check.update(ok=False, problem=f"Telegram answered {exc.code}")
        return None
    except Exception as exc:  # network down, timeout, bad reply
        last_check.update(ok=False, problem=f"Telegram could not be reached ({type(exc).__name__})")
        return None
    updates = reply.get("result", [])
    last_check.update(ok=True, problem="", seen=len(updates))
    if len(updates) > KEEP_UPDATES:  # Telegram keeps unread updates for a day; let go of all but the newest
        try:
            _post("getUpdates", {"offset": updates[-KEEP_UPDATES]["update_id"], "limit": 1})
        except Exception:
            pass
    for update in reversed(updates):
        message = update.get("message") or {}
        words = (message.get("text") or "").split()
        if len(words) == 2 and words[0].split("@")[0] == "/start" and words[1] == code:
            chat = message["chat"]
            return str(chat["id"]), chat.get("title") or chat.get("first_name") or ""
    return None


def _post(method: str, fields: dict, files: dict | None = None) -> dict:
    token = _setting("TELEGRAM_BOT_TOKEN")
    url = f"{API}/bot{token}/{method}"
    if files:
        boundary = "----crowdshield" + os.urandom(8).hex()
        body = b""
        for key, value in fields.items():
            body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n").encode()
        for key, (filename, data) in files.items():
            body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"; filename=\"{filename}\"\r\n"
                     "Content-Type: image/jpeg\r\n\r\n").encode() + data + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        request = urllib.request.Request(url, data=body, headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}"})
    else:
        request = urllib.request.Request(url, data=urllib.parse.urlencode(fields).encode())
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return json.loads(response.read().decode())


def send_telegram(text: str, image_bytes: bytes | None = None, chat_id: str | None = None) -> tuple[bool, str]:
    """Send a message (and optionally a photo) to `chat_id`, or the fixed chat. Returns (ok, message)."""
    if not telegram_configured(chat_id):
        return False, "Telegram is not set up: connect your Telegram in Team and alerts."
    chat = _chat(chat_id)
    try:
        if image_bytes:
            reply = _post("sendPhoto", {"chat_id": chat, "caption": text[:1024]},
                          {"photo": ("zone.jpg", image_bytes)})
        else:
            reply = _post("sendMessage", {"chat_id": chat, "text": text[:4096]})
    except urllib.error.HTTPError as exc:
        reason = {401: "the bot token is wrong", 400: "the chat id is wrong, or the bot was never started "
                  "(open the bot in Telegram and press Start)", 403: "the bot was blocked or removed from the chat",
                  404: "the bot token is wrong"}.get(exc.code, f"Telegram answered {exc.code}")
        return False, f"Telegram refused the message: {reason}."
    except Exception as exc:  # network down, timeout, bad reply
        return False, f"Could not reach Telegram ({type(exc).__name__})."
    return (True, "Sent to Telegram.") if reply.get("ok") else (False, "Telegram did not accept the message.")


def alert_text(zone: str, name: str, risk: str, count: float, per_m2: float | None = None,
               minutes_to_danger: float | None = None) -> str:
    """The text of a crowd alert: short enough to read on a lock screen."""
    head = {"Critical": "CRITICAL", "High Risk": "HIGH RISK"}.get(risk, risk.upper())
    lines = [f"Crowd Shield: {head} in zone {zone}" + (f" ({name})" if name and name != zone else ""),
             f"About {count:,.0f} people" + (f", {per_m2:.1f} per m2" if per_m2 else "") + "."]
    if minutes_to_danger is not None and minutes_to_danger > 0:
        lines.append(f"At this pace it reaches 5 per m2 in about {minutes_to_danger:.0f} min.")
    lines.append("Close entries to this zone and open its exits.")
    return "\n".join(lines)


def broadcast(text: str, image_bytes: bytes | None = None, email_to: list[str] | None = None,
              chat_id: str | None = None) -> list[str]:
    """Send an alert on every channel that is set up. Returns one status line per attempt."""
    results = []
    if telegram_configured(chat_id):
        ok, message = send_telegram(text, image_bytes, chat_id)
        results.append(f"Telegram: {message}")
    if email_to:
        from emailer import email_configured, send_email

        if email_configured():
            for address in email_to:
                ok, message = send_email(address, "Crowd Shield alert", f"<pre>{text}</pre>", image_bytes)
                results.append(f"Email {address}: {message}")
    if not results:
        results.append("No alert channel is set up (connect Telegram or add Gmail in Team and alerts).")
    return results

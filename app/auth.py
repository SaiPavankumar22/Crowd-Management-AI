"""Optional API authentication with two roles.

Disabled unless a token is configured, so a loopback-only demo behind a password
proxy behaves exactly as before. When tokens exist, reads accept either token and
every mutating request needs the operator token. Static assets, the dashboard
shell and the health endpoint stay public so container checks and the page itself
keep working.
"""
from __future__ import annotations

import hmac

from fastapi import Request

from . import config

PUBLIC_PATHS = {"/", "/index.html", "/api/health", "/openapi.json",
                "/docs", "/redoc"}
PUBLIC_PREFIXES = ("/static/",)


def configured() -> bool:
    return bool(config.AUTH_OPERATOR_TOKEN or config.AUTH_VIEWER_TOKEN)


def is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


def token_from(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    key = request.headers.get("x-api-key", "").strip()
    if key:
        return key
    # Streams and image tags cannot set headers; the query form is the escape
    # hatch for those, and for a dashboard URL opened with ?token=...
    return (request.query_params.get("token") or "").strip()


def role_for(token: str) -> str | None:
    if not token:
        return None
    if config.AUTH_OPERATOR_TOKEN and hmac.compare_digest(
            token, config.AUTH_OPERATOR_TOKEN):
        return "operator"
    if config.AUTH_VIEWER_TOKEN and hmac.compare_digest(
            token, config.AUTH_VIEWER_TOKEN):
        return "viewer"
    return None


async def _deny(send, status: int, detail: str):
    body = ('{"detail": "%s"}' % detail).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                            (b"www-authenticate", b"Bearer")]})
    await send({"type": "http.response.body", "body": body})


class ApiAuthMiddleware:
    """ASGI middleware: authenticate before any route sees the request."""

    def __init__(self, application):
        self.application = application

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not configured():
            return await self.application(scope, receive, send)
        path = scope.get("path", "")
        if is_public(path):
            return await self.application(scope, receive, send)

        role = role_for(token_from(Request(scope)))
        if role is None:
            return await _deny(send, 401, "missing or invalid API token")
        method = scope.get("method", "GET").upper()
        if method not in ("GET", "HEAD", "OPTIONS") and role != "operator":
            return await _deny(send, 403, "operator token required")
        return await self.application(scope, receive, send)

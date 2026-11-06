"""FastAPI composition root."""
import asyncio
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .api import router as api_router
from .auth import ApiAuthMiddleware
from .district import publisher as district_publisher
from .mongo_sync import mirror as mongo_mirror, mirror_camera
from .state import broadcaster, model, registry

app = FastAPI(title="Godavari Crowd Monitor")
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

class UploadLimitMiddleware:
    """Reject oversized camera uploads before Starlette spools multipart data."""

    def __init__(self, application, limit: int):
        self.application, self.limit = application, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/api/cameras" \
                or scope.get("method") != "POST":
            return await self.application(scope, receive, send)
        headers = dict(scope.get("headers", []))
        length = headers.get(b"content-length")
        if length:
            try:
                if int(length) > self.limit + 1024 * 1024:
                    return await self._too_large(send)
            except ValueError:
                pass
        total = 0

        async def limited_receive():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.limit + 1024 * 1024:
                    raise OverflowError
            return message

        try:
            await self.application(scope, limited_receive, send)
        except OverflowError:
            await self._too_large(send)

    @staticmethod
    async def _too_large(send):
        body = b"upload exceeds configured size limit"
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"text/plain; charset=utf-8"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

app.add_middleware(UploadLimitMiddleware, limit=config.MAX_UPLOAD_BYTES)
app.add_middleware(ApiAuthMiddleware)

class StaticRevalidateMiddleware:
    """Make the dashboard revalidate its JS/CSS instead of trusting a stale copy.

    Without an explicit Cache-Control header browsers apply heuristic caching,
    so after redeploying static/ an operator could keep running the previous
    dashboard for hours. `no-cache` still allows the 304 path: the ETag check
    costs one conditional request per asset per load.
    """

    def __init__(self, application):
        self.application = application

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope.get("path", "").startswith("/static/"):
            return await self.application(scope, receive, send)

        async def send_with_header(message):
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", [])
                           if k.lower() != b"cache-control"]
                headers.append((b"cache-control", b"no-cache"))
                message = {**message, "headers": headers}
            await send(message)

        await self.application(scope, receive, send_with_header)

app.add_middleware(StaticRevalidateMiddleware)

@app.on_event("startup")
async def startup():
    mongo_mirror.start()
    for camera in list(registry.cameras.values()):
        mirror_camera(camera)
    if district_publisher.enabled:
        district_publisher.start()
    asyncio.create_task(_bootstrap())

async def _bootstrap():
    await asyncio.to_thread(model.load)
    await asyncio.to_thread(registry.load_and_start)
    await broadcaster()

@app.get("/")
async def index():
    return FileResponse(os.path.join(config.STATIC_DIR, "index.html"))

app.include_router(api_router)

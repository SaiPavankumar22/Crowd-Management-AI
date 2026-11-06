"""Live video: MJPEG streams, snapshots and the dashboard websocket."""
import time

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from .. import auth, config
from ..state import registry, ws_clients

router = APIRouter()

@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    # A websocket never reaches HTTP middleware, so the token has to be checked
    # on the handshake itself; otherwise enabling auth would leave the full
    # snapshot stream open to anyone who can reach the port.
    if auth.configured() and auth.role_for(ws.query_params.get("token", "")) is None:
        await ws.close(code=1008)
        return
    await ws.accept()
    ws_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)

@router.get("/api/cameras/{cam_id}/stream")
async def cam_stream(cam_id: str, raw: int = 0):
    """MJPEG stream. `raw=1` serves the same frames without detection boxes,
    so a split view can show the annotated and the clean picture together."""
    if registry.get(cam_id) is None:
        raise HTTPException(404, "camera not found")

    def frames():
        period = 1.0 / max(1, config.STREAM_FPS)
        while registry.get(cam_id) is not None:
            worker = registry.workers.get(cam_id)
            jpeg = None
            if worker is not None:
                jpeg = worker.raw_jpeg() if raw else worker.latest_jpeg
            if jpeg is None:
                time.sleep(0.2)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(period)

    return StreamingResponse(frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/api/cameras/{cam_id}/snapshot")
async def cam_snapshot(cam_id: str):
    worker = registry.workers.get(cam_id)
    jpeg = worker.latest_jpeg if worker else None
    if jpeg is None:
        raise HTTPException(503, "no frame yet")
    return StreamingResponse(iter([jpeg]), media_type="image/jpeg")

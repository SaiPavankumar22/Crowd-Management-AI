"""HTTP routers, one module per concern.

`api.router` is the single include point for the application.
"""
from fastapi import APIRouter

from . import alerts, cameras, engine, insights, stream
from ..district import router as district_router

router = APIRouter()
router.include_router(cameras.router)
router.include_router(alerts.router)
router.include_router(engine.router)
router.include_router(insights.router)
router.include_router(district_router)
router.include_router(stream.router)

__all__ = ["router"]

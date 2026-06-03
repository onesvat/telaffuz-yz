from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/api/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from api.services.recordings import list_recordings, recording_paths

router = APIRouter(prefix="/api/v1/admin")


def _admin_password(request: Request) -> str:
    return str(getattr(request.app.state, "admin_password", ""))


def _require_admin(request: Request, password_value: str | None) -> None:
    password = _admin_password(request)
    if not password:
        raise HTTPException(503, "admin password is not configured")
    if password_value != password:
        raise HTTPException(401, "invalid admin password")


@router.post("/login")
async def admin_login(request: Request, body: dict[str, str]) -> dict[str, bool]:
    if not _admin_password(request):
        raise HTTPException(503, "admin password is not configured")
    if body.get("password") != _admin_password(request):
        raise HTTPException(401, "invalid admin password")
    return {"ok": True}


@router.get("/recordings")
async def admin_recordings(
    request: Request,
    x_admin_password: Annotated[str | None, Header()] = None,
    limit: int = 100,
    word: str | None = Query(None),
    source: str | None = Query(None),
    status: str | None = Query(None),
) -> dict[str, object]:
    _require_admin(request, x_admin_password)
    safe_limit = max(1, min(limit, 500))
    return {
        "items": list_recordings(
            request.app.state.config.app_recordings_dir,
            limit=safe_limit,
            word=word,
            source=source,
            status=status,
        )
    }


@router.get("/recordings/{recording_id}/audio")
async def admin_recording_audio(
    request: Request,
    recording_id: str,
    x_admin_password: Annotated[str | None, Header()] = None,
    admin_password: str | None = Query(None),
) -> FileResponse:
    _require_admin(request, x_admin_password or admin_password)
    try:
        wav_path, _ = recording_paths(request.app.state.config.app_recordings_dir, recording_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(wav_path, media_type="audio/wav", filename=wav_path.name)


@router.get("/recordings/{recording_id}/result")
async def admin_recording_result(
    request: Request,
    recording_id: str,
    x_admin_password: Annotated[str | None, Header()] = None,
) -> JSONResponse:
    _require_admin(request, x_admin_password)
    try:
        _, result_path = recording_paths(request.app.state.config.app_recordings_dir, recording_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    if not result_path.exists():
        raise HTTPException(404, "recording result not found")
    return JSONResponse(content=json.loads(result_path.read_text(encoding="utf-8")))

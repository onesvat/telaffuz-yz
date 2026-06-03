"""Recording capture router (prompt manifest + per-session WAV attempts).

The same capture flow serves two isolated collections, selected by the config
attributes a router is built against:

- ``studio.router`` — validation set (``prompts_tsv`` / ``recordings_dir``).
- ``demo_studio.router`` — demo showcase set (``demo_prompts_tsv`` /
  ``demo_recordings_dir``), so demo captures never mix into the validation set.

Endpoints (per collection prefix):

- ``GET    .../prompts``                                — prompt manifest.
- ``GET    .../recordings/{session}``                   — session attempts.
- ``POST   .../recordings/{session}/{prompt}``          — save attempt.
- ``GET    .../recordings/{session}/{prompt}/audio``    — play attempt.
- ``DELETE .../recordings/{session}/{prompt}``          — delete attempt.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from api.services import studio


def build_studio_router(
    *, prefix: str, prompts_attr: str, recordings_attr: str, collection_name: str
) -> APIRouter:
    """Build a capture router bound to a config prompts/recordings collection.

    ``prompts_attr`` and ``recordings_attr`` name the ``AppConfig`` properties
    the endpoints read, so a single implementation serves both the validation
    and demo collections without sharing storage.
    """
    router = APIRouter(prefix=prefix)

    def prompts_path(request: Request) -> Path:
        return getattr(request.app.state.config, prompts_attr)

    def recordings_path(request: Request) -> Path:
        return getattr(request.app.state.config, recordings_attr)

    @router.get("/prompts")
    async def studio_prompts(request: Request) -> dict[str, object]:
        try:
            items = studio.load_studio_prompts(prompts_path(request))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"items": items}

    @router.get("/recordings/{session_id}")
    async def studio_session(request: Request, session_id: str) -> dict[str, object]:
        try:
            return studio.session_recordings(recordings_path(request), session_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.post("/recordings/{session_id}/{prompt_id}")
    async def studio_upload(
        request: Request,
        session_id: str,
        prompt_id: str,
        audio: UploadFile = File(...),
        duration_ms: int = Form(1000),
    ) -> dict[str, object]:
        try:
            path, quality, recording_id = await studio.save_recording(
                recordings_path(request),
                session_id,
                prompt_id,
                audio,
                duration_ms=duration_ms,
                archive_recordings_dir=request.app.state.config.app_recordings_dir,
                collection_name=collection_name,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        if path is None:
            raise HTTPException(422, detail={"quality": quality.as_dict()})

        stem = path.stem.rsplit("-a", 1)
        attempt = int(stem[1]) if len(stem) == 2 and stem[1].isdigit() else None
        return {
            "ok": True,
            "recording_filename": path.name,
            "recording_id": recording_id,
            "attempt": attempt,
            "quality": quality.as_dict(),
        }

    @router.get("/recordings/{session_id}/{prompt_id}/audio")
    async def studio_audio(
        request: Request,
        session_id: str,
        prompt_id: str,
        attempt: int | None = Query(None),
    ) -> FileResponse:
        try:
            path = studio.recording_path(
                recordings_path(request), session_id, prompt_id, attempt
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if path is None:
            raise HTTPException(404, "recording not found")
        return FileResponse(path, media_type="audio/wav", filename=path.name)

    @router.delete("/recordings/{session_id}/{prompt_id}")
    async def studio_delete(
        request: Request,
        session_id: str,
        prompt_id: str,
        attempt: int = Query(...),
    ) -> dict[str, object]:
        try:
            remaining = studio.delete_recording(
                recordings_path(request), session_id, prompt_id, attempt
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True, "remaining_attempts": remaining}

    return router


router = build_studio_router(
    prefix="/api/v1/studio",
    prompts_attr="prompts_tsv",
    recordings_attr="recordings_dir",
    collection_name="studio",
)

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import load_config
from api.routers import admin, demo, demo_studio, health, studio


DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
DEFAULT_ADMIN_PASSWORD = "iuc-onur"


def _cors_origins_from_env() -> list[str]:
    configured = os.getenv("TELAFFUZ_CORS_ORIGINS", "")
    origins = [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    return DEFAULT_CORS_ORIGINS + origins


def create_app(*, data_root: Path | None = None) -> FastAPI:
    config = load_config(data_root=data_root)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config.sandbox_dir.mkdir(parents=True, exist_ok=True)
        config.app_recordings_dir.mkdir(parents=True, exist_ok=True)
        app.state.config = config
        configured_admin_password = os.getenv("TELAFFUZ_ADMIN_PASSWORD", "")
        if configured_admin_password:
            app.state.admin_password = configured_admin_password
        else:
            app.state.admin_password = DEFAULT_ADMIN_PASSWORD
            print(
                "TELAFFUZ_ADMIN_PASSWORD is not set; using default admin password: "
                f"{DEFAULT_ADMIN_PASSWORD}",
                flush=True,
            )
        yield

    app = FastAPI(title="Telaffuz YZ Demo API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins_from_env(),
        allow_origin_regex=r"https?://[^/]+:5173",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(demo.router)
    app.include_router(studio.router)
    app.include_router(demo_studio.router)
    app.include_router(admin.router)
    return app


app = create_app()

"""Demo showcase capture router.

A second instance of the validation capture flow, bound to the isolated
``demo_capture`` collection so demo recordings never land in the validation
set. Exposed publicly (no admin gate) at ``/api/v1/demo-studio`` so the demo
word set can be recorded by several speakers from a shareable link.
"""

from __future__ import annotations

from api.routers.studio import build_studio_router

router = build_studio_router(
    prefix="/api/v1/demo-studio",
    prompts_attr="demo_prompts_tsv",
    recordings_attr="demo_recordings_dir",
    collection_name="demo_studio",
)

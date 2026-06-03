from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence

import uvicorn


SHUTDOWN_TIMEOUT_SECONDS = 3.0
DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8000
DEFAULT_FRONTEND_HOST = "0.0.0.0"
DEFAULT_FRONTEND_PORT = 5173


def run_api() -> None:
    uvicorn.run(
        "api.main:create_app",
        host=os.getenv("TELAFFUZ_API_HOST", DEFAULT_API_HOST),
        port=_int_env("TELAFFUZ_API_PORT", DEFAULT_API_PORT),
        reload=True,
        reload_dirs=["src"],
        factory=True,
    )


def run_dev() -> None:
    command_specs = [
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:create_app",
            "--factory",
            "--host",
            os.getenv("TELAFFUZ_API_HOST", DEFAULT_API_HOST),
            "--port",
            str(_int_env("TELAFFUZ_API_PORT", DEFAULT_API_PORT)),
            "--reload",
            "--reload-dir",
            "src",
        ],
        [
            "pnpm",
            "--dir",
            "frontend",
            "exec",
            "vite",
            "--host",
            os.getenv("TELAFFUZ_FRONTEND_HOST", DEFAULT_FRONTEND_HOST),
            "--port",
            str(_int_env("TELAFFUZ_FRONTEND_PORT", DEFAULT_FRONTEND_PORT)),
        ],
    ]
    procs: list[subprocess.Popen[bytes]] = []
    for command in command_specs:
        procs.append(subprocess.Popen(command, start_new_session=True))
    stop_code = 0

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stop_code
        stop_code = 128 + _signum
        raise SystemExit(stop_code)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        raise SystemExit(_wait_for_exit(procs, stop_code=lambda: stop_code))
    finally:
        _terminate(procs)


def _wait_for_exit(
    procs: Sequence[subprocess.Popen[bytes]],
    *,
    stop_code: Callable[[], int],
) -> int:
    while True:
        for proc in procs:
            code = proc.poll()
            if code is None:
                continue
            _terminate(procs)
            return stop_code() or code
        time.sleep(0.2)


def _terminate(procs: Sequence[subprocess.Popen[bytes]]) -> None:
    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_SECONDS

    for proc in procs:
        if proc.poll() is not None:
            continue
        _signal_process_group(proc, signal.SIGTERM)

    while time.monotonic() < deadline:
        if all(proc.poll() is not None for proc in procs):
            break
        time.sleep(0.05)

    for proc in procs:
        if proc.poll() is None:
            _signal_process_group(proc, signal.SIGKILL)
            try:
                proc.wait()
            except (ProcessLookupError, OSError):
                pass


def _signal_process_group(proc: subprocess.Popen[bytes], signum: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signum)
    except (AttributeError, ProcessLookupError, OSError):
        if signum == signal.SIGTERM:
            proc.terminate()
        else:
            proc.kill()


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {value!r}") from exc


if __name__ == "__main__":
    run_api()

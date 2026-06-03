#!/usr/bin/env python3.14
"""Export ``thesis/thesis.docx`` to PDF after refreshing Writer fields.

LibreOffice's batch ``--convert-to pdf`` path does not reliably update the
Pandoc-generated table-of-contents field. This script opens the document
through UNO, refreshes document indexes/text fields, saves the DOCX, and then
exports the PDF.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("SAL_USE_VCLPLUGIN", "svp")
os.environ.pop("DISPLAY", None)
_runtime_dir = "/tmp/telaffuz-yz-lo-runtime"
os.makedirs(_runtime_dir, mode=0o700, exist_ok=True)
os.chmod(_runtime_dir, 0o700)
os.environ["XDG_RUNTIME_DIR"] = _runtime_dir

import uno
from com.sun.star.beans import PropertyValue

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCX_PATH = REPO_ROOT / "thesis" / "thesis.docx"
PDF_PATH = REPO_ROOT / "thesis" / "thesis.pdf"
COMBINED_MD_PATH = REPO_ROOT / "thesis" / "_build" / "thesis-combined.md"
REFS_PATH = REPO_ROOT / "thesis" / "refs.bib"


def prop(name: str, value: object) -> PropertyValue:
    item = PropertyValue()
    item.Name = name
    item.Value = value
    return item


def file_url(path: Path) -> str:
    return uno.systemPathToFileUrl(str(path.resolve()))


def start_headless_office() -> tuple[object, subprocess.Popen[bytes]]:
    office = shutil.which("soffice") or shutil.which("libreoffice")
    if office is None:
        raise RuntimeError("missing LibreOffice executable: soffice/libreoffice")

    pipe_name = f"telaffuz_yz_{os.getpid()}"
    profile_dir = Path("/tmp/telaffuz-yz-lo-profile")
    profile_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        office,
        "--headless",
        "--norestore",
        "--nolockcheck",
        "--nodefault",
        "--nofirststartwizard",
        f"-env:UserInstallation={file_url(profile_dir)}",
        f"--accept=pipe,name={pipe_name};urp;StarOffice.ComponentContext",
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    local_context = uno.getComponentContext()
    resolver = local_context.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver",
        local_context,
    )
    connection = f"uno:pipe,name={pipe_name};urp;StarOffice.ComponentContext"
    for _ in range(60):
        if process.poll() is not None:
            break
        try:
            return resolver.resolve(connection), process
        except Exception:
            time.sleep(0.5)

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    raise RuntimeError("could not connect to headless LibreOffice")


def command_available(command: str) -> bool:
    try:
        return subprocess.run(
            [command, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
    except OSError:
        return False


def find_typst() -> str:
    candidates = [
        shutil.which("typst"),
        "/home/onur/.local/share/mise/installs/typst/0.14.2/"
        "typst-x86_64-unknown-linux-musl/typst",
        "/home/onur/.local/opt/quarto-1.9.36/bin/tools/x86_64/typst",
    ]
    for candidate in candidates:
        if candidate and command_available(candidate):
            return candidate
    raise RuntimeError("missing usable typst executable")


def refresh_indexes(document: object) -> None:
    text_fields = document.getTextFields()
    if text_fields is not None:
        text_fields.refresh()

    indexes = document.getDocumentIndexes()
    for index in range(indexes.getCount()):
        indexes.getByIndex(index).update()


def export_pdf_with_uno() -> None:
    context, office_process = start_headless_office()
    service_manager = context.ServiceManager
    desktop = service_manager.createInstanceWithContext(
        "com.sun.star.frame.Desktop", context
    )

    document: object | None = None
    document = desktop.loadComponentFromURL(
        file_url(DOCX_PATH),
        "_blank",
        0,
        (
            prop("Hidden", True),
            prop("ReadOnly", False),
            prop("UpdateDocMode", 3),
        ),
    )
    if document is None:
        raise RuntimeError(f"could not open DOCX: {DOCX_PATH}")

    try:
        document.lockControllers()
        refresh_indexes(document)
        document.unlockControllers()
        document.store()
        document.storeToURL(
            file_url(PDF_PATH),
            (
                prop("FilterName", "writer_pdf_Export"),
                prop("Overwrite", True),
            ),
        )
    finally:
        try:
            if document.hasControllersLocked():
                document.unlockControllers()
        except Exception:
            pass
        if document is not None:
            document.close(True)
        if office_process is not None:
            office_process.terminate()
            try:
                office_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                office_process.kill()


def export_pdf_with_typst() -> None:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise RuntimeError("missing pandoc executable")
    typst = find_typst()
    if not COMBINED_MD_PATH.exists():
        raise RuntimeError(f"missing combined Markdown: {COMBINED_MD_PATH}")
    if not REFS_PATH.exists():
        raise RuntimeError(f"missing bibliography: {REFS_PATH}")
    cmd = [
        pandoc,
        str(COMBINED_MD_PATH),
        "--from=markdown+tex_math_dollars",
        "--citeproc",
        f"--bibliography={REFS_PATH}",
        f"--pdf-engine={typst}",
        "--pdf-engine-opt=--root",
        "--pdf-engine-opt=/",
        "-V",
        "mainfont=Liberation Serif",
        "-V",
        "sansfont=Liberation Sans",
        "-V",
        "monofont=Liberation Mono",
        "-V",
        "papersize=a4",
        "-o",
        str(PDF_PATH),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    if not DOCX_PATH.exists():
        sys.exit(f"missing DOCX: {DOCX_PATH}")

    try:
        export_pdf_with_uno()
    except Exception as exc:
        print(
            f"LibreOffice UNO export unavailable ({exc}); falling back to Pandoc+Typst.",
            file=sys.stderr,
        )
        export_pdf_with_typst()

    print(f"OK: {PDF_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

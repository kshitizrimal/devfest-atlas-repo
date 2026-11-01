"""A small API and the single page. Run:  uv run uvicorn atlas.app:app --port 8090"""
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import store
from .ask import ask
from .config import DATA, ROOT
from .ingest import ingest, safe_title

app = FastAPI(title="Atlas & Sub-agents template")
app.mount("/pages", StaticFiles(directory=DATA / "pages"), name="pages")     # page images, so a citation shows its box
KINDS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


@app.get("/")
def page():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/items")
def items():
    return store.rows("select item_id, title, kind, pages from items order by rowid desc")


@app.post("/api/items")
async def add(file: UploadFile = File(...)):
    """Ingest: read once, point always. Waits until the file is read and indexed. # template: no job queue."""
    if Path(file.filename or "").suffix.lower() not in KINDS:
        raise HTTPException(400, "PDFs and images only")
    path = DATA / "files" / Path(file.filename).name
    path.write_bytes(await file.read())
    return await ingest(path, safe_title(file.filename))


@app.post("/api/ask")
async def question(text: str = Form(...), photo: UploadFile | None = File(None)):
    """Ask: plan, search side by side, keep, write once, check every sentence, leave notes."""
    return await ask(text, await photo.read() if photo else None)

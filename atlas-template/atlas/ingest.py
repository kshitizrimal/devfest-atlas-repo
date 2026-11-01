"""PHASE 1, INGEST: read once, point always. Runs once per file, when it arrives.

    check  →  read with place  →  organise  →  point + index

No model writes anything about the file. Gemini only copies the words it sees, with the box where it saw them.
"""
import asyncio
import hashlib
import io
import json
import re
from pathlib import Path

import fitz                                   # PyMuPDF: a PDF's own text layer already knows where its words are
from PIL import Image
from pydantic import BaseModel

from . import embed, gemini, store, system1
from .config import DATA

DPI = 110                                     # page images for the UI. PDF points × DPI/72 = pixels on that image
READS_AT_ONCE = 4                             # scans read by Gemini at the same time


# ---- 1. check ----------------------------------------------------------------------------------------
def fingerprint(data: bytes) -> str:
    """Same bytes, same fingerprint: a file is never read twice."""
    return hashlib.sha256(data).hexdigest()


# ---- 2. read with place --------------------------------------------------------------------------------
class Region(BaseModel):
    box_2d: list[int]       # [ymin, xmin, ymax, xmax] on a 0–1000 grid, Gemini's own convention
    kind: str               # heading, paragraph, caption, sign, photo …
    text: str               # the words, copied as printed, in any script
    lang: str = ""          # the language of those words (English, Nepali, Thai …), so a question can ask about it


class PageRead(BaseModel):
    regions: list[Region]


READ = ("Copy every region of this page in reading order: headings, paragraphs, captions, and any words inside "
        "photos, signs or banners, and the language of its words. Give each its box_2d as [ymin, xmin, ymax, xmax] on a 0-1000 grid. Copy the words "
        "exactly as printed, in any script. Do not summarise, translate or describe.")


async def read_picture(png: bytes, w: int, h: int) -> list[dict]:
    """A scan or a photo: one Gemini call returns its regions, each with words and a box. Boxes become pixels."""
    r = await gemini.typed([gemini.image(png), READ], PageRead)
    out = []
    for g in r.regions:
        if len(g.box_2d) != 4 or not g.text.strip():
            continue
        y0, x0, y1, x1 = g.box_2d
        text = f"[{g.lang}] {g.text.strip()}" if g.lang else g.text.strip()
        out.append({"kind": g.kind, "text": text, "box": [x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000]})
    return out


def text_layer(page: fitz.Page) -> list[dict]:
    """A page with real text needs no model: each text block already has its box."""
    s = DPI / 72
    return [{"kind": "paragraph", "text": " ".join(t.split()), "box": [x0 * s, y0 * s, x1 * s, y1 * s]}
            for x0, y0, x1, y1, t, _, typ in page.get_text("blocks") if typ == 0 and t.strip()]


async def read_pdf(path: Path, item_id: str) -> list[dict]:
    doc, gate = fitz.open(path), asyncio.Semaphore(READS_AT_ONCE)

    async def one(n: int) -> dict:
        page = doc[n]
        pix = page.get_pixmap(dpi=DPI)
        image = DATA / "pages" / f"{item_id}-{n + 1}.png"
        pix.save(image)
        regions = text_layer(page)
        # template: photos printed on a text page are not read here; a page agent with a zoom tool could copy their words.
        if sum(len(r["text"]) for r in regions) < 40:          # no real text layer: a scan, so Gemini reads it
            async with gate:
                regions = await read_picture(image.read_bytes(), pix.width, pix.height)
        return {"page": n + 1, "image": image.name, "w": pix.width, "h": pix.height, "regions": regions}

    return list(await asyncio.gather(*(one(n) for n in range(len(doc)))))


async def read_photo(data: bytes, item_id: str) -> list[dict]:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    image = DATA / "pages" / f"{item_id}-1.png"
    image.write_bytes(buf.getvalue())
    regions = await read_picture(buf.getvalue(), img.width, img.height)
    # the whole photo is a region too, so a question can land on the picture itself
    whole = {"kind": "picture", "text": " · ".join(r["text"] for r in regions) or "a photo", "box": [0, 0, img.width, img.height]}
    return [{"page": 1, "image": image.name, "w": img.width, "h": img.height, "regions": [whole, *regions], "photo": img}]


# ---- 3. organise ---------------------------------------------------------------------------------------
def group(blocks: list[dict], limit: int = 900) -> list[list[dict]]:
    """Blocks become passages: neighbours on one page, up to about `limit` characters, never across a page."""
    out: list[list[dict]] = []
    for b in blocks:
        if out and out[-1][-1]["page"] == b["page"] and sum(len(x["text"]) for x in out[-1]) + len(b["text"]) <= limit:
            out[-1].append(b)
        else:
            out.append([b])
    return out


def block_id(item_id: str, page: int, k: int) -> str:
    """A pointer's address: the item, the page, the box number on that page."""
    return f"{item_id}#p{page}#b{k}"


# ---- the whole phase -----------------------------------------------------------------------------------
async def ingest(path: Path, title: str) -> dict:
    data = path.read_bytes()
    sha = fingerprint(data)                                                   # 1. check
    if seen := store.rows("select * from items where sha256 = ?", sha):
        return {**seen[0], "duplicate": True}
    item_id, is_pdf = sha[:10], data[:4] == b"%PDF"

    pages = await (read_pdf(path, item_id) if is_pdf else read_photo(data, item_id))   # 2. read with place

    blocks = [{"block_id": block_id(item_id, p["page"], k), "page": p["page"], **r}     # 3. organise
              for p in pages for k, r in enumerate(p["regions"], 1)]
    passages = [{"passage_id": f"{item_id}#c{n}", "page": g[0]["page"], "header": f"{title} · p.{g[0]['page']}",
                 "text": "\n".join(b["text"] for b in g), "block_ids": [b["block_id"] for b in g]}
                for n, g in enumerate(group(blocks), 1)]
    # template: "personal" is only a stored label (passages.personal); nothing hides these passages yet.
    # Access control would read it in ask.py before the search.
    flags = await system1.decide_many("personal", [{"text": p["text"][:1500]} for p in passages])

    atlas = [(f"{item_id}#item", "item", 1, f"{title} · {'PDF' if is_pdf else 'photo'} · {len(pages)} page(s) · "   # 4. point
              + " ".join(b["text"] for b in blocks)[:400])]
    atlas += [(f"{item_id}#page{p['page']}", "page", p["page"], f"{title} · p.{p['page']} · {p['regions'][0]['text'][:200]}")
              for p in pages if p["regions"]]
    atlas += [(f"{b['block_id']}#pic", "picture", b["page"], f"{title} · p.{b['page']} · {b['kind']}: {b['text'][:300]}")
              for b in blocks if b["kind"] in ("picture", "photo", "sign", "banner")]

    # index: every passage and atlas row, and the photo itself, in EmbeddingGemma 2's one space.
    # Embed first, then store everything together: a file that fails part-way leaves nothing behind, so it can be retried.
    pv = await asyncio.to_thread(embed.documents, [f"{p['header']}\n{p['text']}" for p in passages], title)
    av = await asyncio.to_thread(embed.documents, [t for *_, t in atlas], title)
    photo = pages[0].get("photo")
    picv = await asyncio.to_thread(embed.picture, photo) if photo is not None else None

    with store.con() as c:
        c.execute("insert into items values (?, ?, ?, ?, ?)", (item_id, sha, title, "pdf" if is_pdf else "photo", len(pages)))
        c.executemany("insert into pages values (?, ?, ?, ?, ?)", [(item_id, p["page"], p["image"], p["w"], p["h"]) for p in pages])
        c.executemany("insert into blocks values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      [(b["block_id"], item_id, b["page"], *b["box"], b["kind"], b["text"]) for b in blocks])
        c.executemany("insert into passages values (?, ?, ?, ?, ?, ?, ?)",
                      [(p["passage_id"], item_id, p["page"], p["header"], p["text"], json.dumps(p["block_ids"]), int(bool(f.value)))
                       for p, f in zip(passages, flags)])
        c.executemany("insert into atlas values (?, ?, ?, ?, ?)", [(k, item_id, lvl, pg, t) for k, lvl, pg, t in atlas])
    store.put_vectors("passage", [(p["passage_id"], item_id, v) for p, v in zip(passages, pv)])
    store.put_vectors("atlas", [(k, item_id, v) for (k, *_), v in zip(atlas, av)])
    if picv is not None:
        store.put_vectors("picture", [(blocks[0]["block_id"], item_id, picv)])
    return {"item_id": item_id, "title": title, "pages": len(pages), "blocks": len(blocks), "passages": len(passages),
            "atlas_rows": len(atlas), "personal": sum(bool(f.value) for f in flags)}


def safe_title(name: str) -> str:
    return re.sub(r"[_\-]+", " ", Path(name).stem).strip() or "untitled"

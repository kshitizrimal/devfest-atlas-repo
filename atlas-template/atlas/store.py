"""Where reading leaves its pointers: one SQLite file, plus vectors searched in numpy.

Nothing in here is written by a model. Every row is written by code from what was read, and every row points back
to a page and a box. That is the "point always" rule.

  items     one per file, keyed by its fingerprint (read once)
  pages     the page image the UI shows, so a citation can light up its box
  blocks    THE POINTERS: item, page, box in pixels, the words copied from that box
  passages  blocks grouped for search, each with a header written by code
  atlas     the map of the library: one row per item and per picture, each pointing at a page
  notes     what a checked answer leaves behind: the sentence, its pointer, the question it answered
  vectors   EmbeddingGemma 2 vectors for passages, atlas rows and notes
"""
import json
import sqlite3

import numpy as np

from .config import DATA

DB = DATA / "atlas.db"

SCHEMA = """
create table if not exists items   (item_id text primary key, sha256 text unique, title text, kind text, pages int);
create table if not exists pages   (item_id text, page int, image text, w int, h int, primary key (item_id, page));
create table if not exists blocks  (block_id text primary key, item_id text, page int, x0 real, y0 real, x1 real, y1 real,
                                    kind text, text text);
create table if not exists passages(passage_id text primary key, item_id text, page int, header text, text text,
                                    block_ids text, personal int default 0);
create table if not exists atlas   (row_id text primary key, item_id text, level text, page int, text text);
create table if not exists notes   (note_id text primary key, text text, question text, block_id text);
create table if not exists vectors (key text, kind text, item_id text, vec blob, primary key (key, kind));
"""


def con() -> sqlite3.Connection:
    """A fresh connection per call: agent tools may run on other threads. # template: fine for one person, not a server."""
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def rows(sql: str, *args) -> list[dict]:
    with con() as c:
        return [dict(r) for r in c.execute(sql, args)]


def put_vectors(kind: str, keyed: list[tuple[str, str, np.ndarray]]) -> None:
    """keyed: (key, item_id, unit vector). kind: passage | atlas | note | note_q."""
    with con() as c:
        c.executemany("insert or replace into vectors values (?, ?, ?, ?)",
                      [(k, kind, item, np.asarray(v, dtype=np.float32).tobytes()) for k, item, v in keyed])


def search(kinds: list[str], q: np.ndarray, k: int = 12, item_id: str | None = None) -> list[tuple[str, str, str, float]]:
    """Nearest vectors by cosine (vectors are unit length, so a dot product): (key, kind, item_id, score), best first.
    # template: brute force over every vector, fine for a few thousand; use a vector database beyond that."""
    sql = f"select key, kind, item_id, vec from vectors where kind in ({','.join('?' * len(kinds))})"
    args: list = list(kinds)
    if item_id:
        sql, args = sql + " and item_id = ?", args + [item_id]
    found = rows(sql, *args)
    if not found:
        return []
    m = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in found])
    scores = m @ np.asarray(q, dtype=np.float32)
    top = np.argsort(-scores)[:k]
    return [(found[i]["key"], found[i]["kind"], found[i]["item_id"], float(scores[i])) for i in top]


def passage(pid: str) -> dict | None:
    r = rows("select * from passages where passage_id = ?", pid)
    if not r:
        return None
    r[0]["block_ids"] = json.loads(r[0]["block_ids"])
    return r[0]


def pointers(block_ids: list[str]) -> list[dict]:
    """The addresses an answer may cite: item, page, box, and the page image to show."""
    if not block_ids:
        return []
    q = ",".join("?" * len(block_ids))
    return rows(f"""select b.block_id, b.item_id, i.title, b.page, b.x0, b.y0, b.x1, b.y1, p.image, p.w, p.h
                    from blocks b join items i using (item_id) join pages p on p.item_id = b.item_id and p.page = b.page
                    where b.block_id in ({q})""", *block_ids)

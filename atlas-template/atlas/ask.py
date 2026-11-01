"""PHASE 2, ASK: search side by side, prove every sentence. Runs every time a question arrives.

    plan  →  search (scouts)  →  keep  →  write  →  check  →  note

A model writes once, after the evidence is found, and every sentence it writes is checked against its source
before anyone sees it.
"""
import asyncio
import io
import re
import time
import unicodedata
import uuid
from collections import Counter

from PIL import Image
from pydantic import BaseModel

from . import embed, gemini, scouts, store, system1
from .lens import steer

MAX_SCOUTS = 8          # never more than this many scouts for one question
KEEP, RELEVANT = 8, 1.5 # keep the best 8 passages that score at least 1.5 of 4


class Sentence(BaseModel):
    text: str
    cites: list[int]    # evidence row numbers it rests on; empty only for "the records do not say"


class Draft(BaseModel):
    sentences: list[Sentence]


WRITE = ("Answer the question using only the numbered evidence below. Write short, plain sentences in the language "
         "of the question. Every sentence must cite the evidence rows it rests on. If the evidence does not answer, "
         "write one sentence that says so, with no cites.\n\nQuestion: {q}\n\nEvidence:\n{rows}")


# ---- helpers (pure, tested in tests/test_core.py) ------------------------------------------------------
def ascii_digits(s: str) -> str:
    """Any script's digits (०१२, ๑๒๓ …) as 0-9, and no thousands commas, so numbers compare across languages."""
    return "".join(str(unicodedata.digit(ch)) if ch.isdigit() else ch for ch in s).replace(",", "")


def numbers_ok(sentence: str, source: str) -> str | None:
    """Plain code: every number in the sentence must appear in its cited source. Returns the reason, or None."""
    src = ascii_digits(source)
    for n in re.findall(r"\d+(?:\.\d+)?", ascii_digits(sentence)):
        if n not in src:
            return f"{n} is not in the cited source"
    return None


def passage_of_block(block_id: str) -> str | None:
    """The passage that holds a block (a note or a picture points at a block)."""
    for r in store.rows("select passage_id, block_ids from passages where block_ids like ?", f'%"{block_id}"%'):
        return r["passage_id"]
    return None


# ---- the whole phase -----------------------------------------------------------------------------------
async def ask(question: str, photo: bytes | None = None) -> dict:
    trace, t0 = [], time.perf_counter()

    def step(name: str, detail: str) -> None:
        trace.append({"step": name, "detail": detail, "s": round(time.perf_counter() - t0, 1)})

    # 1. PLAN: the atlas, the notes and any lens say where to look. Code sets the budget, no model.
    # template: code picks the jobs straight from the atlas; a planner agent could write richer jobs per scout.
    q = await asyncio.to_thread(embed.query, question)
    lenses = [await asyncio.to_thread(embed.picture, Image.open(io.BytesIO(photo)))] if photo else []
    qv = steer(q, lenses)                                                   # big idea 3: the lens bends the question
    hits = store.search(["atlas", "picture", "note", "note_q"], qv, k=16)
    items = [i for i, _ in Counter(i for _, _, i, _ in hits).most_common()][:MAX_SCOUTS - 1]
    titles = {r["item_id"]: r["title"] for r in store.rows("select item_id, title from items")}
    jobs = [{"item_id": i, "title": titles.get(i, i), "seeds": [k for k, *_ in store.search(["passage"], qv, 6, i)]} for i in items]
    jobs.append({"item_id": None, "title": "the whole library", "seeds": [k for k, *_ in store.search(["passage"], qv, 6)]})
    from_notes = [passage_of_block(r["block_id"]) for k, kind, *_ in hits if kind in ("note", "note_q")
                  for r in store.rows("select block_id from notes where note_id = ?", k)]
    from_pictures = [passage_of_block(k) for k, kind, *_ in hits if kind == "picture"]
    step("plan", f"{len(jobs)} scouts: " + ", ".join(j["title"] for j in jobs))

    # 2. SEARCH: every scout at once (Antigravity, sealed and on a budget)
    reports = await scouts.run(jobs, question)
    chosen = [p for r in reports if r and r.verdict != "absent" for p in r.passages]
    seeds = [p for j in jobs for p in j["seeds"]]
    candidates = [p for p in dict.fromkeys([*chosen, *from_notes, *from_pictures, *seeds]) if p and store.passage(p)][:20]
    step("search", f"{sum(r is not None for r in reports)} of {len(jobs)} scouts reported, {len(candidates)} passages")

    # 3. KEEP: System 1 scores each passage 0-4, all at once, and the best are kept
    ps = [store.passage(p) for p in candidates]
    scores = await system1.decide_many("relevance", [{"question": question, "passage": f"{p['header']}\n{p['text'][:1500]}"} for p in ps])
    ranked = sorted(zip(ps, scores), key=lambda x: -(x[1].value or 0))
    kept = [p for p, d in ranked if (d.value or 0) >= RELEVANT][:KEEP] or [p for p, _ in ranked[:3]]
    step("keep", f"{len(kept)} of {len(ps)} passages kept by System 1")

    # 4. WRITE: Gemini, once, from the original passages (never from a summary)
    rows = "\n\n".join(f"[{n}] {p['header']}\n{p['text']}" for n, p in enumerate(kept, 1))
    draft = await gemini.typed([WRITE.format(q=question, rows=rows)], Draft)
    step("write", f"{len(draft.sentences)} sentences from {len(kept)} evidence rows")

    # 5. CHECK: System 1 asks if each sentence is supported, and code checks every number
    cited = [[kept[c - 1] for c in s.cites if 0 < c <= len(kept)] for s in draft.sentences]
    to_check = [i for i, c in enumerate(cited) if c]
    support = dict(zip(to_check, await system1.decide_many(
        "supported", [{"sentence": draft.sentences[i].text, "sources": "\n---\n".join(p["text"] for p in cited[i])[:4000]} for i in to_check])))
    sentences, removed = [], []
    for i, s in enumerate(draft.sentences):
        if not cited[i]:
            sentences.append({"text": s.text, "citations": [], "checked": "no evidence cited"})
            continue
        reason = numbers_ok(s.text, " ".join(p["text"] for p in cited[i])) or (None if support[i].value else "the cited sources do not say this")
        if reason:
            removed.append({"text": s.text, "reason": reason})
        else:
            blocks = [b for p in cited[i] for b in p["block_ids"]]
            sentences.append({"text": s.text, "citations": store.pointers(blocks), "checked": f"supported ({support[i].by}, {support[i].confidence:.2f})"})
    step("check", f"{sum(1 for s in sentences if s['citations'])} sentences checked, {len(removed)} removed")

    # 6. NOTE: each checked sentence leaves a pointer behind, found by the next similar question (big idea 4)
    notes = [(f"note-{uuid.uuid4().hex[:8]}", s) for s in sentences if s["citations"]]
    if notes:
        with store.con() as c:
            c.executemany("insert into notes values (?, ?, ?, ?)", [(k, s["text"], question, s["citations"][0]["block_id"]) for k, s in notes])
        sv = await asyncio.to_thread(embed.documents, [s["text"] for _, s in notes])
        store.put_vectors("note", [(k, s["citations"][0]["item_id"], v) for (k, s), v in zip(notes, sv)])
        store.put_vectors("note_q", [(k, s["citations"][0]["item_id"], q) for k, s in notes])   # as the question, too
    step("note", f"{len(notes)} notes left for next time")

    return {"question": question, "sentences": sentences, "removed": removed, "trace": trace}

# Atlas & Sub-agents · a lean template

A small, readable template of a **knowledge backbone** for archives of scans, photos and documents in many languages.
It accompanies the DevFest Bangkok 2026 talk *Every archive holds the answer. Almost nobody can find it.*

This is an **architectural template, not a product**. It is about 850 lines: a FastAPI backend, one HTML page, and
comments that say which idea each part shows. Use it to learn the pattern, then build your own.

## The idea, in four rules

```
        INGEST · once, when a file arrives                     ASK · every time a question arrives
  ┌─────────┬─────────────────┬──────────┬──────────┐    ┌──────┬────────┬──────┬───────┬───────┬──────┐
  │  check  │ read with place │ organise │  point + │    │ plan │ search │ keep │ write │ check │ note │
  │ (code)  │    (Gemini)     │(code+S1) │  index   │    │(code)│(scouts)│ (S1) │(Gemini│ (S1 + │(code)│
  └─────────┴─────────────────┴──────────┴────┬─────┘    └──┬───┴────────┴──────┴ once)┴ code) ┴──┬───┘
                                              ▼             ▲                                      │
                                  ┌──────────────────────────────────┐                             │
                                  │ the atlas of pointers + one index │◀────── notes ───────────────┘
                                  └──────────────────────────────────┘
  Read once.  Point always.                                Search side by side.  Prove every sentence.
```

It started from Andrej Karpathy's [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f):
compile knowledge once instead of re-deriving it on every question. The change here is who writes. In the wiki a
model writes pages. **Here no model writes anything about a file.** Code writes pointers, and each pointer leads back
to a page and a box in the original. A model writes once per question, from the original passages, and every
sentence it writes is checked before you see it.

## Four big ideas, and where they live

| Idea | What it is | Made in | Used in | Code |
|---|---|---|---|---|
| **Pointer** | a fact's address: item, page, box | ingest | ask: every sentence cites them | `store.py` (blocks), `ingest.py` |
| **Atlas** | a map of every item and picture, each row a pointer | ingest | ask: the plan starts here | `ingest.py` (atlas rows), `ask.py` (plan) |
| **Lens** | something already known, as a direction the question is pulled toward | a photo asked with the question | ask: before the search | `lens.py`, `ask.py` |
| **Note** | a checked sentence with its pointer and the question it answered | ask | the next similar question | `ask.py` (note) |

## Who does what

| Job | Part | File |
|---|---|---|
| Read the hard pages, write each answer once | **Gemini 3.8 Flash** (`google-genai`) | `gemini.py` |
| Find, in any language and medium | **EmbeddingGemma 2**, on your machine (`sentence-transformers`) | `embed.py` |
| Run the agents: sealed, typed, on a budget, side by side | **Antigravity SDK** (`google-antigravity`) | `scouts.py` |
| Decide: yes or no, or a score, with a probability | a **System 1** model ([TypeSafe](https://typesafe.ai) `typesafe-sdk`) | `system1.py` |

## Run it

You need Python 3.11+, [uv](https://docs.astral.sh/uv/), a [Gemini API key](https://aistudio.google.com/apikey) and a
TypeSafe API key. EmbeddingGemma 2 (about 740M parameters) downloads from Hugging Face on first use, so the first
upload is slower. It runs on a laptop GPU or a CPU.

```bash
cd atlas-template
cp .env.example .env          # fill in GEMINI_API_KEY and TYPESAFE_API_KEY
uv sync
uv run uvicorn atlas.app:app --port 8090
# open http://localhost:8090, add a PDF or a photo, then ask in any language
uv run pytest tests           # small checks of the pure parts
```

Each answer shows its sentences with citation chips (click one to see the page with its box lit up), any sentence
removed in checking with the reason, and a trace of the steps: plan, search, keep, write, check, note.

## The API, for your own UI

`web/index.html` is only one client. Any UI needs just these:

| Call | Send | Get back |
|---|---|---|
| `GET /api/items` | | `[{item_id, title, kind, pages}]` |
| `POST /api/items` | form field `file` (PDF or image) | `{item_id, title, pages, blocks, passages, atlas_rows, personal}`, or the item with `duplicate: true` |
| `POST /api/ask` | form fields `text`, optional `photo` (the lens) | the answer, below |
| `GET /pages/{image}` | | a page image, so a citation can show its box |

```jsonc
{
  "question": "…",
  "sentences": [{                       // checked sentences, in order
    "text": "…",
    "checked": "supported (system1, 0.97)",
    "citations": [{                     // pointers: draw the box on /pages/{image}
      "block_id": "a1b2c3#p2#b4", "item_id": "a1b2c3", "title": "…", "page": 2,
      "image": "a1b2c3-2.png", "w": 935, "h": 1210,   // page image size in pixels
      "x0": 80, "y0": 312, "x1": 860, "y1": 410       // the box, in those pixels
    }]
  }],
  "removed": [{"text": "…", "reason": "250 is not in the cited source"}],
  "trace": [{"step": "plan", "detail": "…", "s": 0.4}]   // plan, search, keep, write, check, note
}
```

To place a box on a page image shown at width `W`, scale every coordinate by `W / w`.

## The files

```
atlas/
  config.py   settings from the environment
  store.py    SQLite: items, pages, blocks (pointers), passages, atlas, notes, plus vectors searched in numpy
  embed.py    EmbeddingGemma 2: questions, passages and photos in one space
  gemini.py   Gemini, always asked for a typed object (a Pydantic schema)
  system1.py  typed decisions: System 1, then Gemini as a second opinion, then a safe default
  lens.py     steer(): pull the question toward a lens
  ingest.py   PHASE 1: check, read with place, organise, point + index
  scouts.py   sealed Antigravity scouts, all started at once
  ask.py      PHASE 2: plan, search, keep, write, check, note
  app.py      the API and the page
web/index.html  the whole frontend: library, ask, answer, citations, trace
tests/          checks of pointers, passages, search, the lens and the number check
```

## Deliberately left out

This template keeps only the shape of the architecture. A real system for an institution needs more, for example:
access control and sensitivity labels, a job queue so uploads run in the background, a real vector database,
audio and video (speech with timestamps, key frames), page agents that zoom into the pictures on text pages, a
planner agent that writes the plan, identity lenses built from tagged people, retries and monitoring, and an
evaluation set to measure answers. Every simplification in the code is marked with a `# template:` comment.

## Credits and license

Inspired by Andrej Karpathy's *LLM Wiki* (2026). Built on Gemini 3.8 Flash, EmbeddingGemma 2 and the Antigravity SDK
from Google, and a System 1 model from TypeSafe. Code under the [Apache License 2.0](LICENSE).

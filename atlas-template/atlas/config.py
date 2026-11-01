"""Settings, read from the environment (or a .env file beside pyproject.toml). See .env.example."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# a tiny .env reader, so the template needs no extra package for it
_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.split(" #")[0].strip())   # drop a trailing comment

DATA = Path(os.environ.get("ATLAS_DATA", ROOT / "data"))           # SQLite file, page images, uploaded files
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")    # reads hard pages, writes each answer once
TYPESAFE_API_KEY = os.environ.get("TYPESAFE_API_KEY", "")           # the System 1 decision model
TYPESAFE_MODEL = os.environ.get("TYPESAFE_MODEL", "jev-1.13.0")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "google/embeddinggemma-2")  # one index for every language and medium
THETA = float(os.environ.get("S1_THRESHOLD", "0.85"))              # trust System 1 at or above this confidence

for d in ("pages", "files", "agents"):
    (DATA / d).mkdir(parents=True, exist_ok=True)

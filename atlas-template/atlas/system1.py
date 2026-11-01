"""DECIDE: a small System 1 model that never writes text.

It answers a typed question (yes or no, or a score) with a probability, in a fraction of a second. Most decisions in
a knowledge system are like this: is this passage relevant, is this sentence supported, does this block name a person.

The cascade, in three lines:
  1. System 1 answers.  At or above THETA, we act on it.
  2. Below THETA, the same question goes to Gemini as a second opinion.
  3. Still unsure: the safe default for that decision (for an answer sentence: leave it out).
"""
import asyncio
from dataclasses import dataclass

from pydantic import BaseModel
from typesafe_sdk import AsyncTypeSafeClient, Noul, Score

from . import gemini
from .config import THETA, TYPESAFE_API_KEY, TYPESAFE_MODEL

LEVELS = ["irrelevant", "tangential", "related", "relevant", "directly answers"]

# One entry per kind of decision: its type, its question, and what to do when nobody is sure.
DECISIONS = {
    "supported": {"type": "yes/no", "safe": False, "escalate": True,
                  "q": "Is the answer sentence supported by its cited source text? Check the facts and every number."},
    "relevance": {"type": "score", "safe": None, "escalate": False,     # ranking is cheap to get slightly wrong
                  "q": "How well does the passage help answer the question?"},
    "personal":  {"type": "yes/no", "safe": True, "escalate": True,
                  "q": "Does this text name a private person, or give their personal details?"},
}

_client = None


def client() -> AsyncTypeSafeClient:
    global _client
    if _client is None:
        _client = AsyncTypeSafeClient(api_key=TYPESAFE_API_KEY, model=TYPESAFE_MODEL)
    return _client


@dataclass
class Decision:
    value: bool | float | None
    confidence: float
    by: str                     # "system1", "gemini" or "safe default"


class SecondOpinion(BaseModel):
    answer: bool
    confidence: float           # 0 to 1


async def decide(key: str, state: dict) -> Decision:
    spec = DECISIONS[key]
    question = Noul(instructions=spec["q"]) if spec["type"] == "yes/no" else Score(instructions=spec["q"], criteria=LEVELS)
    r = await client().system_one(state, {key: question})
    a = r.answers[key]
    if spec["type"] == "score":
        return Decision(float(a.score), float(a.confidence), "system1")
    p = float(a.noul)                                           # the probability of "yes"
    d = Decision(p >= 0.5, max(p, 1 - p), "system1")
    if d.confidence >= THETA or not spec["escalate"]:
        return d
    s = await gemini.typed([f"{spec['q']} Answer with a confidence from 0 to 1.\n\n{state}"], SecondOpinion)
    if s.confidence >= THETA:
        return Decision(s.answer, s.confidence, "gemini")
    return Decision(spec["safe"], s.confidence, "safe default")


async def decide_many(key: str, states: list[dict]) -> list[Decision]:
    """Many small decisions at once."""
    return list(await asyncio.gather(*(decide(key, s) for s in states)))

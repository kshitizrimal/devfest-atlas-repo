"""RUN THE AGENTS: scouts, each a sealed Antigravity agent, all started at once.

One scout = one job from the plan (one item, or the whole library). What makes it safe enough for an archive:
  1. tools    plain Python functions; the SDK reads their type hints and docstrings
  2. sealed   we deny everything, then allow only these tools by name; files and shell stay off
  3. typed    it must finish with a Report we can validate
  4. budget   code sets how many tool calls it may make, and a timeout

Why start them from our own code, and not with the SDK's built-in sub-agents? With built-in delegation the parent
waits for each one, so they run one after another. asyncio.gather runs them side by side.
"""
import asyncio
from typing import Literal

from google.antigravity import Agent, BuiltinTools, CapabilitiesConfig, LocalAgentConfig, models, types
from google.antigravity.hooks.policy import allow, deny
from pydantic import BaseModel

from . import embed, store
from .config import DATA, GEMINI_API_KEY, GEMINI_MODEL

AT_ONCE = 8            # scouts running at the same time
TOOL_CALLS = 4         # each scout's budget
TIMEOUT_S = 60

RULE = ("You are a scout searching an archive for passages that answer a question. Use search and read, then finish "
        "with a Report: the ids of the passages that answer, best first. Never invent an id. If nothing answers, say absent.")


class Report(BaseModel):
    verdict: Literal["answered", "partial", "absent"]
    passages: list[str] = []          # passage ids that answer, best first
    note: str = ""                    # one line: what was found, or what is missing


def tools_for(item_id: str | None):
    """Two tools, scoped to one item (or the whole library when item_id is None)."""

    def search(query: str) -> str:
        """Search for passages about the query, in any language. Returns passage ids with a short preview."""
        hits = store.search(["passage"], embed.query(query), k=6, item_id=item_id)
        return "\n".join(f"{pid}: {(store.passage(pid) or {}).get('text', '')[:160]}" for pid, *_ in hits) or "no passages"

    def read(passage_id: str) -> str:
        """Read one passage in full, with the page it is on."""
        p = store.passage(passage_id)
        return f"{p['header']}\n{p['text']}" if p else "no such passage"

    return [search, read]


def config(item_id: str | None) -> LocalAgentConfig:
    tools = tools_for(item_id)
    return LocalAgentConfig(
        model=types.ModelTarget(name=GEMINI_MODEL, endpoint=models.GeminiAPIEndpoint(
            api_key=GEMINI_API_KEY, options=models.GeminiModelOptions(thinking_level=models.ThinkingLevel("low")))),
        system_instructions=RULE,
        tools=tools,                                                                  # 1 tools
        capabilities=CapabilitiesConfig(enable_subagents=False, agent_behavior=types.AgentBehavior.MINIMAL,
                                        enabled_tools=[BuiltinTools.FINISH]),         # 2 sealed: no files, no shell
        policies=[deny("*"), allow("finish"), *[allow(t.__name__) for t in tools]],   # 2 deny all, then allow
        response_schema=Report,                                                       # 3 typed
        budget_config=types.BudgetConfig(max_tool_calls=TOOL_CALLS),                  # 4 budget
        workspaces=[str(DATA / "agents")],
    )


async def scout(job: dict, question: str, gate: asyncio.Semaphore) -> Report | None:
    """One sealed agent. It starts with the passages already found in its corner of the archive."""
    seeds = "\n".join(f"{pid}: {(store.passage(pid) or {}).get('text', '')[:200]}" for pid in job["seeds"])
    prompt = f"Question: {question}\nYour scope: {job['title']}\nPassages found so far:\n{seeds or '(none yet)'}"
    async with gate:
        try:
            async with asyncio.timeout(TIMEOUT_S), Agent(config(job["item_id"])) as agent:
                r = await agent.chat(prompt)
                async for _ in r.chunks:          # let the agent work through its tool calls
                    pass
                out = await r.structured_output()
            return out if isinstance(out, Report) else Report.model_validate(out)
        except Exception:                         # one scout failing never fails the question
            return None


async def run(jobs: list[dict], question: str) -> list[Report | None]:
    """Every scout at once, at most AT_ONCE running together."""
    gate = asyncio.Semaphore(AT_ONCE)
    return list(await asyncio.gather(*(scout(j, question, gate) for j in jobs)))

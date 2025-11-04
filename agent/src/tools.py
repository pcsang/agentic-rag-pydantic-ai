from pydantic import BaseModel, Field
from typing import List
from pydantic_ai import RunContext
from pydantic_ai.ag_ui import StateDeps
from ag_ui.core import EventType, StateSnapshotEvent


class ProverbsState(BaseModel):
    """List of the proverbs being written."""
    proverbs: List[str] = Field(
        default_factory=list,
        description='The list of already written proverbs',
    )


########## Plain tool functions (no agent decoration) ##########
def get_proverbs(ctx: RunContext[StateDeps["ProverbsState"]]) -> list[str]:
    """Get the current list of proverbs."""
    try:
        print(f"📖 Getting proverbs: {ctx.deps.state.proverbs}")
        return ctx.deps.state.proverbs
    except Exception:
        # defensive fallback
        return []


async def add_proverbs(ctx: RunContext[StateDeps["ProverbsState"]], proverbs: list[str]) -> StateSnapshotEvent:
    ctx.deps.state.proverbs.extend(proverbs)
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )


async def set_proverbs(ctx: RunContext[StateDeps["ProverbsState"]], proverbs: list[str]) -> StateSnapshotEvent:
    ctx.deps.state.proverbs = proverbs
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )


def get_weather(ctx: RunContext[StateDeps["ProverbsState"]], location: str) -> str:
    """Get the weather for a given location. Ensure location is fully spelled out."""
    return f"The weather in {location} is sunny."


def emit_state(ctx: RunContext[StateDeps["ProverbsState"]]) -> StateSnapshotEvent:
    """Emit the current agent state as a StateSnapshotEvent so frontends can render it."""
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )


__all__ = [
    "ProverbsState",
    "get_proverbs",
    "add_proverbs",
    "set_proverbs",
    "get_weather",
    "emit_state",
]

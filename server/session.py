# server/session.py
"""
Module-level session state for the Skill Graph MCP server.

Because stdio transport = 1 process = 1 session, a module-level singleton
is the correct scope for tracking rate limits and visited state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GET_SKILL_RATE_LIMIT: int = 10
ACTIVE_TOOL_CAP: int = 15


@dataclass
class SessionState:
    """Tracks per-session usage for the MCP server."""

    get_skill_calls: int = 0
    visited_nodes: set[str] = field(default_factory=set)
    active_tools: list[str] = field(default_factory=list)


_state: SessionState = SessionState()


def get_state() -> SessionState:
    """Return the current session state singleton."""
    return _state


def reset_state() -> None:
    """Reset state to defaults. Not yet called anywhere — tests/ is currently empty."""
    global _state
    _state = SessionState()
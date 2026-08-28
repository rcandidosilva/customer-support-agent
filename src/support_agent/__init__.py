"""Customer-support deflection with a confidence-gated escalation ladder."""

from .config import Settings, Thresholds
from .graph import build_ladder_graph
from .handoff_lint import lint_packet
from .ladder import Ladder, run_ticket
from .llm import AnthropicLLM, LLMError, ScriptedLLM
from .models import (
    Classification,
    ConfidenceReport,
    CritiqueReport,
    CustomerContext,
    DraftAnswer,
    HandoffPacket,
    Resolution,
    Ticket,
)
from .nodes import LadderDeps, LadderState
from .render import render_packet, render_resolution
from .retrieval import KnowledgeBase

__all__ = [
    "AnthropicLLM",
    "Classification",
    "ConfidenceReport",
    "CritiqueReport",
    "CustomerContext",
    "DraftAnswer",
    "HandoffPacket",
    "KnowledgeBase",
    "LLMError",
    "Ladder",
    "LadderDeps",
    "LadderState",
    "Resolution",
    "ScriptedLLM",
    "Settings",
    "Thresholds",
    "Ticket",
    "build_ladder_graph",
    "lint_packet",
    "render_packet",
    "render_resolution",
    "run_ticket",
]

__version__ = "0.1.0"

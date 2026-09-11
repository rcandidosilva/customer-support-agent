"""Customer-support deflection with a confidence-gated escalation ladder."""

from . import review_log
from .checkpointing import serializer, sqlite_saver
from .config import ReviewConfig, Settings, Thresholds
from .graph import build_ladder_graph
from .handoff_lint import lint_packet
from .ladder import Ladder, NoReviewPending, run_ticket
from .llm import AnthropicLLM, LLMError, ScriptedLLM, UnavailableLLM
from .models import (
    Classification,
    ConfidenceReport,
    CritiqueReport,
    CustomerContext,
    DraftAnswer,
    HandoffPacket,
    Resolution,
    ReviewRecord,
    ReviewRequest,
    ReviewVerdict,
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
    "NoReviewPending",
    "Resolution",
    "ReviewConfig",
    "ReviewRecord",
    "ReviewRequest",
    "ReviewVerdict",
    "ScriptedLLM",
    "Settings",
    "Thresholds",
    "Ticket",
    "UnavailableLLM",
    "build_ladder_graph",
    "lint_packet",
    "render_packet",
    "render_resolution",
    "review_log",
    "run_ticket",
    "serializer",
    "sqlite_saver",
]

__version__ = "0.1.0"

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import ClassVar


@dataclass
class Action:
    tool_name: str
    tool_input: dict
    result_stdout: str | None = None
    result_stderr: str | None = None
    success: bool = True


@dataclass
class Turn:
    user_prompt: str
    thinking: list[str] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    response_text: str = ""
    timestamp: str = ""
    cwd: str = ""


@dataclass
class Subtask:
    id: str
    session_id: str
    raw_description: str
    generalized_description: str
    turns: list[Turn]


@dataclass
class SubtaskIntelligence:
    reasoning_categories: dict[str, list[str]]  # analytical/planning/validation/reflection
    cognitive_patterns: list[str]
    outcome: str  # "clean_success" | "inefficient_success" | "recovery" | "failure"


@dataclass
class SubtaskAttribution:
    root_causes: list[str]
    contributing_factors: list[str]
    causal_chain: list[str]


@dataclass
class Tip:
    category: str  # "strategy" | "recovery" | "optimization"
    content: str
    purpose: str
    steps: list[str]
    trigger: str
    priority: str  # "critical" | "high" | "medium" | "low"
    source_session_id: str
    source_project: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    negative_example: str | None = None
    task_context: str | None = None
    subtask_id: str | None = None
    subtask_description: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    VALID_CATEGORIES: ClassVar[tuple[str, ...]] = ("strategy", "recovery", "optimization")
    VALID_PRIORITIES: ClassVar[tuple[str, ...]] = ("critical", "high", "medium", "low")

    def __post_init__(self) -> None:
        if self.category not in self.VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category: {self.category}. "
                f"Must be one of {self.VALID_CATEGORIES}"
            )
        if self.priority not in self.VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority: {self.priority}. "
                f"Must be one of {self.VALID_PRIORITIES}"
            )

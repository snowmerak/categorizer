"""Shared scores for choosing among sibling categories."""

from dataclasses import dataclass, field
from typing import Protocol


class CategoryInputTooLong(ValueError):
    """The selected scorer cannot accept the combined category input."""


@dataclass(frozen=True)
class SiblingScore:
    log_probability: float
    details: dict[str, float] = field(default_factory=dict)


class CategoryScorer(Protocol):
    model_name: str

    def score(self, query: str, documents: list[str]) -> list[SiblingScore]: ...

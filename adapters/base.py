"""Adapter abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Adapter(ABC):
    """Run a task variant through a system and return a RunResult."""

    @abstractmethod
    def run(self, task: dict[str, Any], variant_id: str, prompt: str) -> "RunResult":
        """
        Args:
            task: task dict loaded from YAML
            variant_id: "canonical" | "p1" | "p2" | ...
            prompt: the actual prompt text to send (canonical or paraphrase)

        Returns:
            RunResult — never raises; errors go in result.error
        """
        ...

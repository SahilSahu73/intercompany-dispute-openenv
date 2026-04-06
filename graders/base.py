"""Base grader interface for all task graders."""

from abc import ABC, abstractmethod

from domain.scenario_models import EpisodeContext


class BaseGrader(ABC):
    """Base class for task graders.

    Graders compute the terminal task score by comparing the
    episode's final state against the ground truth checklist.
    """

    @abstractmethod
    def score(self, ctx: EpisodeContext) -> float:
        """Compute terminal score in [0.0, 1.0]."""
        ...

    @abstractmethod
    def detailed_report(self, ctx: EpisodeContext) -> dict:
        """Return a detailed scoring breakdown for debugging."""
        ...

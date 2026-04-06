"""Task graders for the intercompany dispute environment."""

from .base import BaseGrader
from .easy_grader import EasyGrader
from .hard_grader import HardGrader
from .medium_grader import MediumGrader


def get_grader(difficulty: str) -> BaseGrader:
    """Return the appropriate grader for the given difficulty level.

    Args:
        difficulty: One of "easy", "medium", "hard".

    Returns:
        A BaseGrader subclass instance.

    Raises:
        ValueError: If difficulty is not recognized.
    """
    if difficulty == "easy":
        return EasyGrader()
    if difficulty == "medium":
        return MediumGrader()
    if difficulty == "hard":
        return HardGrader()
    raise ValueError(f"No grader implemented for difficulty: {difficulty!r}")


__all__ = ["BaseGrader", "EasyGrader", "MediumGrader", "HardGrader", "get_grader"]

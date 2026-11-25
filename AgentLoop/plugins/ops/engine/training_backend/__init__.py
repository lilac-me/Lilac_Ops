"""Public training hook; backend selection is a launch-time concern."""
from .hook import TrainingHook

__all__ = ["TrainingHook"]

"""
Schedules for curriculum data mixing.
Provides functions/classes that return the fraction of fresh data at a given step.
"""

from abc import ABC, abstractmethod


class Schedule(ABC):
    @abstractmethod
    def get_fresh_fraction(self, step: int, max_steps: int) -> float:
        """Return the fraction of fresh data at the given step (0.0 to 1.0)."""
        pass


class ConstantSchedule(Schedule):
    """A schedule that maintains a constant fraction of fresh data."""
    def __init__(self, fresh_fraction: float = 1.0):
        self.fresh_fraction = max(0.0, min(1.0, fresh_fraction))

    def get_fresh_fraction(self, step: int, max_steps: int) -> float:
        return self.fresh_fraction


class LinearDecaySchedule(Schedule):
    def __init__(self, start_fresh: float = 1.0, end_fresh: float = 0.0, end_step_ratio: float = 1.0):
        """
        Decay the fresh data fraction linearly.
        end_step_ratio: The fraction of max_steps at which the decay ends (e.g. 0.5 means it hits end_fresh halfway through).
        """
        self.start_fresh = max(0.0, min(1.0, start_fresh))
        self.end_fresh = max(0.0, min(1.0, end_fresh))
        self.end_step_ratio = max(0.0, min(1.0, end_step_ratio))

    def get_fresh_fraction(self, step: int, max_steps: int) -> float:
        if max_steps <= 1:
            return self.end_fresh

        end_step = int(max_steps * self.end_step_ratio)
        if step >= end_step:
            return self.end_fresh
        if end_step == 0:
            return self.end_fresh

        progress = step / end_step
        return self.start_fresh + progress * (self.end_fresh - self.start_fresh)


class StepPhaseOutSchedule(Schedule):
    def __init__(self, switch_step: int, before_fresh: float = 0.0, after_fresh: float = 1.0):
        """
        Switch from `before_fresh` fraction to `after_fresh` fraction exactly at `switch_step`.
        Default models a "rescue" scenario: starts fully collapsed (0.0 fresh), then switches to fully fresh (1.0).
        """
        self.switch_step = switch_step
        self.before_fresh = max(0.0, min(1.0, before_fresh))
        self.after_fresh = max(0.0, min(1.0, after_fresh))

    def get_fresh_fraction(self, step: int, max_steps: int) -> float:
        if step >= self.switch_step:
            return self.after_fresh
        return self.before_fresh

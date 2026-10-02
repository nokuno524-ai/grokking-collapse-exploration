from dataclasses import dataclass
from typing import List

@dataclass
class ScheduleConfig:
    """Configuration for a dynamic contamination schedule."""
    schedule_type: str = "fixed"  # 'fixed', 'ramp_up', 'ramp_down', 'block_interleaved', 'switch'
    collapse_level: float = 0.0
    collapse_severity: float = 0.5
    switch_step: int = 25000  # For 'switch' type
    switch_to: float = 0.15   # Collapse level to switch to
    block_size: int = 5000    # For 'block_interleaved' type
    ramp_steps: int = 10000   # For 'ramp_up' or 'ramp_down' (steps between phase updates)

@dataclass
class PhaseConfig:
    """Represents a discrete phase of training with fixed contamination parameters."""
    start_step: int
    end_step: int
    collapse_level: float
    collapse_severity: float

def build_schedule(config: ScheduleConfig, max_steps: int) -> List[PhaseConfig]:
    """
    Builds a list of training phases based on the schedule configuration.

    Args:
        config: The schedule configuration.
        max_steps: The maximum number of training steps.

    Returns:
        A list of PhaseConfig objects detailing the start step, end step,
        and contamination parameters for each phase.
    """
    phases = []

    if config.schedule_type == "fixed":
        phases.append(PhaseConfig(
            start_step=0,
            end_step=max_steps,
            collapse_level=config.collapse_level,
            collapse_severity=config.collapse_severity
        ))
    elif config.schedule_type == "switch":
        phases.append(PhaseConfig(
            start_step=0,
            end_step=config.switch_step,
            collapse_level=config.collapse_level,
            collapse_severity=config.collapse_severity
        ))
        phases.append(PhaseConfig(
            start_step=config.switch_step,
            end_step=max_steps,
            collapse_level=config.switch_to,
            collapse_severity=config.collapse_severity
        ))
    elif config.schedule_type == "block_interleaved":
        level_a = config.collapse_level
        level_b = config.switch_to
        for start in range(0, max_steps, config.block_size):
            end = min(start + config.block_size, max_steps)
            level = level_a if (start // config.block_size) % 2 == 0 else level_b
            phases.append(PhaseConfig(
                start_step=start,
                end_step=end,
                collapse_level=level,
                collapse_severity=config.collapse_severity
            ))
    elif config.schedule_type in ["ramp_up", "ramp_down"]:
        start_level = config.collapse_level
        end_level = config.switch_to
        # The ramp happens over `ramp_steps`. Then it holds at `end_level` until max_steps.
        # We discretize the ramp into 1000-step phases (or whatever makes sense)
        phase_duration = 1000
        num_ramp_phases = max(1, config.ramp_steps // phase_duration)
        level_step = (end_level - start_level) / num_ramp_phases

        current_step = 0
        for i in range(num_ramp_phases):
            start = current_step
            end = min(current_step + phase_duration, config.ramp_steps)
            if i == num_ramp_phases - 1:
                end = config.ramp_steps # ensure exactly matching the ramp boundary

            level = start_level + i * level_step
            phases.append(PhaseConfig(
                start_step=start,
                end_step=end,
                collapse_level=level,
                collapse_severity=config.collapse_severity
            ))
            current_step = end

        if current_step < max_steps:
            phases.append(PhaseConfig(
                start_step=current_step,
                end_step=max_steps,
                collapse_level=end_level,
                collapse_severity=config.collapse_severity
            ))
    else:
        raise ValueError(f"Unknown schedule_type: {config.schedule_type}")

    return phases

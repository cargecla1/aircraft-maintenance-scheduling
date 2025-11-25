"""
Cost function for Aircraft Maintenance Check Scheduling.

The objective is to minimize wasted intervals - the difference between
actual check intervals and the prescribed intervals.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

from data_loader import MaintenanceData, Config
from state import AircraftState, FleetState


@dataclass
class CheckCost:
    """Cost breakdown for a single check."""
    wasted_dy: float  # Wasted calendar days
    wasted_fh: float  # Wasted flight hours
    wasted_fc: float  # Wasted flight cycles
    penalty: float  # Additional penalties (capacity, etc.)

    @property
    def total(self) -> float:
        """Total cost."""
        return self.wasted_dy + self.wasted_fh + self.wasted_fc + self.penalty


def calculate_a_check_wasted_interval(
    state: AircraftState,
    config: Config,
) -> CheckCost:
    """
    Calculate the wasted interval cost for scheduling an A-check.
    
    Wasted interval = prescribed interval - actual usage
    """
    # Wasted = interval - current usage
    wasted_dy = max(0, state.a_ci_dy - state.dy_a)
    wasted_fh = max(0, state.a_ci_fh - state.fh_a)
    wasted_fc = max(0, state.a_ci_fc - state.fc_a)

    return CheckCost(
        wasted_dy=wasted_dy,
        wasted_fh=wasted_fh,
        wasted_fc=wasted_fc,
        penalty=0,
    )


def calculate_c_check_wasted_interval(
    state: AircraftState,
    config: Config,
) -> CheckCost:
    """
    Calculate the wasted interval cost for scheduling a C-check.
    
    C-check also includes the wasted A-check interval (merged).
    """
    # Wasted C-check interval
    wasted_c_dy = max(0, state.c_ci_dy - state.dy_c)
    wasted_c_fh = max(0, state.c_ci_fh - state.fh_c)
    wasted_c_fc = max(0, state.c_ci_fc - state.fc_c)

    # Wasted A-check interval (merged into C-check)
    wasted_a_dy = max(0, state.a_ci_dy - state.dy_a)
    wasted_a_fh = max(0, state.a_ci_fh - state.fh_a)
    wasted_a_fc = max(0, state.a_ci_fc - state.fc_a)

    return CheckCost(
        wasted_dy=wasted_c_dy + wasted_a_dy,
        wasted_fh=wasted_c_fh + wasted_a_fh,
        wasted_fc=wasted_c_fc + wasted_a_fc,
        penalty=0,
    )


def calculate_overdue_penalty(
    state: AircraftState,
    config: Config,
) -> float:
    """
    Calculate penalty for an aircraft being overdue on checks.
    
    Returns a large penalty (M_COST) if any check is overdue.
    """
    penalty = 0.0

    # Check if C-check is overdue
    if state.c_check_overdue():
        penalty += config.m_cost

    # Check if A-check is overdue
    if state.a_check_overdue():
        penalty += config.m_cost

    return penalty


def calculate_urgency_score(
    state: AircraftState,
    config: Config,
) -> float:
    """
    Calculate an urgency score for an aircraft.
    
    Higher score = more urgent need for maintenance.
    Used for prioritizing checks when capacity is limited.
    """
    # Calculate remaining intervals as fractions
    c_remaining_dy_frac = state.c_remaining_dy / state.c_ci_dy if state.c_ci_dy > 0 else 1
    c_remaining_fh_frac = state.c_remaining_fh / state.c_ci_fh if state.c_ci_fh > 0 else 1
    c_remaining_fc_frac = state.c_remaining_fc / state.c_ci_fc if state.c_ci_fc > 0 else 1

    a_remaining_dy_frac = state.a_remaining_dy / state.a_ci_dy if state.a_ci_dy > 0 else 1
    a_remaining_fh_frac = state.a_remaining_fh / state.a_ci_fh if state.a_ci_fh > 0 else 1
    a_remaining_fc_frac = state.a_remaining_fc / state.a_ci_fc if state.a_ci_fc > 0 else 1

    # Urgency increases as remaining fraction decreases
    c_urgency = max(0, 1 - min(c_remaining_dy_frac, c_remaining_fh_frac, c_remaining_fc_frac))
    a_urgency = max(0, 1 - min(a_remaining_dy_frac, a_remaining_fh_frac, a_remaining_fc_frac))

    # C-check has higher weight
    return 2 * c_urgency + a_urgency


def calculate_scheduling_benefit(
    state: AircraftState,
    check_type: str,  # 'A' or 'C'
    config: Config,
) -> float:
    """
    Calculate the benefit of scheduling a check now vs. later.
    
    Negative benefit = scheduling now wastes more interval than waiting.
    Positive benefit = scheduling now is more efficient.
    """
    if check_type == "A":
        # Benefit = avoiding tolerance usage - wasted interval
        wasted = calculate_a_check_wasted_interval(state, config)
        # If close to due, benefit is high
        if state.a_check_due():
            return 1000 - wasted.total
        elif state.dy_a >= state.a_ci_dy * 0.8:  # Within 80% of interval
            return 100 - wasted.total
        else:
            return -wasted.total

    elif check_type == "C":
        wasted = calculate_c_check_wasted_interval(state, config)
        # If close to due, benefit is high
        if state.c_check_due():
            return 10000 - wasted.total
        elif state.dy_c >= state.c_ci_dy * 0.8:  # Within 80% of interval
            return 1000 - wasted.total
        else:
            return -wasted.total

    return 0


def estimate_future_checks(
    state: AircraftState,
    days_horizon: int,
    daily_fh: float,
    daily_fc: float,
) -> tuple[int, int]:
    """
    Estimate the number of A-checks and C-checks needed over a time horizon.
    
    Returns:
        Tuple of (estimated_a_checks, estimated_c_checks)
    """
    # Estimate based on calendar days
    a_checks_by_dy = (days_horizon + state.dy_a) // state.a_ci_dy
    c_checks_by_dy = (days_horizon + state.dy_c) // state.c_ci_dy

    # Estimate based on flight hours
    total_fh = days_horizon * daily_fh + state.fh_a
    a_checks_by_fh = int(total_fh / state.a_ci_fh)
    total_fh_c = days_horizon * daily_fh + state.fh_c
    c_checks_by_fh = int(total_fh_c / state.c_ci_fh)

    # Estimate based on flight cycles
    total_fc = days_horizon * daily_fc + state.fc_a
    a_checks_by_fc = int(total_fc / state.a_ci_fc)
    total_fc_c = days_horizon * daily_fc + state.fc_c
    c_checks_by_fc = int(total_fc_c / state.c_ci_fc)

    # Take the maximum (most constraining)
    a_checks = max(a_checks_by_dy, a_checks_by_fh, a_checks_by_fc)
    c_checks = max(c_checks_by_dy, c_checks_by_fh, c_checks_by_fc)

    return a_checks, c_checks


def calculate_fleet_cost(
    fleet: FleetState,
    config: Config,
) -> float:
    """
    Calculate the total cost for the current fleet state.
    
    This includes overdue penalties but not scheduling costs.
    """
    total_cost = 0.0

    for tail, state in fleet.aircraft.items():
        total_cost += calculate_overdue_penalty(state, config)

    return total_cost


if __name__ == "__main__":
    from data_loader import load_maintenance_data
    from state import create_initial_fleet_state

    data = load_maintenance_data("Sep 25 2017 (Case Study 2018-2021).xlsx")
    fleet = create_initial_fleet_state(data)

    print("Cost analysis for initial fleet state:")

    for tail in ["Aircraft-1", "Aircraft-10", "Aircraft-20"]:
        state = fleet.aircraft[tail]
        print(f"\n{tail}:")

        a_cost = calculate_a_check_wasted_interval(state, data.config)
        print(f"  A-check wasted interval: DY={a_cost.wasted_dy}, FH={a_cost.wasted_fh:.1f}, FC={a_cost.wasted_fc:.1f}")

        c_cost = calculate_c_check_wasted_interval(state, data.config)
        print(f"  C-check wasted interval: DY={c_cost.wasted_dy}, FH={c_cost.wasted_fh:.1f}, FC={c_cost.wasted_fc:.1f}")

        urgency = calculate_urgency_score(state, data.config)
        print(f"  Urgency score: {urgency:.3f}")

        a_benefit = calculate_scheduling_benefit(state, "A", data.config)
        c_benefit = calculate_scheduling_benefit(state, "C", data.config)
        print(f"  A-check benefit: {a_benefit:.1f}")
        print(f"  C-check benefit: {c_benefit:.1f}")


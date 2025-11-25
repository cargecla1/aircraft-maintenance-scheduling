"""
Constraint checking for Aircraft Maintenance Check Scheduling.

Handles capacity constraints, holiday periods, peak seasons, and
minimum intervals between checks.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from data_loader import MaintenanceData, DateRange, PeakSeason, AdditionalSlots
from state import FleetState, AircraftState


@dataclass
class ConstraintViolation:
    """Details of a constraint violation."""
    constraint_type: str
    description: str
    severity: float  # Higher = more severe


def is_public_holiday(current_date: date, data: MaintenanceData) -> bool:
    """Check if the current date is a public holiday."""
    return current_date in data.public_holidays


def is_c_check_not_allowed(current_date: date, data: MaintenanceData) -> bool:
    """Check if C-checks are not allowed on this date (holidays, peak)."""
    # Check C_NOT_ALLOWED periods
    for period in data.c_not_allowed:
        if period.contains(current_date):
            return True
    return False


def is_c_check_peak_season(current_date: date, data: MaintenanceData) -> bool:
    """Check if the current date is in a C-check peak season."""
    for peak in data.c_peak:
        if peak.begin <= current_date <= peak.end:
            return True
    return False


def is_a_check_not_allowed(current_date: date, data: MaintenanceData) -> bool:
    """Check if A-checks are not allowed on this date."""
    return current_date in data.a_not_allowed


def get_additional_c_slots(current_date: date, data: MaintenanceData) -> int:
    """Get additional C-check slots available on this date."""
    for slot in data.more_c_slots:
        if slot.begin <= current_date <= slot.end:
            return slot.total_slots
    return 0


def get_additional_a_slots(current_date: date, data: MaintenanceData) -> int:
    """Get additional A-check slots available on this date."""
    return data.more_a_slots.get(current_date, 0)


def get_available_c_slots(
    fleet: FleetState,
    data: MaintenanceData,
    day_index: int,
) -> int:
    """
    Get the number of available C-check slots on a given day.
    
    Returns:
        Number of available slots (0 if no slots available)
    """
    current_date = fleet.current_date + timedelta(days=day_index - fleet.day_index)

    # Check if C-checks are allowed
    if is_c_check_not_allowed(current_date, data):
        # During not-allowed periods, check for additional slots
        additional = get_additional_c_slots(current_date, data)
        if additional == 0:
            return 0
        base_capacity = additional
    else:
        base_capacity = data.config.max_c_check
        # Add additional slots if available
        additional = get_additional_c_slots(current_date, data)
        base_capacity += additional

    # Subtract slots already in use
    in_use = fleet.get_c_slots_on_day(day_index)

    return max(0, base_capacity - in_use)


def get_available_a_slots(
    fleet: FleetState,
    data: MaintenanceData,
    day_index: int,
) -> int:
    """
    Get the number of available A-check slots on a given day.
    
    Returns:
        Number of available slots (0 if no slots available)
    """
    current_date = fleet.current_date + timedelta(days=day_index - fleet.day_index)

    # Check if A-checks are allowed
    if is_a_check_not_allowed(current_date, data):
        return 0

    base_capacity = data.config.max_a_check

    # Add additional slots if available
    additional = get_additional_a_slots(current_date, data)
    base_capacity += additional

    # Subtract slots already in use
    in_use = fleet.get_a_slots_on_day(day_index)

    return max(0, base_capacity - in_use)


def can_schedule_c_check(
    fleet: FleetState,
    data: MaintenanceData,
    tail: str,
    duration: int,
) -> tuple[bool, Optional[ConstraintViolation]]:
    """
    Check if a C-check can be scheduled for an aircraft.
    
    Checks:
    - Aircraft is not already grounded
    - C-check slots are available for the duration
    - Not too close to previous C-check start (START_DAY_INTERVAL)
    - Aircraft has not been checked too recently (DTC_MIN)
    
    Returns:
        Tuple of (can_schedule, violation_if_not)
    """
    state = fleet.aircraft[tail]

    # Check if aircraft is grounded
    if state.is_grounded:
        return False, ConstraintViolation(
            constraint_type="grounded",
            description=f"{tail} is already grounded for maintenance",
            severity=1.0,
        )

    # Check minimum days since last C-check
    if state.dy_c < data.config.dtc_min:
        return False, ConstraintViolation(
            constraint_type="dtc_min",
            description=f"{tail} C-check too recent ({state.dy_c} < {data.config.dtc_min} days)",
            severity=0.5,
        )

    # Check START_DAY_INTERVAL constraint
    for start_date in fleet.recent_c_starts:
        days_since = (fleet.current_date - start_date).days
        if days_since < data.config.start_day_interval:
            return False, ConstraintViolation(
                constraint_type="start_day_interval",
                description=f"Too close to previous C-check start ({days_since} < {data.config.start_day_interval} days)",
                severity=0.3,
            )

    # Check C-check slot availability for the duration
    for d in range(duration):
        day_idx = fleet.day_index + d
        available = get_available_c_slots(fleet, data, day_idx)
        if available <= 0:
            return False, ConstraintViolation(
                constraint_type="capacity",
                description=f"No C-check slots available on day {day_idx}",
                severity=0.8,
            )

    return True, None


def can_schedule_a_check(
    fleet: FleetState,
    data: MaintenanceData,
    tail: str,
) -> tuple[bool, Optional[ConstraintViolation]]:
    """
    Check if an A-check can be scheduled for an aircraft.
    
    Checks:
    - Aircraft is not already grounded
    - A-check slots are available
    - Aircraft has not been checked too recently (DTA_MIN)
    
    Returns:
        Tuple of (can_schedule, violation_if_not)
    """
    state = fleet.aircraft[tail]

    # Check if aircraft is grounded
    if state.is_grounded:
        return False, ConstraintViolation(
            constraint_type="grounded",
            description=f"{tail} is already grounded for maintenance",
            severity=1.0,
        )

    # Check minimum days since last A-check
    if state.dy_a < data.config.dta_min:
        return False, ConstraintViolation(
            constraint_type="dta_min",
            description=f"{tail} A-check too recent ({state.dy_a} < {data.config.dta_min} days)",
            severity=0.5,
        )

    # Check A-check slot availability
    available = get_available_a_slots(fleet, data, fleet.day_index)
    if available <= 0:
        return False, ConstraintViolation(
            constraint_type="capacity",
            description="No A-check slots available",
            severity=0.8,
        )

    return True, None


def must_schedule_c_check(
    state: AircraftState,
    data: MaintenanceData,
    lookahead_days: int = 0,
) -> bool:
    """
    Check if a C-check must be scheduled (aircraft will be overdue soon).
    
    Args:
        state: Aircraft state
        data: Maintenance data
        lookahead_days: Days to look ahead
    
    Returns:
        True if C-check is mandatory within lookahead period
    """
    # Check if already overdue or will be overdue soon
    remaining_dy = state.c_remaining_dy
    return remaining_dy <= lookahead_days


def must_schedule_a_check(
    state: AircraftState,
    data: MaintenanceData,
    lookahead_days: int = 0,
) -> bool:
    """
    Check if an A-check must be scheduled (aircraft will be overdue soon).
    
    Args:
        state: Aircraft state
        data: Maintenance data
        lookahead_days: Days to look ahead
    
    Returns:
        True if A-check is mandatory within lookahead period
    """
    # Check if already overdue or will be overdue soon
    remaining_dy = state.a_remaining_dy
    return remaining_dy <= lookahead_days


def get_feasible_c_check_aircraft(
    fleet: FleetState,
    data: MaintenanceData,
) -> list[str]:
    """
    Get list of aircraft that can have a C-check scheduled today.
    
    Returns:
        List of aircraft tail numbers
    """
    from dynamics import get_c_check_elapsed_time

    feasible = []
    for tail, state in fleet.aircraft.items():
        duration = get_c_check_elapsed_time(data, tail, 0)
        can_schedule, _ = can_schedule_c_check(fleet, data, tail, duration)
        if can_schedule:
            feasible.append(tail)
    return feasible


def get_feasible_a_check_aircraft(
    fleet: FleetState,
    data: MaintenanceData,
) -> list[str]:
    """
    Get list of aircraft that can have an A-check scheduled today.
    
    Returns:
        List of aircraft tail numbers
    """
    feasible = []
    for tail, state in fleet.aircraft.items():
        can_schedule, _ = can_schedule_a_check(fleet, data, tail)
        if can_schedule:
            feasible.append(tail)
    return feasible


def get_mandatory_c_check_aircraft(
    fleet: FleetState,
    data: MaintenanceData,
    lookahead_days: int = 7,
) -> list[str]:
    """
    Get list of aircraft that must have a C-check scheduled soon.
    
    Returns:
        List of aircraft tail numbers sorted by urgency
    """
    mandatory = []
    for tail, state in fleet.aircraft.items():
        if state.is_grounded:
            continue
        if must_schedule_c_check(state, data, lookahead_days):
            mandatory.append((tail, state.c_remaining_dy))

    # Sort by remaining days (most urgent first)
    mandatory.sort(key=lambda x: x[1])
    return [tail for tail, _ in mandatory]


def get_mandatory_a_check_aircraft(
    fleet: FleetState,
    data: MaintenanceData,
    lookahead_days: int = 3,
) -> list[str]:
    """
    Get list of aircraft that must have an A-check scheduled soon.
    
    Returns:
        List of aircraft tail numbers sorted by urgency
    """
    mandatory = []
    for tail, state in fleet.aircraft.items():
        if state.is_grounded:
            continue
        if must_schedule_a_check(state, data, lookahead_days):
            mandatory.append((tail, state.a_remaining_dy))

    # Sort by remaining days (most urgent first)
    mandatory.sort(key=lambda x: x[1])
    return [tail for tail, _ in mandatory]


def validate_fleet_state(
    fleet: FleetState,
    data: MaintenanceData,
) -> list[ConstraintViolation]:
    """
    Validate the current fleet state and return any violations.
    
    Returns:
        List of constraint violations
    """
    violations = []

    for tail, state in fleet.aircraft.items():
        # Check for overdue C-checks
        if state.c_check_overdue():
            violations.append(ConstraintViolation(
                constraint_type="overdue_c",
                description=f"{tail} C-check is overdue",
                severity=10.0,
            ))

        # Check for overdue A-checks
        if state.a_check_overdue():
            violations.append(ConstraintViolation(
                constraint_type="overdue_a",
                description=f"{tail} A-check is overdue",
                severity=5.0,
            ))

    return violations


if __name__ == "__main__":
    from data_loader import load_maintenance_data
    from state import create_initial_fleet_state

    data = load_maintenance_data("Sep 25 2017 (Case Study 2018-2021).xlsx")
    fleet = create_initial_fleet_state(data)

    print(f"Constraint analysis on {fleet.current_date}")
    print(f"\nC-check not allowed: {is_c_check_not_allowed(fleet.current_date, data)}")
    print(f"C-check peak season: {is_c_check_peak_season(fleet.current_date, data)}")
    print(f"A-check not allowed: {is_a_check_not_allowed(fleet.current_date, data)}")
    print(f"Public holiday: {is_public_holiday(fleet.current_date, data)}")

    print(f"\nAvailable C-check slots: {get_available_c_slots(fleet, data, fleet.day_index)}")
    print(f"Available A-check slots: {get_available_a_slots(fleet, data, fleet.day_index)}")

    print(f"\nFeasible C-check aircraft: {len(get_feasible_c_check_aircraft(fleet, data))}")
    print(f"Feasible A-check aircraft: {len(get_feasible_a_check_aircraft(fleet, data))}")

    print(f"\nMandatory C-checks (7 day lookahead): {get_mandatory_c_check_aircraft(fleet, data, 7)}")
    print(f"Mandatory A-checks (3 day lookahead): {get_mandatory_a_check_aircraft(fleet, data, 3)}")

    violations = validate_fleet_state(fleet, data)
    print(f"\nConstraint violations: {len(violations)}")
    for v in violations[:5]:
        print(f"  - {v.description}")


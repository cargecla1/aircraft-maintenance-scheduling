"""
State transition dynamics for Aircraft Maintenance Check Scheduling.

Handles aircraft aging (accumulating days, FH, FC) and state transitions
when maintenance checks are scheduled.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from data_loader import MaintenanceData, CCheckInfo
from state import AircraftState, FleetState


def get_daily_fh(data: MaintenanceData, tail: str, current_date: date) -> float:
    """Get daily flight hours for an aircraft based on month."""
    rates = data.flight_rates.get(tail)
    if rates is None:
        return 10.0  # Default
    month = current_date.month
    return rates.dfh.get(month, 10.0)


def get_daily_fc(data: MaintenanceData, tail: str, current_date: date) -> float:
    """Get daily flight cycles for an aircraft based on month."""
    rates = data.flight_rates.get(tail)
    if rates is None:
        return 4.0  # Default
    month = current_date.month
    return rates.dfc.get(month, 4.0)


def age_aircraft(
    state: AircraftState,
    data: MaintenanceData,
    current_date: date,
    is_grounded: bool = False,
) -> AircraftState:
    """
    Age an aircraft by one day.
    
    If grounded, only calendar days increment (DY).
    If operational, FH and FC also increment based on daily rates.
    """
    new_state = state.copy()

    # Check if aircraft is operational (after operation date)
    if state.operation_date and current_date < state.operation_date:
        return new_state

    # Increment calendar days
    new_state.dy_d += 1
    new_state.dy_c += 1
    new_state.dy_a += 1

    # If not grounded, increment flight hours and cycles
    if not is_grounded:
        dfh = get_daily_fh(data, state.tail, current_date)
        dfc = get_daily_fc(data, state.tail, current_date)

        new_state.fh_c += dfh
        new_state.fc_c += dfc
        new_state.fh_a += dfh
        new_state.fc_a += dfc

    return new_state


def get_c_check_elapsed_time(
    data: MaintenanceData,
    tail: str,
    c_check_index: int,
) -> int:
    """
    Get the elapsed time (TAT) for a C-check.
    
    Args:
        data: Maintenance data
        tail: Aircraft tail number
        c_check_index: Index of the C-check (0 = next, 1 = second next, etc.)
    
    Returns:
        Elapsed time in days
    """
    info = data.c_check_info.get(tail)
    if info and c_check_index < len(info.elapsed_times):
        return info.elapsed_times[c_check_index]
    return data.config.c_e_t  # Default elapsed time


def get_next_c_check_code(
    data: MaintenanceData,
    tail: str,
    c_check_index: int,
) -> Optional[float]:
    """Get the check code for the next C-check."""
    info = data.c_check_info.get(tail)
    if info and c_check_index < len(info.check_codes):
        return info.check_codes[c_check_index]
    return None


def perform_a_check(
    state: AircraftState,
    data: MaintenanceData,
    current_date: date,
) -> tuple[AircraftState, float]:
    """
    Perform an A-check on an aircraft.
    
    Returns:
        Tuple of (new_state, wasted_interval)
    """
    new_state = state.copy()

    # Calculate wasted interval (how much earlier than due the check was done)
    # Wasted = interval - actual usage
    wasted_dy = max(0, state.a_ci_dy - state.dy_a)
    wasted_fh = max(0, state.a_ci_fh - state.fh_a)
    wasted_fc = max(0, state.a_ci_fc - state.fc_a)

    # Calculate tolerance used (if check was done after interval)
    tolu_dy = max(0, state.dy_a - state.a_ci_dy)
    tolu_fh = max(0, state.fh_a - state.a_ci_fh)
    tolu_fc = max(0, state.fc_a - state.a_ci_fc)

    # Reset A-check counters
    new_state.dy_a = 0
    new_state.fh_a = 0
    new_state.fc_a = 0

    # Update tolerance used
    new_state.a_tolu_dy = tolu_dy
    new_state.a_tolu_fh = tolu_fh
    new_state.a_tolu_fc = tolu_fc

    # Update A-check serial number
    a_cycle = int(state.a_sn)
    a_minor = round((state.a_sn - a_cycle) * 100)
    a_minor += 1
    if a_minor >= state.a_max * 10:  # Reset cycle (handle decimal like 4.27)
        a_cycle += 1
        a_minor = 1
    new_state.a_sn = a_cycle + a_minor / 100
    new_state.a_cycle = a_cycle

    # Total wasted interval (normalized)
    wasted = wasted_dy + wasted_fh + wasted_fc

    return new_state, wasted


def perform_c_check(
    state: AircraftState,
    data: MaintenanceData,
    current_date: date,
    is_d_check: bool = False,
) -> tuple[AircraftState, float, int]:
    """
    Perform a C-check (or D-check) on an aircraft.
    
    A C-check also resets the A-check counters (A-check is merged).
    
    Returns:
        Tuple of (new_state, wasted_interval, elapsed_time)
    """
    new_state = state.copy()

    # Calculate wasted interval for C-check
    wasted_dy = max(0, state.c_ci_dy - state.dy_c)
    wasted_fh = max(0, state.c_ci_fh - state.fh_c)
    wasted_fc = max(0, state.c_ci_fc - state.fc_c)

    # Calculate wasted interval for A-check (merged)
    wasted_a_dy = max(0, state.a_ci_dy - state.dy_a)
    wasted_a_fh = max(0, state.a_ci_fh - state.fh_a)
    wasted_a_fc = max(0, state.a_ci_fc - state.fc_a)

    # Calculate tolerance used
    tolu_dy = max(0, state.dy_c - state.c_ci_dy)
    tolu_fh = max(0, state.fh_c - state.c_ci_fh)
    tolu_fc = max(0, state.fc_c - state.c_ci_fc)

    # Get elapsed time for this check
    elapsed_time = get_c_check_elapsed_time(data, state.tail, 0)
    if is_d_check:
        elapsed_time = max(elapsed_time, 21)  # D-checks typically longer

    # Reset C-check counters
    new_state.dy_c = 0
    new_state.fh_c = 0
    new_state.fc_c = 0

    # Reset A-check counters (A-check is merged into C-check)
    new_state.dy_a = 0
    new_state.fh_a = 0
    new_state.fc_a = 0

    # Update tolerance used
    new_state.c_tolu_dy = tolu_dy
    new_state.c_tolu_fh = tolu_fh
    new_state.c_tolu_fc = tolu_fc
    new_state.a_tolu_dy = 0
    new_state.a_tolu_fh = 0
    new_state.a_tolu_fc = 0

    # Update C-check serial number
    next_code = get_next_c_check_code(data, state.tail, 0)
    if next_code:
        new_state.c_sn = next_code
        new_state.c_cycle = int(next_code)
    else:
        c_cycle = int(state.c_sn)
        c_minor = round((state.c_sn - c_cycle) * 10)
        c_minor += 1
        if c_minor > state.c_max:
            c_cycle += 1
            c_minor = 1
        new_state.c_sn = c_cycle + c_minor / 10
        new_state.c_cycle = c_cycle

    # Update D-check state
    if is_d_check:
        new_state.d_cycle = 0
        new_state.dy_d = 0
        new_state.d_tolu_dy = 0
    else:
        new_state.d_cycle += 1

    # Update A-check serial number (reset after C-check)
    a_cycle = int(state.a_sn)
    new_state.a_sn = a_cycle + 0.01
    new_state.a_cycle = a_cycle

    # Total wasted interval
    wasted = (wasted_dy + wasted_fh + wasted_fc +
              wasted_a_dy + wasted_a_fh + wasted_a_fc)

    return new_state, wasted, elapsed_time


def advance_fleet_one_day(
    fleet: FleetState,
    data: MaintenanceData,
) -> FleetState:
    """
    Advance the fleet state by one day.
    
    - Ages all non-grounded aircraft
    - Updates grounding status for aircraft in maintenance
    - Cleans up completed checks
    """
    new_fleet = fleet.copy()
    new_date = fleet.current_date + timedelta(days=1)
    new_fleet.current_date = new_date
    new_fleet.day_index += 1

    # Update each aircraft
    for tail, state in new_fleet.aircraft.items():
        is_grounded = state.is_grounded

        # Check if maintenance is complete
        if state.is_grounded and state.grounded_until:
            if new_date > state.grounded_until:
                state.is_grounded = False
                state.grounded_until = None
                state.check_type_in_progress = None
                is_grounded = False

        # Age the aircraft
        new_fleet.aircraft[tail] = age_aircraft(state, data, new_date, is_grounded)
        new_fleet.aircraft[tail].is_grounded = is_grounded
        new_fleet.aircraft[tail].grounded_until = state.grounded_until
        new_fleet.aircraft[tail].check_type_in_progress = state.check_type_in_progress

    # Clean up old C-check starts (keep only last START_DAY_INTERVAL days)
    interval = data.config.start_day_interval
    new_fleet.recent_c_starts = [
        d for d in fleet.recent_c_starts
        if (new_date - d).days <= interval
    ]

    return new_fleet


def schedule_a_check(
    fleet: FleetState,
    data: MaintenanceData,
    tail: str,
) -> tuple[FleetState, float]:
    """
    Schedule an A-check for an aircraft.
    
    A-checks are assumed to complete in 1 day (overnight).
    
    Returns:
        Tuple of (new_fleet_state, wasted_interval)
    """
    new_fleet = fleet.copy()
    state = new_fleet.aircraft[tail]

    # Perform the A-check
    new_state, wasted = perform_a_check(state, data, fleet.current_date)
    new_fleet.aircraft[tail] = new_state

    # Record the scheduled check
    new_fleet.scheduled_a_checks.append((tail, fleet.current_date))

    # Update A-check slot usage
    day_idx = fleet.day_index
    new_fleet.a_slots_in_use[day_idx] = new_fleet.a_slots_in_use.get(day_idx, 0) + 1

    return new_fleet, wasted


def schedule_c_check(
    fleet: FleetState,
    data: MaintenanceData,
    tail: str,
) -> tuple[FleetState, float, int]:
    """
    Schedule a C-check for an aircraft.
    
    C-checks ground the aircraft for several days.
    
    Returns:
        Tuple of (new_fleet_state, wasted_interval, elapsed_time)
    """
    new_fleet = fleet.copy()
    state = new_fleet.aircraft[tail]

    # Check if this is also a D-check
    is_d_check = state.next_c_check_is_d()

    # Perform the C-check
    new_state, wasted, elapsed_time = perform_c_check(
        state, data, fleet.current_date, is_d_check
    )

    # Ground the aircraft
    new_state.is_grounded = True
    new_state.grounded_until = fleet.current_date + timedelta(days=elapsed_time)
    new_state.check_type_in_progress = "D" if is_d_check else "C"

    new_fleet.aircraft[tail] = new_state

    # Record the scheduled check
    new_fleet.scheduled_c_checks.append((tail, fleet.current_date, elapsed_time))

    # Update C-check slot usage for duration
    for d in range(elapsed_time):
        day_idx = fleet.day_index + d
        new_fleet.c_slots_in_use[day_idx] = new_fleet.c_slots_in_use.get(day_idx, 0) + 1

    # Record C-check start date
    new_fleet.recent_c_starts.append(fleet.current_date)

    return new_fleet, wasted, elapsed_time


if __name__ == "__main__":
    from data_loader import load_maintenance_data
    from state import create_initial_fleet_state

    data = load_maintenance_data("Sep 25 2017 (Case Study 2018-2021).xlsx")
    fleet = create_initial_fleet_state(data)

    print(f"Initial fleet state on {fleet.current_date}")

    # Test aging
    print("\nAging Aircraft-1 by 30 days...")
    ac = fleet.aircraft["Aircraft-1"]
    for _ in range(30):
        ac = age_aircraft(ac, data, fleet.current_date + timedelta(days=_))
    print(f"  C-check: DY={ac.dy_c}, FH={ac.fh_c:.1f}, FC={ac.fc_c:.1f}")
    print(f"  A-check: DY={ac.dy_a}, FH={ac.fh_a:.1f}, FC={ac.fc_a:.1f}")

    # Test A-check
    print("\nPerforming A-check on Aircraft-1...")
    new_ac, wasted = perform_a_check(ac, data, fleet.current_date)
    print(f"  Wasted interval: {wasted:.1f}")
    print(f"  New A-check state: DY={new_ac.dy_a}, FH={new_ac.fh_a:.1f}, FC={new_ac.fc_a:.1f}")

    # Test fleet advance
    print("\nAdvancing fleet by 1 day...")
    new_fleet = advance_fleet_one_day(fleet, data)
    print(f"  New date: {new_fleet.current_date}")
    ac2 = new_fleet.aircraft["Aircraft-1"]
    print(f"  Aircraft-1 A-check: DY={ac2.dy_a}, FH={ac2.fh_a:.1f}")


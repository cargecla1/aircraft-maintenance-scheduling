"""
Core module for Aircraft Maintenance Check Scheduling.

This module combines:
- State representation (AircraftState, FleetState)
- State transition dynamics (aging, check execution)
- Constraint checking (capacity, holidays, intervals)
- Cost functions (wasted intervals, urgency scoring)
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional
import copy

from .data_loader import (
    MaintenanceData,
    AircraftDInitial,
    AircraftCInitial,
    AircraftAInitial,
    FlightRates,
    CCheckInfo,
    Config,
    DateRange,
    PeakSeason,
    AdditionalSlots,
)


# =============================================================================
# STATE REPRESENTATION
# =============================================================================


@dataclass
class AircraftState:
    """State of a single aircraft."""
    tail: str
    fleet: str

    # D-check state
    d_cycle: int  # Current cycle in D-check
    dy_d: int  # Days since last D-check
    d_tolu_dy: int  # Tolerance used in previous D-check

    # C-check state
    c_sn: float  # Check serial number (e.g., 11.1)
    c_cycle: int  # Derived: which C-check in the cycle
    dy_c: int  # Days since last C-check
    fh_c: float  # Flight hours since last C-check
    fc_c: float  # Flight cycles since last C-check
    c_tolu_dy: int  # Tolerance used (days)
    c_tolu_fh: float  # Tolerance used (flight hours)
    c_tolu_fc: float  # Tolerance used (flight cycles)

    # A-check state
    a_sn: float  # Check serial number (e.g., 4.27)
    a_cycle: int  # Derived: which A-check in the cycle
    dy_a: int  # Days since last A-check
    fh_a: float  # Flight hours since last A-check
    fc_a: float  # Flight cycles since last A-check
    a_tolu_dy: int  # Tolerance used (days)
    a_tolu_fh: float  # Tolerance used (flight hours)
    a_tolu_fc: float  # Tolerance used (flight cycles)

    # Check intervals and tolerances (from aircraft type)
    c_ci_dy: int  # C-check interval (days)
    c_ci_fh: float  # C-check interval (flight hours)
    c_ci_fc: float  # C-check interval (flight cycles)
    c_tol_dy: int  # C-check tolerance allowed (days)
    c_tol_fh: float  # C-check tolerance allowed (flight hours)
    c_tol_fc: float  # C-check tolerance allowed (flight cycles)

    a_ci_dy: int  # A-check interval (days)
    a_ci_fh: float  # A-check interval (flight hours)
    a_ci_fc: float  # A-check interval (flight cycles)
    a_tol_dy: int  # A-check tolerance allowed (days)
    a_tol_fh: float  # A-check tolerance allowed (flight hours)
    a_tol_fc: float  # A-check tolerance allowed (flight cycles)

    d_ci_dy: int  # D-check interval (days)
    d_tol_dy: int  # D-check tolerance allowed (days)
    d_max: int  # Max C-check cycles before D-check

    c_max: int  # Total C-checks in a cycle
    a_max: int  # Total A-checks in a cycle

    # Grounding state (during maintenance)
    is_grounded: bool = False
    grounded_until: Optional[date] = None
    check_type_in_progress: Optional[str] = None  # 'A', 'C', or 'D'

    # Operation start date
    operation_date: Optional[date] = None

    def copy(self) -> "AircraftState":
        """Create a deep copy of this state."""
        return copy.deepcopy(self)

    @property
    def c_remaining_dy(self) -> int:
        """Remaining days until C-check is due (including tolerance)."""
        return self.c_ci_dy + self.c_tol_dy - self.c_tolu_dy - self.dy_c

    @property
    def c_remaining_fh(self) -> float:
        """Remaining flight hours until C-check is due (including tolerance)."""
        return self.c_ci_fh + self.c_tol_fh - self.c_tolu_fh - self.fh_c

    @property
    def c_remaining_fc(self) -> float:
        """Remaining flight cycles until C-check is due (including tolerance)."""
        return self.c_ci_fc + self.c_tol_fc - self.c_tolu_fc - self.fc_c

    @property
    def a_remaining_dy(self) -> int:
        """Remaining days until A-check is due (including tolerance)."""
        return self.a_ci_dy + self.a_tol_dy - self.a_tolu_dy - self.dy_a

    @property
    def a_remaining_fh(self) -> float:
        """Remaining flight hours until A-check is due (including tolerance)."""
        return self.a_ci_fh + self.a_tol_fh - self.a_tolu_fh - self.fh_a

    @property
    def a_remaining_fc(self) -> float:
        """Remaining flight cycles until A-check is due (including tolerance)."""
        return self.a_ci_fc + self.a_tol_fc - self.a_tolu_fc - self.fc_a

    @property
    def d_remaining_dy(self) -> int:
        """Remaining days until D-check is due (including tolerance)."""
        return self.d_ci_dy + self.d_tol_dy - self.d_tolu_dy - self.dy_d

    def c_check_due(self) -> bool:
        """Check if C-check is due (any metric exceeds interval)."""
        return (
            self.dy_c >= self.c_ci_dy
            or self.fh_c >= self.c_ci_fh
            or self.fc_c >= self.c_ci_fc
        )

    def c_check_overdue(self) -> bool:
        """Check if C-check is overdue (exceeds interval + tolerance)."""
        return (
            self.dy_c > self.c_ci_dy + self.c_tol_dy - self.c_tolu_dy
            or self.fh_c > self.c_ci_fh + self.c_tol_fh - self.c_tolu_fh
            or self.fc_c > self.c_ci_fc + self.c_tol_fc - self.c_tolu_fc
        )

    def a_check_due(self) -> bool:
        """Check if A-check is due (any metric exceeds interval)."""
        return (
            self.dy_a >= self.a_ci_dy
            or self.fh_a >= self.a_ci_fh
            or self.fc_a >= self.a_ci_fc
        )

    def a_check_overdue(self) -> bool:
        """Check if A-check is overdue (exceeds interval + tolerance)."""
        return (
            self.dy_a > self.a_ci_dy + self.a_tol_dy - self.a_tolu_dy
            or self.fh_a > self.a_ci_fh + self.a_tol_fh - self.a_tolu_fh
            or self.fc_a > self.a_ci_fc + self.a_tol_fc - self.a_tolu_fc
        )

    def d_check_due(self) -> bool:
        """Check if D-check is due (at the last C-check of the D-cycle)."""
        return self.d_cycle >= self.d_max - 1

    def next_c_check_is_d(self) -> bool:
        """Check if the next C-check will be a D-check."""
        return self.d_cycle >= self.d_max - 1

    def tolerance_signature(self) -> tuple:
        """
        Return a signature based on tolerance used.
        Used for state aggregation in DP.
        """
        return (
            self.c_tolu_dy,
            round(self.c_tolu_fh, 1),
            round(self.c_tolu_fc, 1),
            self.a_tolu_dy,
            round(self.a_tolu_fh, 1),
            round(self.a_tolu_fc, 1),
        )


@dataclass
class FleetState:
    """State of the entire fleet."""
    aircraft: dict[str, AircraftState]  # tail -> state
    current_date: date
    day_index: int  # Days since planning start

    # Track scheduled checks
    scheduled_c_checks: list[tuple[str, date, int]]  # (tail, start_date, duration)
    scheduled_a_checks: list[tuple[str, date]]  # (tail, date)

    # Track C-check slots in use on each day
    c_slots_in_use: dict[int, int]  # day_index -> count
    a_slots_in_use: dict[int, int]  # day_index -> count

    # Recent C-check start dates (for START_DAY_INTERVAL constraint)
    recent_c_starts: list[date]

    def copy(self) -> "FleetState":
        """Create a deep copy of this state."""
        return FleetState(
            aircraft={tail: state.copy() for tail, state in self.aircraft.items()},
            current_date=self.current_date,
            day_index=self.day_index,
            scheduled_c_checks=list(self.scheduled_c_checks),
            scheduled_a_checks=list(self.scheduled_a_checks),
            c_slots_in_use=dict(self.c_slots_in_use),
            a_slots_in_use=dict(self.a_slots_in_use),
            recent_c_starts=list(self.recent_c_starts),
        )

    def get_c_slots_on_day(self, day_idx: int) -> int:
        """Get number of C-check slots in use on a given day."""
        return self.c_slots_in_use.get(day_idx, 0)

    def get_a_slots_on_day(self, day_idx: int) -> int:
        """Get number of A-check slots in use on a given day."""
        return self.a_slots_in_use.get(day_idx, 0)


def create_initial_fleet_state(data: MaintenanceData) -> FleetState:
    """Create the initial fleet state from loaded maintenance data."""
    aircraft = {}

    for tail in data.aircraft_tails:
        d_init = data.d_initial[tail]
        c_init = data.c_initial[tail]
        a_init = data.a_initial[tail]

        # Parse check serial numbers to get cycle position
        c_sn_parts = str(c_init.c_sn).split(".")
        c_cycle = int(c_sn_parts[0]) if c_sn_parts else 0

        a_sn_parts = str(a_init.a_sn).split(".")
        a_cycle = int(a_sn_parts[0]) if a_sn_parts else 0

        aircraft[tail] = AircraftState(
            tail=tail,
            fleet=d_init.fleet,
            # D-check state
            d_cycle=d_init.d_cycle,
            dy_d=d_init.dy_d,
            d_tolu_dy=d_init.d_tolu_dy,
            # C-check state
            c_sn=c_init.c_sn,
            c_cycle=c_cycle,
            dy_c=c_init.dy_c,
            fh_c=c_init.fh_c,
            fc_c=c_init.fc_c,
            c_tolu_dy=c_init.c_tolu_dy,
            c_tolu_fh=c_init.c_tolu_fh,
            c_tolu_fc=c_init.c_tolu_fc,
            # A-check state
            a_sn=a_init.a_sn,
            a_cycle=a_cycle,
            dy_a=a_init.dy_a,
            fh_a=a_init.fh_a,
            fc_a=a_init.fc_a,
            a_tolu_dy=a_init.a_tolu_dy,
            a_tolu_fh=a_init.a_tolu_fh,
            a_tolu_fc=a_init.a_tolu_fc,
            # Check intervals
            c_ci_dy=c_init.c_ci_dy,
            c_ci_fh=c_init.c_ci_fh,
            c_ci_fc=c_init.c_ci_fc,
            c_tol_dy=c_init.c_tol_dy,
            c_tol_fh=c_init.c_tol_fh,
            c_tol_fc=c_init.c_tol_fc,
            a_ci_dy=a_init.a_ci_dy,
            a_ci_fh=a_init.a_ci_fh,
            a_ci_fc=a_init.a_ci_fc,
            a_tol_dy=a_init.a_tol_dy,
            a_tol_fh=a_init.a_tol_fh,
            a_tol_fc=a_init.a_tol_fc,
            d_ci_dy=d_init.d_ci_dy,
            d_tol_dy=d_init.d_tol_dy,
            d_max=d_init.d_max,
            c_max=c_init.c_max,
            a_max=a_init.a_max,
            # Operation date
            operation_date=d_init.operation,
        )

    return FleetState(
        aircraft=aircraft,
        current_date=data.config.begin_day,
        day_index=0,
        scheduled_c_checks=[],
        scheduled_a_checks=[],
        c_slots_in_use={},
        a_slots_in_use={},
        recent_c_starts=[],
    )


# =============================================================================
# DYNAMICS (State Transitions)
# =============================================================================


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


# =============================================================================
# CONSTRAINTS
# =============================================================================


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


# =============================================================================
# COST FUNCTIONS
# =============================================================================


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


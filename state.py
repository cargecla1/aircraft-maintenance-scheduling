"""
State representation for Aircraft Maintenance Check Scheduling.

Each aircraft state tracks days, flight hours, and flight cycles since
the last A-check and C-check, along with check serial numbers and tolerances.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional
import copy

from data_loader import (
    MaintenanceData,
    AircraftDInitial,
    AircraftCInitial,
    AircraftAInitial,
    FlightRates,
    CCheckInfo,
)


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


if __name__ == "__main__":
    from data_loader import load_maintenance_data

    data = load_maintenance_data("Sep 25 2017 (Case Study 2018-2021).xlsx")
    fleet = create_initial_fleet_state(data)

    print(f"Fleet state on {fleet.current_date}")
    print(f"Number of aircraft: {len(fleet.aircraft)}")

    # Show first aircraft state
    ac = fleet.aircraft["Aircraft-1"]
    print(f"\nAircraft-1 state:")
    print(f"  C-check: DY={ac.dy_c}, FH={ac.fh_c:.1f}, FC={ac.fc_c:.1f}")
    print(f"  C remaining: DY={ac.c_remaining_dy}, FH={ac.c_remaining_fh:.1f}, FC={ac.c_remaining_fc:.1f}")
    print(f"  C-check due: {ac.c_check_due()}")
    print(f"  A-check: DY={ac.dy_a}, FH={ac.fh_a:.1f}, FC={ac.fc_a:.1f}")
    print(f"  A remaining: DY={ac.a_remaining_dy}, FH={ac.a_remaining_fh:.1f}, FC={ac.a_remaining_fc:.1f}")
    print(f"  A-check due: {ac.a_check_due()}")


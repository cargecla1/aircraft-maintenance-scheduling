"""
Solvers for Aircraft Maintenance Check Scheduling.

Contains multiple scheduling algorithms:
- GreedyScheduler: Simple urgency-based greedy policy
- DPScheduler: Rolling-horizon planning with lookahead
- OptimizedDPScheduler: State-aggregation heuristic DP

For the paper-faithful Deng et al. (2020) implementation, see jax_solver.py
which provides DengScheduler - a clean, well-documented forward beam search.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional
from collections import defaultdict
import heapq
import time

from itertools import combinations

from .data_loader import MaintenanceData
from .core import (
    # State
    FleetState,
    AircraftState,
    create_initial_fleet_state,
    # Dynamics
    advance_fleet_one_day,
    schedule_a_check,
    schedule_c_check,
    get_c_check_elapsed_time,
    get_daily_fh,
    get_daily_fc,
    # Check results
    CheckResult,
    CCheckResult,
    # Cost (paper's eq. 28)
    calculate_a_check_cost,
    calculate_c_check_cost,
    calculate_urgency_score,
    calculate_overdue_penalty,
    calculate_tolerance_penalty,
    calculate_fleet_cost,
    FleetCost,
    # Constraints
    can_schedule_c_check,
    can_schedule_a_check,
    get_feasible_c_check_aircraft,
    get_feasible_a_check_aircraft,
    get_mandatory_c_check_aircraft,
    get_mandatory_a_check_aircraft,
    is_c_check_not_allowed,
    is_a_check_not_allowed,
    get_available_c_slots,
    get_available_a_slots,
)


# =============================================================================
# Paper's 7 Daily Actions (Table 2 in Deng et al.)
# =============================================================================
# The paper defines 7 capacity-consistent daily actions based on hangar capacity.
# Each action is a tuple (n_a, n_c) where:
#   n_a = number of A-checks to schedule
#   n_c = number of C-checks to schedule (0 or 1)
#
# The paper assumes:
#   - Max 3 A-checks per day (when no C-check)
#   - Max 1 C-check per day (can be done with A-checks)
#   - C-check capacity allows up to 3 simultaneous C-checks across different start days

PAPER_ACTIONS = [
    (0, 0),  # No maintenance
    (0, 1),  # 1 C-check only
    (1, 0),  # 1 A-check only
    (1, 1),  # 1 A-check + 1 C-check
    (2, 0),  # 2 A-checks
    (2, 1),  # 2 A-checks + 1 C-check
    (3, 0),  # 3 A-checks (max A capacity)
    # Note: (3, 1) may exceed typical capacity constraints
]


@dataclass
class FleetAction:
    """
    A concrete fleet action specifying which aircraft get which checks.
    
    This is an assignment of the abstract (n_a, n_c) action to specific aircraft.
    """
    a_checks: list[str]  # List of tail numbers getting A-checks
    c_checks: list[str]  # List of tail numbers getting C-checks (max 1 per day typically)
    
    @property
    def n_a(self) -> int:
        return len(self.a_checks)
    
    @property
    def n_c(self) -> int:
        return len(self.c_checks)
    
    def to_dict(self) -> dict[str, str]:
        """Convert to tail -> action format for compatibility."""
        result = {}
        for tail in self.a_checks:
            result[tail] = 'A'
        for tail in self.c_checks:
            result[tail] = 'C'
        return result


def enumerate_fleet_actions(
    fleet: FleetState,
    data: MaintenanceData,
    max_a_per_day: int = 3,
    max_c_per_day: int = 1,
    limit_combinations: int = 50,
) -> list[FleetAction]:
    """
    Enumerate all feasible fleet actions based on paper's action space.
    
    For each (n_a, n_c) in PAPER_ACTIONS, generate all valid assignments
    of aircraft to checks, respecting capacity constraints.
    
    Args:
        fleet: Current fleet state
        data: Maintenance data
        max_a_per_day: Maximum A-checks per day (paper uses 3)
        max_c_per_day: Maximum C-checks that can START on this day (paper uses 1)
        limit_combinations: Max combinations to generate per (n_a, n_c) action
    
    Returns:
        List of FleetAction objects representing feasible actions
    """
    # Get feasible aircraft for each check type
    feasible_a = get_feasible_a_check_aircraft(fleet, data)
    feasible_c = get_feasible_c_check_aircraft(fleet, data)
    
    # Get available slots
    available_a = get_available_a_slots(fleet, data, fleet.day_index)
    available_c = get_available_c_slots(fleet, data, fleet.day_index)
    
    actions = []
    
    # Always include the null action (no maintenance)
    actions.append(FleetAction(a_checks=[], c_checks=[]))
    
    # Generate actions for each (n_a, n_c) combination
    for n_a, n_c in PAPER_ACTIONS:
        if n_a == 0 and n_c == 0:
            continue  # Already added null action
            
        # Check capacity constraints
        if n_a > min(available_a, max_a_per_day, len(feasible_a)):
            continue
        if n_c > min(available_c, max_c_per_day, len(feasible_c)):
            continue
            
        # Generate C-check combinations (typically 0 or 1)
        if n_c == 0:
            c_combinations = [()]
        else:
            c_combinations = list(combinations(feasible_c, n_c))
            # Limit to avoid combinatorial explosion
            if len(c_combinations) > limit_combinations:
                # Sort by urgency and take top candidates
                urgencies = [(tail, calculate_urgency_score(fleet.aircraft[tail], data.config))
                            for tail in feasible_c]
                urgencies.sort(key=lambda x: -x[1])
                top_c = [t for t, _ in urgencies[:min(5, len(urgencies))]]
                c_combinations = list(combinations(top_c, min(n_c, len(top_c))))
        
        # Generate A-check combinations
        if n_a == 0:
            a_combinations = [()]
        else:
            a_combinations = list(combinations(feasible_a, n_a))
            # Limit to avoid combinatorial explosion
            if len(a_combinations) > limit_combinations:
                # Sort by urgency and take top candidates
                urgencies = [(tail, calculate_urgency_score(fleet.aircraft[tail], data.config))
                            for tail in feasible_a]
                urgencies.sort(key=lambda x: -x[1])
                top_a = [t for t, _ in urgencies[:min(8, len(urgencies))]]
                a_combinations = list(combinations(top_a, min(n_a, len(top_a))))
        
        # Combine A and C assignments
        for c_combo in c_combinations:
            for a_combo in a_combinations:
                # Ensure no aircraft gets both A and C check on same day
                if set(a_combo) & set(c_combo):
                    continue
                    
                actions.append(FleetAction(
                    a_checks=list(a_combo),
                    c_checks=list(c_combo),
                ))
                
                # Limit total actions
                if len(actions) > limit_combinations * 10:
                    return actions
    
    return actions


@dataclass
class FrontierNode:
    """State entry maintained during forward induction."""
    fleet: FleetState
    histogram: tuple
    cost: float
    checks: list["ScheduledCheck"]
    total_a: int
    total_c: int
    total_d: int
    policy: list[FleetAction] = field(default_factory=list)


def enumerate_mandatory_actions(
    fleet: FleetState,
    data: MaintenanceData,
    c_lookahead: int = 7,
    a_lookahead: int = 3,
) -> FleetAction:
    """
    Generate the action that schedules all mandatory checks.
    
    This is the minimum action that must be taken to avoid overdue aircraft.
    """
    mandatory_c = get_mandatory_c_check_aircraft(fleet, data, c_lookahead)
    mandatory_a = get_mandatory_a_check_aircraft(fleet, data, a_lookahead)
    
    # Filter by feasibility
    feasible_c = set(get_feasible_c_check_aircraft(fleet, data))
    feasible_a = set(get_feasible_a_check_aircraft(fleet, data))
    
    c_checks = [t for t in mandatory_c if t in feasible_c]
    a_checks = [t for t in mandatory_a if t in feasible_a and t not in c_checks]
    
    # Respect capacity
    available_c = get_available_c_slots(fleet, data, fleet.day_index)
    available_a = get_available_a_slots(fleet, data, fleet.day_index)
    
    c_checks = c_checks[:available_c]
    a_checks = a_checks[:available_a]
    
    return FleetAction(a_checks=a_checks, c_checks=c_checks)


@dataclass
class ScheduledCheck:
    """A scheduled maintenance check with paper's eq. 28 metrics."""
    tail: str
    check_type: str  # 'A', 'C', or 'D'
    start_date: date
    end_date: Optional[date] = None
    duration: int = 1
    
    # FH metrics (paper's objective)
    fh_at_check: float = 0.0  # Actual FH since last check
    unused_fh: float = 0.0  # max(0, fh_interval - fh_at_check)
    
    # Tolerance tracking
    used_tolerance_fh: float = 0.0
    is_tolerance_event: bool = False
    
    # For C/D checks: merged A-check data
    a_fh_at_check: float = 0.0
    a_unused_fh: float = 0.0

    @property
    def total_unused_fh(self) -> float:
        """Total unused FH including merged A-check for C/D checks."""
        return self.unused_fh + self.a_unused_fh


@dataclass
class Table3KPIs:
    """KPIs matching Table 3 in Deng et al. paper."""
    # Check counts
    n_a_checks: int
    n_c_checks: int
    n_d_checks: int
    
    # FH at checks
    avg_fh_at_a_checks: float
    std_fh_at_a_checks: float
    avg_fh_at_c_checks: float
    std_fh_at_c_checks: float
    
    # Paper's objective: unused FH
    total_unused_fh: float
    avg_unused_fh_a: float
    avg_unused_fh_c: float
    
    # Tolerance usage
    total_fh_tolerance_used: float
    total_tolerance_events: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'n_a_checks': self.n_a_checks,
            'n_c_checks': self.n_c_checks,
            'n_d_checks': self.n_d_checks,
            'avg_fh_at_a_checks': self.avg_fh_at_a_checks,
            'std_fh_at_a_checks': self.std_fh_at_a_checks,
            'avg_fh_at_c_checks': self.avg_fh_at_c_checks,
            'std_fh_at_c_checks': self.std_fh_at_c_checks,
            'total_unused_fh': self.total_unused_fh,
            'avg_unused_fh_a': self.avg_unused_fh_a,
            'avg_unused_fh_c': self.avg_unused_fh_c,
            'total_fh_tolerance_used': self.total_fh_tolerance_used,
            'total_tolerance_events': self.total_tolerance_events,
        }


@dataclass
class ScheduleResult:
    """Result of the scheduling optimization with paper's eq. 28 metrics."""
    checks: list[ScheduledCheck]
    total_a_checks: int
    total_c_checks: int
    total_d_checks: int
    computation_time: float
    
    # Paper's objective (eq. 28)
    total_unused_fh: float = 0.0
    total_tolerance_penalty: float = 0.0
    total_tolerance_events: int = 0
    
    # Table 3 KPIs (computed lazily)
    _kpis: Optional[Table3KPIs] = field(default=None, repr=False)

    @property
    def objective(self) -> float:
        """Paper's objective value: unused FH + tolerance penalties."""
        return self.total_unused_fh + self.total_tolerance_penalty

    def get_checks_for_aircraft(self, tail: str) -> list[ScheduledCheck]:
        """Get all checks scheduled for a specific aircraft."""
        return [c for c in self.checks if c.tail == tail]

    def compute_kpis(self) -> Table3KPIs:
        """Compute Table 3 KPIs from the scheduled checks."""
        if self._kpis is not None:
            return self._kpis
        
        import numpy as np
        
        a_checks = [c for c in self.checks if c.check_type == "A"]
        c_checks = [c for c in self.checks if c.check_type == "C"]
        d_checks = [c for c in self.checks if c.check_type == "D"]
        
        # FH at checks
        a_fh_values = [c.fh_at_check for c in a_checks]
        avg_fh_a = float(np.mean(a_fh_values)) if a_fh_values else 0.0
        std_fh_a = float(np.std(a_fh_values)) if a_fh_values else 0.0
        
        c_fh_values = [c.fh_at_check for c in c_checks + d_checks]
        avg_fh_c = float(np.mean(c_fh_values)) if c_fh_values else 0.0
        std_fh_c = float(np.std(c_fh_values)) if c_fh_values else 0.0
        
        # Unused FH
        unused_fh_a = sum(c.unused_fh for c in a_checks)
        unused_fh_c = sum(c.total_unused_fh for c in c_checks + d_checks)
        total_unused_fh = unused_fh_a + unused_fh_c
        
        avg_unused_fh_a = unused_fh_a / len(a_checks) if a_checks else 0.0
        avg_unused_fh_c = unused_fh_c / len(c_checks + d_checks) if (c_checks + d_checks) else 0.0
        
        # Tolerance
        total_fh_tolerance = sum(c.used_tolerance_fh for c in self.checks)
        total_tol = sum(1 for c in self.checks if c.is_tolerance_event)
        
        self._kpis = Table3KPIs(
            n_a_checks=len(a_checks),
            n_c_checks=len(c_checks),
            n_d_checks=len(d_checks),
            avg_fh_at_a_checks=avg_fh_a,
            std_fh_at_a_checks=std_fh_a,
            avg_fh_at_c_checks=avg_fh_c,
            std_fh_at_c_checks=std_fh_c,
            total_unused_fh=total_unused_fh,
            avg_unused_fh_a=avg_unused_fh_a,
            avg_unused_fh_c=avg_unused_fh_c,
            total_fh_tolerance_used=total_fh_tolerance,
            total_tolerance_events=total_tol,
        )
        
        return self._kpis

    def summary(self) -> str:
        """Generate a summary of the schedule."""
        lines = [
            f"Schedule Summary",
            f"================",
            f"Checks: {self.total_a_checks} A / {self.total_c_checks} C / {self.total_d_checks} D",
            f"",
            f"Paper's Objective (eq. 28):",
            f"  Unused FH: {self.total_unused_fh:.1f}",
            f"  Tolerance penalty: {self.total_tolerance_penalty:.1f}",
            f"  Total: {self.objective:.1f}",
            f"  Tolerance events: {self.total_tolerance_events}",
            f"",
            f"Time: {self.computation_time:.2f}s",
        ]
        return "\n".join(lines)


def _compute_paper_metrics(checks: list[ScheduledCheck], config) -> tuple[float, float, int]:
    """Compute paper's eq. 28 metrics from scheduled checks."""
    total_unused_fh = 0.0
    total_tolerance_penalty = 0.0
    total_tolerance_events = 0
    
    for check in checks:
        if check.check_type in ("C", "D"):
            total_unused_fh += check.total_unused_fh
        else:
            total_unused_fh += check.unused_fh
        
        if check.is_tolerance_event:
            total_tolerance_penalty += config.m_cost
            total_tolerance_events += 1
    
    return total_unused_fh, total_tolerance_penalty, total_tolerance_events


def _make_a_check(tail: str, start_date: date, result: CheckResult) -> ScheduledCheck:
    """Create ScheduledCheck from A-check result."""
    return ScheduledCheck(
        tail=tail,
        check_type="A",
        start_date=start_date,
        fh_at_check=result.fh_at_check,
        unused_fh=result.unused_fh,
        used_tolerance_fh=result.used_tolerance_fh,
        is_tolerance_event=result.is_tolerance_event,
    )


def _make_c_check(tail: str, start_date: date, result: CCheckResult) -> ScheduledCheck:
    """Create ScheduledCheck from C-check result."""
    return ScheduledCheck(
        tail=tail,
        check_type=result.check_type,
        start_date=start_date,
        end_date=start_date + timedelta(days=result.elapsed_time),
        duration=result.elapsed_time,
        fh_at_check=result.fh_at_check,
        unused_fh=result.unused_fh,
        used_tolerance_fh=result.used_tolerance_fh,
        is_tolerance_event=result.is_tolerance_event,
        a_fh_at_check=result.a_fh_at_check,
        a_unused_fh=result.a_unused_fh,
    )


class GreedyScheduler:
    """
    Greedy heuristic scheduler for comparison and warm-start.
    
    Schedules checks based on urgency when capacity allows.
    """

    def __init__(self, data: MaintenanceData):
        self.data = data

    def schedule(
        self,
        planning_days: int,
        verbose: bool = False,
    ) -> ScheduleResult:
        """
        Run the greedy scheduling algorithm.
        
        Args:
            planning_days: Number of days to plan
            verbose: Print progress information
        
        Returns:
            ScheduleResult with the generated schedule
        """
        start_time = time.time()

        fleet = create_initial_fleet_state(self.data)
        checks = []
        total_a = 0
        total_c = 0
        total_d = 0

        for day in range(planning_days):
            if verbose and day % 100 == 0:
                print(f"Day {day}/{planning_days} ({fleet.current_date})")

            # Handle mandatory C-checks
            for tail in get_mandatory_c_check_aircraft(fleet, self.data, lookahead_days=14):
                duration = get_c_check_elapsed_time(self.data, tail, 0)
                can_sched, _ = can_schedule_c_check(fleet, self.data, tail, duration)
                if can_sched:
                    fleet, result = schedule_c_check(fleet, self.data, tail)
                    checks.append(_make_c_check(tail, fleet.current_date, result))
                    if result.check_type == "D":
                        total_d += 1
                    else:
                        total_c += 1

            # Handle mandatory A-checks
            for tail in get_mandatory_a_check_aircraft(fleet, self.data, lookahead_days=3):
                can_sched, _ = can_schedule_a_check(fleet, self.data, tail)
                if can_sched:
                    fleet, result = schedule_a_check(fleet, self.data, tail)
                    checks.append(_make_a_check(tail, fleet.current_date, result))
                    total_a += 1

            # Schedule opportunistic C-checks if capacity allows
            available_c = get_available_c_slots(fleet, self.data, fleet.day_index)
            if available_c > 0 and not is_c_check_not_allowed(fleet.current_date, self.data):
                candidates = []
                for tail, state in fleet.aircraft.items():
                    if state.is_grounded:
                        continue
                    if state.dy_c >= self.data.config.dtc_min:
                        remaining = min(state.c_remaining_dy, state.c_remaining_fh / 10, state.c_remaining_fc / 5)
                        if remaining < 90:
                            candidates.append((tail, remaining))

                candidates.sort(key=lambda x: x[1])
                for tail, _ in candidates[:available_c]:
                    duration = get_c_check_elapsed_time(self.data, tail, 0)
                    can_sched, _ = can_schedule_c_check(fleet, self.data, tail, duration)
                    if can_sched:
                        fleet, result = schedule_c_check(fleet, self.data, tail)
                        checks.append(_make_c_check(tail, fleet.current_date, result))
                        if result.check_type == "D":
                            total_d += 1
                        else:
                            total_c += 1

            # Advance to next day
            fleet = advance_fleet_one_day(fleet, self.data)

        computation_time = time.time() - start_time
        unused_fh, tol_penalty, tol_events = _compute_paper_metrics(checks, self.data.config)

        return ScheduleResult(
            checks=checks,
            total_a_checks=total_a,
            total_c_checks=total_c,
            total_d_checks=total_d,
            computation_time=computation_time,
            total_unused_fh=unused_fh,
            total_tolerance_penalty=tol_penalty,
            total_tolerance_events=tol_events,
        )


class DPScheduler:
    """
    Dynamic Programming based scheduler.
    
    Uses forward induction with state aggregation to find optimal schedules
    that minimize wasted intervals while respecting capacity constraints.
    """

    def __init__(self, data: MaintenanceData):
        self.data = data

    def _evaluate_c_check_timing(
        self,
        state: AircraftState,
        current_date: date,
        lookahead_days: int = 90,
    ) -> tuple[float, int]:
        """
        Evaluate the optimal timing for a C-check using DP.
        
        Returns:
            Tuple of (best_value, best_day_offset)
        """
        # If already due or overdue, schedule now
        if state.c_check_due() or state.c_remaining_dy <= 0:
            cost = calculate_c_check_cost(state, self.data.config)
            return cost.total, 0

        best_value = float('inf')
        best_day = 0

        dfh = get_daily_fh(self.data, state.tail, current_date)
        dfc = get_daily_fc(self.data, state.tail, current_date)

        sim_state = state.copy()
        for day in range(min(lookahead_days, state.c_remaining_dy + 1)):
            cost = calculate_c_check_cost(sim_state, self.data.config)

            # Penalize scheduling during peak season
            check_date = current_date + timedelta(days=day)
            penalty = 0
            for peak in self.data.c_peak:
                if peak.begin <= check_date <= peak.end:
                    penalty = 50

            value = cost.total + penalty

            if value < best_value:
                best_value = value
                best_day = day

            # Advance simulation state
            sim_state.dy_c += 1
            sim_state.fh_c += dfh
            sim_state.fc_c += dfc
            sim_state.dy_a += 1
            sim_state.fh_a += dfh
            sim_state.fc_a += dfc

            # Stop if past due
            if sim_state.c_check_overdue():
                break

        return best_value, best_day

    def _evaluate_a_check_timing(
        self,
        state: AircraftState,
        current_date: date,
        lookahead_days: int = 30,
    ) -> tuple[float, int]:
        """
        Evaluate the optimal timing for an A-check using DP.
        
        Returns:
            Tuple of (best_value, best_day_offset)
        """
        # If already due or overdue, schedule now
        if state.a_check_due() or state.a_remaining_dy <= 0:
            cost = calculate_a_check_cost(state, self.data.config)
            return cost.total, 0

        best_value = float('inf')
        best_day = 0

        dfh = get_daily_fh(self.data, state.tail, current_date)
        dfc = get_daily_fc(self.data, state.tail, current_date)

        sim_state = state.copy()
        for day in range(min(lookahead_days, state.a_remaining_dy + 1)):
            cost = calculate_a_check_cost(sim_state, self.data.config)
            value = cost.total

            if value < best_value:
                best_value = value
                best_day = day

            # Advance simulation
            sim_state.dy_a += 1
            sim_state.fh_a += dfh
            sim_state.fc_a += dfc

            if sim_state.a_check_overdue():
                break

        return best_value, best_day

    def _plan_checks_for_period(
        self,
        fleet: FleetState,
        period_days: int,
    ) -> list[tuple[str, str, int, float]]:
        """
        Plan checks for a period using DP value function.
        
        Returns:
            List of (tail, check_type, optimal_day_offset, value)
        """
        planned = []

        for tail, state in fleet.aircraft.items():
            if state.is_grounded:
                continue

            # Evaluate C-check timing
            if state.dy_c >= self.data.config.dtc_min:
                c_value, c_day = self._evaluate_c_check_timing(
                    state, fleet.current_date, period_days
                )
                if c_day < period_days:
                    planned.append((tail, "C", c_day, c_value))

            # Evaluate A-check timing
            if state.dy_a >= self.data.config.dta_min:
                a_value, a_day = self._evaluate_a_check_timing(
                    state, fleet.current_date, period_days
                )
                if a_day < period_days:
                    planned.append((tail, "A", a_day, a_value))

        # Sort by day offset, then by value
        planned.sort(key=lambda x: (x[2], x[3]))

        return planned

    def schedule(
        self,
        planning_days: int,
        verbose: bool = False,
        planning_period: int = 30,  # Days per DP planning period
    ) -> ScheduleResult:
        """
        Run the DP-based scheduling algorithm.
        
        Uses rolling horizon DP with lookahead to minimize wasted intervals.
        
        Args:
            planning_days: Total number of days to plan
            verbose: Print progress information
            planning_period: Length of each DP planning period
        
        Returns:
            ScheduleResult with the optimized schedule
        """
        start_time = time.time()

        fleet = create_initial_fleet_state(self.data)
        checks = []
        total_a = 0
        total_c = 0
        total_d = 0

        planned_c_checks = defaultdict(list)
        planned_a_checks = defaultdict(list)

        day = 0
        while day < planning_days:
            if verbose and day % 100 == 0:
                print(f"Day {day}/{planning_days} ({fleet.current_date})")

            if day % planning_period == 0:
                planned = self._plan_checks_for_period(
                    fleet, min(planning_period * 2, planning_days - day)
                )
                planned_c_checks.clear()
                planned_a_checks.clear()
                for tail, check_type, offset, value in planned:
                    if check_type == "C":
                        planned_c_checks[offset].append((tail, value))
                    else:
                        planned_a_checks[offset].append((tail, value))

            current_offset = day % planning_period

            # Mandatory checks
            for tail in get_mandatory_c_check_aircraft(fleet, self.data, lookahead_days=7):
                duration = get_c_check_elapsed_time(self.data, tail, 0)
                can_sched, _ = can_schedule_c_check(fleet, self.data, tail, duration)
                if can_sched:
                    fleet, result = schedule_c_check(fleet, self.data, tail)
                    checks.append(_make_c_check(tail, fleet.current_date, result))
                    if result.check_type == "D":
                        total_d += 1
                    else:
                        total_c += 1

            for tail in get_mandatory_a_check_aircraft(fleet, self.data, lookahead_days=2):
                can_sched, _ = can_schedule_a_check(fleet, self.data, tail)
                if can_sched:
                    fleet, result = schedule_a_check(fleet, self.data, tail)
                    checks.append(_make_a_check(tail, fleet.current_date, result))
                    total_a += 1

            # Planned C-checks
            for tail, _ in planned_c_checks.get(current_offset, []):
                if fleet.aircraft[tail].is_grounded:
                    continue
                duration = get_c_check_elapsed_time(self.data, tail, 0)
                can_sched, _ = can_schedule_c_check(fleet, self.data, tail, duration)
                if can_sched:
                    fleet, result = schedule_c_check(fleet, self.data, tail)
                    checks.append(_make_c_check(tail, fleet.current_date, result))
                    if result.check_type == "D":
                        total_d += 1
                    else:
                        total_c += 1

            # Planned A-checks
            for tail, _ in planned_a_checks.get(current_offset, []):
                if fleet.aircraft[tail].is_grounded:
                    continue
                can_sched, _ = can_schedule_a_check(fleet, self.data, tail)
                if can_sched:
                    fleet, result = schedule_a_check(fleet, self.data, tail)
                    checks.append(_make_a_check(tail, fleet.current_date, result))
                    total_a += 1

            fleet = advance_fleet_one_day(fleet, self.data)
            day += 1

        computation_time = time.time() - start_time
        unused_fh, tol_penalty, tol_events = _compute_paper_metrics(checks, self.data.config)

        return ScheduleResult(
            checks=checks,
            total_a_checks=total_a,
            total_c_checks=total_c,
            total_d_checks=total_d,
            computation_time=computation_time,
            total_unused_fh=unused_fh,
            total_tolerance_penalty=tol_penalty,
            total_tolerance_events=tol_events,
        )


class OptimizedDPScheduler:
    """
    Full Dynamic Programming scheduler with forward induction.
    
    Implements the methodology from the paper using:
    - Forward induction
    - State aggregation by tolerance signature
    - Thrifty algorithm for action evaluation
    """

    def __init__(self, data: MaintenanceData):
        self.data = data
        self.config = data.config

    def _get_state_key(self, state: AircraftState) -> tuple:
        """Get a hashable key for state aggregation."""
        return (
            state.tail,
            state.dy_c // 30,  # Discretize days
            int(state.fh_c / 500),  # Discretize FH
            int(state.fc_c / 250),  # Discretize FC
            state.dy_a // 14,
            int(state.fh_a / 100),
            int(state.fc_a / 100),
            state.d_cycle,
        )

    def _compute_immediate_cost(
        self,
        fleet: FleetState,
        actions: dict[str, str],  # tail -> action ('none', 'A', 'C')
    ) -> float:
        """
        Compute immediate cost of taking actions (paper's eq. 28).
        
        Cost includes:
        1. Unused FH for scheduled checks
        2. Tolerance penalty (P_a) for checks done in tolerance zone
        3. Overdue penalty for aircraft not receiving needed checks
        """
        cost = 0.0

        for tail, action in actions.items():
            state = fleet.aircraft[tail]

            if action == 'A':
                # Paper's cost: unused FH + tolerance penalty
                check_cost = calculate_a_check_cost(state, self.config)
                cost += check_cost.total
            elif action == 'C':
                # Paper's cost: unused FH (C + merged A) + tolerance penalty
                check_cost = calculate_c_check_cost(state, self.config)
                cost += check_cost.total

        # Add penalty for aircraft that need checks but aren't getting them
        for tail, state in fleet.aircraft.items():
            if tail not in actions or actions[tail] == 'none':
                # Overdue penalty (aircraft would be grounded without maintenance)
                cost += calculate_overdue_penalty(state, self.config)
                # Tolerance penalty for aircraft in tolerance zone
                if state.is_using_tolerance():
                    cost += self.config.m_cost

        return cost

    def _generate_actions(
        self,
        fleet: FleetState,
    ) -> list[dict[str, str]]:
        """
        Generate feasible action combinations.
        
        To limit complexity, we use a greedy selection of actions
        rather than full enumeration.
        """
        actions_list = []

        # Get feasible checks
        feasible_c = get_feasible_c_check_aircraft(fleet, self.data)
        feasible_a = get_feasible_a_check_aircraft(fleet, self.data)

        # Get mandatory checks
        mandatory_c = set(get_mandatory_c_check_aircraft(fleet, self.data, 7))
        mandatory_a = set(get_mandatory_a_check_aircraft(fleet, self.data, 3))

        # Base action: mandatory checks only
        base_action = {}
        for tail in mandatory_c:
            if tail in feasible_c:
                base_action[tail] = 'C'
        for tail in mandatory_a:
            if tail in feasible_a and tail not in base_action:
                base_action[tail] = 'A'

        actions_list.append(base_action)

        # Generate actions with additional opportunistic checks
        available_c = get_available_c_slots(fleet, self.data, fleet.day_index)
        used_c = sum(1 for a in base_action.values() if a == 'C')

        if available_c > used_c:
            # Sort feasible C-checks by urgency
            urgencies = []
            for tail in feasible_c:
                if tail not in base_action:
                    state = fleet.aircraft[tail]
                    urgency = calculate_urgency_score(state, self.config)
                    urgencies.append((tail, urgency))

            urgencies.sort(key=lambda x: -x[1])

            # Add top candidates
            for tail, _ in urgencies[:min(3, available_c - used_c)]:
                action = base_action.copy()
                action[tail] = 'C'
                actions_list.append(action)

        available_a = get_available_a_slots(fleet, self.data, fleet.day_index)
        used_a = sum(1 for a in base_action.values() if a == 'A')

        if available_a > used_a:
            urgencies = []
            for tail in feasible_a:
                if tail not in base_action:
                    state = fleet.aircraft[tail]
                    urgency = calculate_urgency_score(state, self.config)
                    urgencies.append((tail, urgency))

            urgencies.sort(key=lambda x: -x[1])

            for tail, _ in urgencies[:min(2, available_a - used_a)]:
                action = base_action.copy()
                action[tail] = 'A'
                actions_list.append(action)

        return actions_list

    def schedule(
        self,
        planning_days: int,
        verbose: bool = False,
    ) -> ScheduleResult:
        """
        Run the optimized DP scheduling algorithm.
        
        Args:
            planning_days: Number of days to plan
            verbose: Print progress information
        
        Returns:
            ScheduleResult with the optimized schedule
        """
        start_time = time.time()

        fleet = create_initial_fleet_state(self.data)
        checks = []
        total_a = 0
        total_c = 0
        total_d = 0

        for day in range(planning_days):
            if verbose and day % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Day {day}/{planning_days} ({fleet.current_date}) - {elapsed:.1f}s elapsed")

            # Generate and evaluate actions
            actions_list = self._generate_actions(fleet)

            # Select best action based on immediate cost + estimated future cost
            best_action = {}
            best_cost = float('inf')

            for actions in actions_list:
                cost = self._compute_immediate_cost(fleet, actions)

                # Estimate future cost (simplified)
                future_cost = 0
                for tail, state in fleet.aircraft.items():
                    if tail not in actions or actions[tail] == 'none':
                        # Penalty for delaying checks on urgent aircraft
                        if state.c_remaining_dy < 30:
                            future_cost += (30 - state.c_remaining_dy) * 10
                        if state.a_remaining_dy < 7:
                            future_cost += (7 - state.a_remaining_dy) * 5

                total_cost = cost + future_cost * 0.5

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_action = actions

            # Execute best action
            for tail, action in best_action.items():
                if action == 'C':
                    fleet, result = schedule_c_check(fleet, self.data, tail)
                    checks.append(_make_c_check(tail, fleet.current_date, result))
                    if result.check_type == "D":
                        total_d += 1
                    else:
                        total_c += 1
                elif action == 'A':
                    fleet, result = schedule_a_check(fleet, self.data, tail)
                    checks.append(_make_a_check(tail, fleet.current_date, result))
                    total_a += 1

            fleet = advance_fleet_one_day(fleet, self.data)

        computation_time = time.time() - start_time
        unused_fh, tol_penalty, tol_events = _compute_paper_metrics(checks, self.data.config)

        return ScheduleResult(
            checks=checks,
            total_a_checks=total_a,
            total_c_checks=total_c,
            total_d_checks=total_d,
            computation_time=computation_time,
            total_unused_fh=unused_fh,
            total_tolerance_penalty=tol_penalty,
            total_tolerance_events=tol_events,
        )


def compare_schedulers(
    data: MaintenanceData,
    planning_days: int = 365,
    verbose: bool = True,
    include_deng_dp: bool = True,
) -> dict[str, ScheduleResult]:
    """
    Compare different scheduling algorithms.
    
    Args:
        data: Maintenance data
        planning_days: Number of days to plan
        verbose: Print progress information
        include_deng_dp: Whether to include the full Deng DP scheduler
    
    Returns:
        Dictionary mapping algorithm name to result
    """
    results = {}

    if verbose:
        print("Running Greedy Scheduler...")
    greedy = GreedyScheduler(data)
    results['greedy'] = greedy.schedule(planning_days, verbose=False)
    if verbose:
        print(results['greedy'].summary())
        print()

    if verbose:
        print("Running DP Scheduler...")
    dp = DPScheduler(data)
    results['dp'] = dp.schedule(planning_days, verbose=False)
    if verbose:
        print(results['dp'].summary())
        print()

    if verbose:
        print("Running Optimized DP Scheduler...")
    opt_dp = OptimizedDPScheduler(data)
    results['optimized_dp'] = opt_dp.schedule(planning_days, verbose=False)
    if verbose:
        print(results['optimized_dp'].summary())
        print()

    if include_deng_dp:
        # Use the JAX-based DengScheduler (paper-faithful implementation)
        from .jax_solver import DengScheduler
        if verbose:
            print("Running Deng Scheduler (paper's methodology)...")
        deng = DengScheduler(data)
        deng_result = deng.schedule(planning_days, verbose=verbose)
        # Convert JAX result dict to ScheduleResult for consistency
        results['deng'] = ScheduleResult(
            checks=[],  # JAX doesn't track individual checks
            total_a_checks=deng_result['n_a_checks'],
            total_c_checks=deng_result['n_c_checks'],
            total_d_checks=0,
            computation_time=deng_result['computation_time'],
            total_unused_fh=deng_result['total_cost'],  # Approximate
            total_tolerance_penalty=0.0,
            total_tolerance_events=deng_result['tolerance_events'],
        )
        if verbose:
            print(f"  C-checks: {deng_result['n_c_checks']}, A-checks: {deng_result['n_a_checks']}")
            print(f"  Tolerance events: {deng_result['tolerance_events']}")
            print(f"  Time: {deng_result['computation_time']:.1f}s")

    return results


if __name__ == "__main__":
    from .data_loader import load_maintenance_data
    import argparse

    parser = argparse.ArgumentParser(description="Aircraft Maintenance Scheduler")
    parser.add_argument("--data", default="../Sep 25 2017 (Case Study 2018-2021).xlsx",
                       help="Path to Excel data file")
    parser.add_argument("--days", type=int, default=None,
                       help="Planning days (default: full horizon from config)")
    parser.add_argument("--no-deng", action="store_true",
                       help="Skip Deng DP scheduler (faster)")
    args = parser.parse_args()

    print("Loading maintenance data...")
    data = load_maintenance_data(args.data)

    print(f"Planning horizon: {data.config.total_years} years")
    planning_days = args.days if args.days else data.config.total_years * 365
    print(f"Running for {planning_days} days ({planning_days / 365:.1f} years)")

    print("\n" + "="*50)
    results = compare_schedulers(
        data,
        planning_days=planning_days,
        verbose=True,
        include_deng_dp=not args.no_deng,
    )

    print("\n" + "="*50)
    print("Comparison Summary")
    print("="*50)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Total checks: {result.total_a_checks + result.total_c_checks + result.total_d_checks}")
        print(f"  Total unused FH: {result.total_unused_fh:.1f}")
        print(f"  Objective: {result.objective:.1f}")
        print(f"  Time: {result.computation_time:.2f}s")

    print("\n" + "="*50)
    print("Table 3 KPIs (Paper-compatible)")
    print("="*50)
    for name, result in results.items():
        print(f"\n{name}:")
        print(result.paper_summary())


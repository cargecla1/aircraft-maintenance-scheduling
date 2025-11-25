"""
Dynamic Programming Solver for Aircraft Maintenance Check Scheduling.

Implements forward induction with state aggregation to find optimal
maintenance schedules that minimize wasted intervals.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional
from collections import defaultdict
import heapq
import time

from data_loader import MaintenanceData
from state import FleetState, AircraftState, create_initial_fleet_state
from dynamics import (
    advance_fleet_one_day,
    schedule_a_check,
    schedule_c_check,
    get_c_check_elapsed_time,
    get_daily_fh,
    get_daily_fc,
)
from cost import (
    calculate_a_check_wasted_interval,
    calculate_c_check_wasted_interval,
    calculate_urgency_score,
    calculate_overdue_penalty,
)
from constraints import (
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


@dataclass
class ScheduledCheck:
    """A scheduled maintenance check."""
    tail: str
    check_type: str  # 'A', 'C', or 'D'
    start_date: date
    end_date: Optional[date] = None
    duration: int = 1
    wasted_interval: float = 0.0


@dataclass
class ScheduleResult:
    """Result of the scheduling optimization."""
    checks: list[ScheduledCheck]
    total_wasted: float
    total_a_checks: int
    total_c_checks: int
    total_d_checks: int
    computation_time: float

    def get_checks_for_aircraft(self, tail: str) -> list[ScheduledCheck]:
        """Get all checks scheduled for a specific aircraft."""
        return [c for c in self.checks if c.tail == tail]

    def summary(self) -> str:
        """Generate a summary of the schedule."""
        lines = [
            f"Schedule Summary",
            f"================",
            f"Total A-checks: {self.total_a_checks}",
            f"Total C-checks: {self.total_c_checks}",
            f"Total D-checks: {self.total_d_checks}",
            f"Total checks: {self.total_a_checks + self.total_c_checks + self.total_d_checks}",
            f"Total wasted interval: {self.total_wasted:.1f}",
            f"Computation time: {self.computation_time:.2f} seconds",
        ]
        return "\n".join(lines)


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
        total_wasted = 0.0
        total_a = 0
        total_c = 0
        total_d = 0

        for day in range(planning_days):
            if verbose and day % 100 == 0:
                print(f"Day {day}/{planning_days} ({fleet.current_date})")

            # First, handle mandatory C-checks
            mandatory_c = get_mandatory_c_check_aircraft(fleet, self.data, lookahead_days=14)
            for tail in mandatory_c:
                duration = get_c_check_elapsed_time(self.data, tail, 0)
                can_sched, _ = can_schedule_c_check(fleet, self.data, tail, duration)
                if can_sched:
                    state = fleet.aircraft[tail]
                    is_d = state.next_c_check_is_d()
                    fleet, wasted, actual_duration = schedule_c_check(fleet, self.data, tail)
                    check_type = "D" if is_d else "C"
                    checks.append(ScheduledCheck(
                        tail=tail,
                        check_type=check_type,
                        start_date=fleet.current_date,
                        end_date=fleet.current_date + timedelta(days=actual_duration),
                        duration=actual_duration,
                        wasted_interval=wasted,
                    ))
                    total_wasted += wasted
                    if is_d:
                        total_d += 1
                    else:
                        total_c += 1

            # Then, handle mandatory A-checks
            mandatory_a = get_mandatory_a_check_aircraft(fleet, self.data, lookahead_days=3)
            for tail in mandatory_a:
                can_sched, _ = can_schedule_a_check(fleet, self.data, tail)
                if can_sched:
                    fleet, wasted = schedule_a_check(fleet, self.data, tail)
                    checks.append(ScheduledCheck(
                        tail=tail,
                        check_type="A",
                        start_date=fleet.current_date,
                        wasted_interval=wasted,
                    ))
                    total_wasted += wasted
                    total_a += 1

            # Schedule opportunistic C-checks if capacity allows
            available_c = get_available_c_slots(fleet, self.data, fleet.day_index)
            if available_c > 0 and not is_c_check_not_allowed(fleet.current_date, self.data):
                # Find aircraft close to C-check due
                candidates = []
                for tail, state in fleet.aircraft.items():
                    if state.is_grounded:
                        continue
                    if state.dy_c >= self.data.config.dtc_min:
                        remaining = min(state.c_remaining_dy, state.c_remaining_fh / 10, state.c_remaining_fc / 5)
                        if remaining < 90:  # Within 90 days of due
                            candidates.append((tail, remaining))

                candidates.sort(key=lambda x: x[1])
                for tail, _ in candidates[:available_c]:
                    duration = get_c_check_elapsed_time(self.data, tail, 0)
                    can_sched, _ = can_schedule_c_check(fleet, self.data, tail, duration)
                    if can_sched:
                        state = fleet.aircraft[tail]
                        is_d = state.next_c_check_is_d()
                        fleet, wasted, actual_duration = schedule_c_check(fleet, self.data, tail)
                        check_type = "D" if is_d else "C"
                        checks.append(ScheduledCheck(
                            tail=tail,
                            check_type=check_type,
                            start_date=fleet.current_date,
                            end_date=fleet.current_date + timedelta(days=actual_duration),
                            duration=actual_duration,
                            wasted_interval=wasted,
                        ))
                        total_wasted += wasted
                        if is_d:
                            total_d += 1
                        else:
                            total_c += 1

            # Advance to next day
            fleet = advance_fleet_one_day(fleet, self.data)

        computation_time = time.time() - start_time

        return ScheduleResult(
            checks=checks,
            total_wasted=total_wasted,
            total_a_checks=total_a,
            total_c_checks=total_c,
            total_d_checks=total_d,
            computation_time=computation_time,
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
            wasted = calculate_c_check_wasted_interval(state, self.data.config)
            return wasted.total, 0

        # Simulate different scheduling times
        best_value = float('inf')
        best_day = 0

        # Get flight rates for simulation
        dfh = get_daily_fh(self.data, state.tail, current_date)
        dfc = get_daily_fc(self.data, state.tail, current_date)

        sim_state = state.copy()
        for day in range(min(lookahead_days, state.c_remaining_dy + 1)):
            # Calculate wasted interval if scheduled on this day
            wasted = calculate_c_check_wasted_interval(sim_state, self.data.config)

            # Penalize scheduling during peak season
            check_date = current_date + timedelta(days=day)
            penalty = 0
            for peak in self.data.c_peak:
                if peak.begin <= check_date <= peak.end:
                    penalty = 50  # Penalty for peak season

            value = wasted.total + penalty

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
            wasted = calculate_a_check_wasted_interval(state, self.data.config)
            return wasted.total, 0

        # Simulate different scheduling times
        best_value = float('inf')
        best_day = 0

        # Get flight rates
        dfh = get_daily_fh(self.data, state.tail, current_date)
        dfc = get_daily_fc(self.data, state.tail, current_date)

        sim_state = state.copy()
        for day in range(min(lookahead_days, state.a_remaining_dy + 1)):
            wasted = calculate_a_check_wasted_interval(sim_state, self.data.config)
            value = wasted.total

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
        total_wasted = 0.0
        total_a = 0
        total_c = 0
        total_d = 0

        # Track planned checks by day
        planned_c_checks = defaultdict(list)  # day_offset -> [(tail, value)]
        planned_a_checks = defaultdict(list)

        day = 0
        while day < planning_days:
            if verbose and day % 100 == 0:
                print(f"Day {day}/{planning_days} ({fleet.current_date})")

            # Re-plan periodically
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

            # First priority: Handle mandatory checks (overdue or imminent)
            mandatory_c = get_mandatory_c_check_aircraft(fleet, self.data, lookahead_days=7)
            for tail in mandatory_c:
                duration = get_c_check_elapsed_time(self.data, tail, 0)
                can_sched, _ = can_schedule_c_check(fleet, self.data, tail, duration)
                if can_sched:
                    state = fleet.aircraft[tail]
                    is_d = state.next_c_check_is_d()
                    fleet, wasted, actual_duration = schedule_c_check(fleet, self.data, tail)
                    check_type = "D" if is_d else "C"
                    checks.append(ScheduledCheck(
                        tail=tail,
                        check_type=check_type,
                        start_date=fleet.current_date,
                        end_date=fleet.current_date + timedelta(days=actual_duration),
                        duration=actual_duration,
                        wasted_interval=wasted,
                    ))
                    total_wasted += wasted
                    if is_d:
                        total_d += 1
                    else:
                        total_c += 1

            mandatory_a = get_mandatory_a_check_aircraft(fleet, self.data, lookahead_days=2)
            for tail in mandatory_a:
                can_sched, _ = can_schedule_a_check(fleet, self.data, tail)
                if can_sched:
                    fleet, wasted = schedule_a_check(fleet, self.data, tail)
                    checks.append(ScheduledCheck(
                        tail=tail,
                        check_type="A",
                        start_date=fleet.current_date,
                        wasted_interval=wasted,
                    ))
                    total_wasted += wasted
                    total_a += 1

            # Second priority: Execute planned C-checks for today
            for tail, value in planned_c_checks.get(current_offset, []):
                if fleet.aircraft[tail].is_grounded:
                    continue
                duration = get_c_check_elapsed_time(self.data, tail, 0)
                can_sched, _ = can_schedule_c_check(fleet, self.data, tail, duration)
                if can_sched:
                    state = fleet.aircraft[tail]
                    is_d = state.next_c_check_is_d()
                    fleet, wasted, actual_duration = schedule_c_check(fleet, self.data, tail)
                    check_type = "D" if is_d else "C"
                    checks.append(ScheduledCheck(
                        tail=tail,
                        check_type=check_type,
                        start_date=fleet.current_date,
                        end_date=fleet.current_date + timedelta(days=actual_duration),
                        duration=actual_duration,
                        wasted_interval=wasted,
                    ))
                    total_wasted += wasted
                    if is_d:
                        total_d += 1
                    else:
                        total_c += 1

            # Third priority: Execute planned A-checks for today
            for tail, value in planned_a_checks.get(current_offset, []):
                if fleet.aircraft[tail].is_grounded:
                    continue
                can_sched, _ = can_schedule_a_check(fleet, self.data, tail)
                if can_sched:
                    fleet, wasted = schedule_a_check(fleet, self.data, tail)
                    checks.append(ScheduledCheck(
                        tail=tail,
                        check_type="A",
                        start_date=fleet.current_date,
                        wasted_interval=wasted,
                    ))
                    total_wasted += wasted
                    total_a += 1

            # Advance to next day
            fleet = advance_fleet_one_day(fleet, self.data)
            day += 1

        computation_time = time.time() - start_time

        return ScheduleResult(
            checks=checks,
            total_wasted=total_wasted,
            total_a_checks=total_a,
            total_c_checks=total_c,
            total_d_checks=total_d,
            computation_time=computation_time,
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
        """Compute immediate cost of taking actions."""
        cost = 0.0

        for tail, action in actions.items():
            state = fleet.aircraft[tail]

            if action == 'A':
                wasted = calculate_a_check_wasted_interval(state, self.config)
                cost += wasted.total
            elif action == 'C':
                wasted = calculate_c_check_wasted_interval(state, self.config)
                cost += wasted.total

        # Add penalty for overdue aircraft
        for tail, state in fleet.aircraft.items():
            if tail not in actions or actions[tail] == 'none':
                cost += calculate_overdue_penalty(state, self.config)

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
        total_wasted = 0.0
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
                    state = fleet.aircraft[tail]
                    is_d = state.next_c_check_is_d()
                    fleet, wasted, duration = schedule_c_check(fleet, self.data, tail)
                    check_type = "D" if is_d else "C"
                    checks.append(ScheduledCheck(
                        tail=tail,
                        check_type=check_type,
                        start_date=fleet.current_date,
                        end_date=fleet.current_date + timedelta(days=duration),
                        duration=duration,
                        wasted_interval=wasted,
                    ))
                    total_wasted += wasted
                    if is_d:
                        total_d += 1
                    else:
                        total_c += 1

                elif action == 'A':
                    fleet, wasted = schedule_a_check(fleet, self.data, tail)
                    checks.append(ScheduledCheck(
                        tail=tail,
                        check_type="A",
                        start_date=fleet.current_date,
                        wasted_interval=wasted,
                    ))
                    total_wasted += wasted
                    total_a += 1

            # Advance to next day
            fleet = advance_fleet_one_day(fleet, self.data)

        computation_time = time.time() - start_time

        return ScheduleResult(
            checks=checks,
            total_wasted=total_wasted,
            total_a_checks=total_a,
            total_c_checks=total_c,
            total_d_checks=total_d,
            computation_time=computation_time,
        )


def compare_schedulers(
    data: MaintenanceData,
    planning_days: int = 365,
    verbose: bool = True,
) -> dict[str, ScheduleResult]:
    """
    Compare different scheduling algorithms.
    
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

    return results


if __name__ == "__main__":
    from data_loader import load_maintenance_data

    print("Loading maintenance data...")
    data = load_maintenance_data("Sep 25 2017 (Case Study 2018-2021).xlsx")

    print(f"Planning horizon: {data.config.total_years} years")
    planning_days = data.config.total_years * 365

    print("\n" + "="*50)
    results = compare_schedulers(data, planning_days=min(365, planning_days), verbose=True)

    print("\n" + "="*50)
    print("Comparison Summary")
    print("="*50)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Total checks: {result.total_a_checks + result.total_c_checks + result.total_d_checks}")
        print(f"  Total wasted: {result.total_wasted:.1f}")
        print(f"  Time: {result.computation_time:.2f}s")


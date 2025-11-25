#!/usr/bin/env python3
"""
Regression tests for Aircraft Maintenance Check Scheduling.

Verifies that the refactored package structure works correctly.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    # Test package-level imports
    from aircraft_scheduler import (
        # Data loading
        MaintenanceData,
        load_maintenance_data,
        AircraftDInitial,
        AircraftCInitial,
        AircraftAInitial,
        FlightRates,
        CCheckInfo,
        Config,
        DateRange,
        PeakSeason,
        ReservedMaintenance,
        AdditionalSlots,
        # State
        AircraftState,
        FleetState,
        create_initial_fleet_state,
        # Dynamics
        age_aircraft,
        advance_fleet_one_day,
        schedule_a_check,
        schedule_c_check,
        perform_a_check,
        perform_c_check,
        get_daily_fh,
        get_daily_fc,
        get_c_check_elapsed_time,
        # Constraints
        ConstraintViolation,
        can_schedule_c_check,
        can_schedule_a_check,
        get_available_c_slots,
        get_available_a_slots,
        get_feasible_c_check_aircraft,
        get_feasible_a_check_aircraft,
        get_mandatory_c_check_aircraft,
        get_mandatory_a_check_aircraft,
        is_c_check_not_allowed,
        is_a_check_not_allowed,
        validate_fleet_state,
        # Cost
        CheckCost,
        calculate_a_check_wasted_interval,
        calculate_c_check_wasted_interval,
        calculate_urgency_score,
        calculate_overdue_penalty,
        # Solver
        ScheduledCheck,
        ScheduleResult,
        GreedyScheduler,
        DPScheduler,
        OptimizedDPScheduler,
        compare_schedulers,
        # Visualization
        plot_gantt_chart,
        plot_check_distribution,
        plot_capacity_utilization,
        generate_report,
    )
    
    print("  All imports successful!")
    return True


def test_data_loading():
    """Test that data loading works correctly."""
    print("Testing data loading...")
    
    from aircraft_scheduler import load_maintenance_data
    
    data_file = Path("Sep 25 2017 (Case Study 2018-2021).xlsx")
    if not data_file.exists():
        print(f"  Skipping: Data file not found: {data_file}")
        return True
    
    data = load_maintenance_data(str(data_file))
    
    assert len(data.aircraft_tails) > 0, "No aircraft loaded"
    assert data.config is not None, "Config not loaded"
    assert data.config.max_c_check > 0, "Invalid max_c_check"
    
    print(f"  Loaded {len(data.aircraft_tails)} aircraft")
    print(f"  Planning horizon: {data.config.total_years} years")
    return True


def test_state_creation():
    """Test that initial state creation works correctly."""
    print("Testing state creation...")
    
    from aircraft_scheduler import load_maintenance_data, create_initial_fleet_state
    
    data_file = Path("Sep 25 2017 (Case Study 2018-2021).xlsx")
    if not data_file.exists():
        print(f"  Skipping: Data file not found: {data_file}")
        return True
    
    data = load_maintenance_data(str(data_file))
    fleet = create_initial_fleet_state(data)
    
    assert len(fleet.aircraft) == len(data.aircraft_tails), "Aircraft count mismatch"
    assert fleet.day_index == 0, "Day index should be 0"
    assert fleet.current_date == data.config.begin_day, "Start date mismatch"
    
    # Test first aircraft state
    first_tail = data.aircraft_tails[0]
    ac = fleet.aircraft[first_tail]
    assert ac.tail == first_tail, "Tail mismatch"
    assert ac.c_ci_dy > 0, "Invalid C-check interval"
    
    print(f"  Created fleet state with {len(fleet.aircraft)} aircraft")
    return True


def test_greedy_scheduler():
    """Test that the greedy scheduler runs correctly."""
    print("Testing Greedy scheduler...")
    
    from aircraft_scheduler import load_maintenance_data, GreedyScheduler
    
    data_file = Path("Sep 25 2017 (Case Study 2018-2021).xlsx")
    if not data_file.exists():
        print(f"  Skipping: Data file not found: {data_file}")
        return True
    
    data = load_maintenance_data(str(data_file))
    scheduler = GreedyScheduler(data)
    
    # Run for a short period
    result = scheduler.schedule(30, verbose=False)
    
    assert result is not None, "Scheduler returned None"
    assert isinstance(result.checks, list), "Checks should be a list"
    assert result.computation_time >= 0, "Invalid computation time"
    
    print(f"  Completed: {len(result.checks)} checks scheduled in {result.computation_time:.2f}s")
    return True


def test_dp_scheduler():
    """Test that the DP scheduler runs correctly."""
    print("Testing DP scheduler...")
    
    from aircraft_scheduler import load_maintenance_data, DPScheduler
    
    data_file = Path("Sep 25 2017 (Case Study 2018-2021).xlsx")
    if not data_file.exists():
        print(f"  Skipping: Data file not found: {data_file}")
        return True
    
    data = load_maintenance_data(str(data_file))
    scheduler = DPScheduler(data)
    
    # Run for a short period
    result = scheduler.schedule(30, verbose=False)
    
    assert result is not None, "Scheduler returned None"
    assert isinstance(result.checks, list), "Checks should be a list"
    assert result.computation_time >= 0, "Invalid computation time"
    
    print(f"  Completed: {len(result.checks)} checks scheduled in {result.computation_time:.2f}s")
    return True


def test_optimized_dp_scheduler():
    """Test that the Optimized DP scheduler runs correctly."""
    print("Testing Optimized DP scheduler...")
    
    from aircraft_scheduler import load_maintenance_data, OptimizedDPScheduler
    
    data_file = Path("Sep 25 2017 (Case Study 2018-2021).xlsx")
    if not data_file.exists():
        print(f"  Skipping: Data file not found: {data_file}")
        return True
    
    data = load_maintenance_data(str(data_file))
    scheduler = OptimizedDPScheduler(data)
    
    # Run for a short period
    result = scheduler.schedule(30, verbose=False)
    
    assert result is not None, "Scheduler returned None"
    assert isinstance(result.checks, list), "Checks should be a list"
    assert result.computation_time >= 0, "Invalid computation time"
    
    print(f"  Completed: {len(result.checks)} checks scheduled in {result.computation_time:.2f}s")
    return True


def test_constraint_functions():
    """Test constraint checking functions."""
    print("Testing constraint functions...")
    
    from aircraft_scheduler import (
        load_maintenance_data,
        create_initial_fleet_state,
        get_available_c_slots,
        get_available_a_slots,
        validate_fleet_state,
    )
    
    data_file = Path("Sep 25 2017 (Case Study 2018-2021).xlsx")
    if not data_file.exists():
        print(f"  Skipping: Data file not found: {data_file}")
        return True
    
    data = load_maintenance_data(str(data_file))
    fleet = create_initial_fleet_state(data)
    
    # Test slot availability
    c_slots = get_available_c_slots(fleet, data, fleet.day_index)
    a_slots = get_available_a_slots(fleet, data, fleet.day_index)
    
    assert c_slots >= 0, "Invalid C-slot count"
    assert a_slots >= 0, "Invalid A-slot count"
    
    # Test fleet validation
    violations = validate_fleet_state(fleet, data)
    assert isinstance(violations, list), "Violations should be a list"
    
    print(f"  C-slots: {c_slots}, A-slots: {a_slots}, Violations: {len(violations)}")
    return True


def test_cost_functions():
    """Test cost calculation functions."""
    print("Testing cost functions...")
    
    from aircraft_scheduler import (
        load_maintenance_data,
        create_initial_fleet_state,
        calculate_a_check_wasted_interval,
        calculate_c_check_wasted_interval,
        calculate_urgency_score,
    )
    
    data_file = Path("Sep 25 2017 (Case Study 2018-2021).xlsx")
    if not data_file.exists():
        print(f"  Skipping: Data file not found: {data_file}")
        return True
    
    data = load_maintenance_data(str(data_file))
    fleet = create_initial_fleet_state(data)
    
    # Test cost functions on first aircraft
    first_tail = data.aircraft_tails[0]
    state = fleet.aircraft[first_tail]
    
    a_cost = calculate_a_check_wasted_interval(state, data.config)
    c_cost = calculate_c_check_wasted_interval(state, data.config)
    urgency = calculate_urgency_score(state, data.config)
    
    assert a_cost.total >= 0, "Invalid A-check cost"
    assert c_cost.total >= 0, "Invalid C-check cost"
    assert urgency >= 0, "Invalid urgency score"
    
    print(f"  A-cost: {a_cost.total:.1f}, C-cost: {c_cost.total:.1f}, Urgency: {urgency:.3f}")
    return True


def run_all_tests():
    """Run all regression tests."""
    print("=" * 60)
    print("Aircraft Scheduler Regression Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_imports,
        test_data_loading,
        test_state_creation,
        test_constraint_functions,
        test_cost_functions,
        test_greedy_scheduler,
        test_dp_scheduler,
        test_optimized_dp_scheduler,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  FAILED: {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  ERROR in {test.__name__}: {e}")
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)


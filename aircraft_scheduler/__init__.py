"""
Aircraft Maintenance Check Scheduling Optimization Package.

A practical dynamic programming based methodology for aircraft maintenance
check scheduling optimization, based on Deng et al. (2019).
"""

from .data_loader import (
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
)

from .core import (
    # State classes
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
    get_next_c_check_code,
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
    is_c_check_peak_season,
    is_public_holiday,
    validate_fleet_state,
    # Cost
    CheckCost,
    calculate_a_check_wasted_interval,
    calculate_c_check_wasted_interval,
    calculate_urgency_score,
    calculate_overdue_penalty,
    calculate_scheduling_benefit,
    calculate_fleet_cost,
    estimate_future_checks,
)

from .solver import (
    ScheduledCheck,
    ScheduleResult,
    GreedyScheduler,
    DPScheduler,
    OptimizedDPScheduler,
    compare_schedulers,
)

from .visualization import (
    plot_gantt_chart,
    plot_check_distribution,
    plot_capacity_utilization,
    plot_wasted_interval_analysis,
    plot_fleet_timeline,
    generate_report,
)

__version__ = "1.0.0"
__all__ = [
    # Data loading
    "MaintenanceData",
    "load_maintenance_data",
    "AircraftDInitial",
    "AircraftCInitial",
    "AircraftAInitial",
    "FlightRates",
    "CCheckInfo",
    "Config",
    "DateRange",
    "PeakSeason",
    "ReservedMaintenance",
    "AdditionalSlots",
    # State
    "AircraftState",
    "FleetState",
    "create_initial_fleet_state",
    # Dynamics
    "age_aircraft",
    "advance_fleet_one_day",
    "schedule_a_check",
    "schedule_c_check",
    "perform_a_check",
    "perform_c_check",
    "get_daily_fh",
    "get_daily_fc",
    "get_c_check_elapsed_time",
    "get_next_c_check_code",
    # Constraints
    "ConstraintViolation",
    "can_schedule_c_check",
    "can_schedule_a_check",
    "get_available_c_slots",
    "get_available_a_slots",
    "get_feasible_c_check_aircraft",
    "get_feasible_a_check_aircraft",
    "get_mandatory_c_check_aircraft",
    "get_mandatory_a_check_aircraft",
    "is_c_check_not_allowed",
    "is_a_check_not_allowed",
    "is_c_check_peak_season",
    "is_public_holiday",
    "validate_fleet_state",
    # Cost
    "CheckCost",
    "calculate_a_check_wasted_interval",
    "calculate_c_check_wasted_interval",
    "calculate_urgency_score",
    "calculate_overdue_penalty",
    "calculate_scheduling_benefit",
    "calculate_fleet_cost",
    "estimate_future_checks",
    # Solver
    "ScheduledCheck",
    "ScheduleResult",
    "GreedyScheduler",
    "DPScheduler",
    "OptimizedDPScheduler",
    "compare_schedulers",
    # Visualization
    "plot_gantt_chart",
    "plot_check_distribution",
    "plot_capacity_utilization",
    "plot_wasted_interval_analysis",
    "plot_fleet_timeline",
    "generate_report",
]


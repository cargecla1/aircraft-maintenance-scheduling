#!/usr/bin/env python3
"""
Faithful reproduction of Deng et al. (2019) for aircraft maintenance scheduling.
"""

import argparse
import sys
from pathlib import Path

from aircraft_scheduler import (
    MaintenanceData,
    ScheduleResult,
    DengDPScheduler,
    load_maintenance_data,
    plot_gantt_chart,
    plot_check_distribution,
    plot_capacity_utilization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forward induction (Deng et al. 2019) maintenance scheduler",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="Sep 25 2017 (Case Study 2018-2021).xlsx",
        help="Path to the Excel data file.",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="Planning horizon in years (default: value from data file).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Directory where artifacts (CSV/plots) will be stored.",
    )
    parser.add_argument(
        "--discretization",
        type=int,
        default=100,
        help="Utilization bin size u for histogram aggregation (paper parameter).",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=500,
        help="Maximum number of states kept in the frontier (beam size).",
    )
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=0,
        help="Workability lookahead horizon (days). 0=fast mode (no simulation).",
    )
    parser.add_argument(
        "--a-threshold",
        type=int,
        default=21,
        help="Remaining-days threshold for A-check priority (R_A in the paper).",
    )
    parser.add_argument(
        "--c-threshold",
        type=int,
        default=365,
        help="Remaining-days threshold for C-check priority (R_C in the paper).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="How often (in days) to print progress updates when --verbose is set.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generation of visualization assets.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information while the scheduler runs.",
    )
    return parser.parse_args()


def print_header() -> None:
    print("=" * 70)
    print("Aircraft Maintenance Check Scheduling (Deng et al. 2019)")
    print("=" * 70)
    print()


def print_data_summary(data: MaintenanceData) -> None:
    print("DATA SUMMARY")
    print("-" * 40)
    print(f"Aircraft count          : {len(data.aircraft_tails)}")
    print(f"Planning start date     : {data.config.begin_day}")
    print(f"Configured horizon      : {data.config.total_years} years")
    print(f"C-check capacity        : {data.config.max_c_check}")
    print(f"A-check capacity        : {data.config.max_a_check}")
    print(f"Min days between C-start: {data.config.start_day_interval}")
    print()


def build_scheduler(data: MaintenanceData, args: argparse.Namespace) -> DengDPScheduler:
    return DengDPScheduler(
        data,
        u=args.discretization,
        max_frontier_size=args.beam_width,
        a_threshold_days=args.a_threshold,
        c_threshold_days=args.c_threshold,
        workability_lookahead=args.lookahead_days,
        progress_interval=args.progress_interval,
    )


def print_result_summary(
    result: ScheduleResult,
    states_visited: int,
    states_pruned: int,
) -> None:
    print("\nDENG DP RESULT")
    print("-" * 40)
    print(result.summary())
    print(f"States expanded: {states_visited:,}")
    print(f"States pruned  : {states_pruned:,}")
    print()


def print_table3_kpis(result: ScheduleResult) -> None:
    kpis = result.compute_kpis()
    print("Table 3 KPIs (paper-compatible)")
    print("-" * 40)
    print(f"A-checks: {kpis.n_a_checks} "
          f"| Avg FH: {kpis.avg_fh_at_a_checks:.1f} "
          f"| Std FH: {kpis.std_fh_at_a_checks:.1f} "
          f"| Avg unused FH: {kpis.avg_unused_fh_a:.1f}")
    print(f"C-checks: {kpis.n_c_checks} "
          f"| Avg FH: {kpis.avg_fh_at_c_checks:.1f} "
          f"| Std FH: {kpis.std_fh_at_c_checks:.1f} "
          f"| Avg unused FH: {kpis.avg_unused_fh_c:.1f}")
    print(f"D-checks: {kpis.n_d_checks}")
    print(f"Total unused FH : {kpis.total_unused_fh:.1f}")
    print(f"Tolerance events: {kpis.total_tolerance_events}")
    print()


def save_schedule_csv(
    result: ScheduleResult,
    output_path: Path,
    data: MaintenanceData,
) -> None:
    import csv

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Aircraft",
            "Fleet",
            "Check Type",
            "Start Date",
            "End Date",
            "Duration (days)",
            "Unused FH (Eq. 28)",
        ])

        for check in sorted(result.checks, key=lambda c: (c.start_date, c.tail)):
            fleet = data.d_initial[check.tail].fleet
            end_date = check.end_date or check.start_date
            unused_fh = (
                check.total_unused_fh
                if check.check_type in ("C", "D")
                else check.unused_fh
            )
            writer.writerow([
                check.tail,
                fleet,
                check.check_type,
                check.start_date.isoformat(),
                end_date.isoformat(),
                check.duration,
                f"{unused_fh:.1f}",
            ])

    print(f"Schedule saved to: {output_path}")


def generate_visualizations(
    result: ScheduleResult,
    data: MaintenanceData,
    output_dir: Path,
) -> None:
    print("\nGenerating visualizations...")
    try:
        plot_gantt_chart(result, data, str(output_dir / "gantt.png"))
        plot_check_distribution(result, data, str(output_dir / "distribution.png"))
        plot_capacity_utilization(result, data, str(output_dir / "capacity.png"))
    except Exception as exc:  # pragma: no cover - visualization failures are non-fatal
        print(f"Warning: could not generate visualizations ({exc})")


def main() -> None:
    args = parse_args()
    print_header()

    print(f"Loading data from: {args.data}")
    try:
        data = load_maintenance_data(args.data)
    except FileNotFoundError:
        print(f"Error: data file not found: {args.data}")
        sys.exit(1)
    except Exception as exc:
        print(f"Error loading data: {exc}")
        sys.exit(1)

    print_data_summary(data)

    planning_years = args.years if args.years else data.config.total_years
    planning_days = planning_years * 365
    print(f"Planning horizon: {planning_years} years ({planning_days} days)\n")

    scheduler = build_scheduler(data, args)
    print("Running Deng DP scheduler...\n")
    result = scheduler.schedule(planning_days, verbose=args.verbose)

    print_result_summary(result, scheduler.states_visited, scheduler.states_pruned)
    print_table3_kpis(result)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_schedule_csv(result, output_dir / "schedule_deng.csv", data)

    if not args.skip_plots:
        generate_visualizations(result, data, output_dir)

    print("Done!")
    print(f"Artifacts written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()


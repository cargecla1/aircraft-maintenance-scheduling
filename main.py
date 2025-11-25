#!/usr/bin/env python3
"""
Aircraft Maintenance Check Scheduling Optimization

A practical dynamic programming based methodology for aircraft maintenance
check scheduling optimization, based on Deng et al. (2019).

Usage:
    python main.py [options]

Options:
    --data PATH       Path to Excel data file (default: Sep 25 2017 (Case Study 2018-2021).xlsx)
    --years N         Planning horizon in years (default: from data file)
    --algorithm ALG   Algorithm to use: greedy, dp, optimized (default: dp)
    --output PATH     Output directory for results (default: output/)
    --compare         Compare all algorithms
    --verbose         Print progress information
    --help            Show this help message
"""

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from data_loader import load_maintenance_data, MaintenanceData
from solver import (
    GreedyScheduler,
    DPScheduler,
    OptimizedDPScheduler,
    ScheduleResult,
    compare_schedulers,
)
from visualization import (
    plot_gantt_chart,
    plot_check_distribution,
    plot_capacity_utilization,
    generate_report,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Aircraft Maintenance Check Scheduling Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data",
        type=str,
        default="Sep 25 2017 (Case Study 2018-2021).xlsx",
        help="Path to Excel data file",
    )

    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="Planning horizon in years (default: from data file)",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["greedy", "dp", "optimized"],
        default="dp",
        help="Algorithm to use (default: dp)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for results",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all algorithms",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information",
    )

    return parser.parse_args()


def print_header():
    """Print program header."""
    print("=" * 60)
    print("Aircraft Maintenance Check Scheduling Optimization")
    print("Based on Deng et al. (2019)")
    print("=" * 60)
    print()


def print_data_summary(data: MaintenanceData):
    """Print summary of loaded data."""
    print("Data Summary")
    print("-" * 40)
    print(f"Number of aircraft: {len(data.aircraft_tails)}")
    print(f"Planning start date: {data.config.begin_day}")
    print(f"Planning horizon: {data.config.total_years} years")
    print(f"C-check hangar capacity: {data.config.max_c_check}")
    print(f"A-check hangar capacity: {data.config.max_a_check}")
    print(f"Min days between C-check starts: {data.config.start_day_interval}")
    print()

    # Count aircraft by fleet
    fleet_counts = {}
    for tail in data.aircraft_tails:
        fleet = data.d_initial[tail].fleet
        fleet_counts[fleet] = fleet_counts.get(fleet, 0) + 1

    print("Fleet composition:")
    for fleet, count in sorted(fleet_counts.items()):
        print(f"  {fleet}: {count} aircraft")
    print()


def print_result_summary(result: ScheduleResult, algorithm: str):
    """Print detailed result summary."""
    print(f"\n{algorithm.upper()} Algorithm Results")
    print("=" * 40)
    print(result.summary())
    print()

    # Additional statistics
    if result.checks:
        a_checks = [c for c in result.checks if c.check_type == "A"]
        c_checks = [c for c in result.checks if c.check_type == "C"]
        d_checks = [c for c in result.checks if c.check_type == "D"]

        if a_checks:
            avg_a_wasted = sum(c.wasted_interval for c in a_checks) / len(a_checks)
            print(f"Average A-check wasted interval: {avg_a_wasted:.1f}")

        if c_checks:
            avg_c_wasted = sum(c.wasted_interval for c in c_checks) / len(c_checks)
            print(f"Average C-check wasted interval: {avg_c_wasted:.1f}")

        if d_checks:
            avg_d_wasted = sum(c.wasted_interval for c in d_checks) / len(d_checks)
            print(f"Average D-check wasted interval: {avg_d_wasted:.1f}")


def save_schedule_csv(result: ScheduleResult, output_path: str, data: MaintenanceData):
    """Save schedule to CSV file."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Aircraft",
            "Fleet",
            "Check Type",
            "Start Date",
            "End Date",
            "Duration (days)",
            "Wasted Interval",
        ])

        for check in sorted(result.checks, key=lambda c: (c.start_date, c.tail)):
            fleet = data.d_initial[check.tail].fleet
            end_date = check.end_date or check.start_date
            writer.writerow([
                check.tail,
                fleet,
                check.check_type,
                check.start_date.isoformat(),
                end_date.isoformat(),
                check.duration,
                f"{check.wasted_interval:.1f}",
            ])

    print(f"Schedule saved to: {output_path}")


def run_single_algorithm(
    data: MaintenanceData,
    algorithm: str,
    planning_days: int,
    verbose: bool,
) -> ScheduleResult:
    """Run a single scheduling algorithm."""
    if algorithm == "greedy":
        scheduler = GreedyScheduler(data)
    elif algorithm == "dp":
        scheduler = DPScheduler(data)
    elif algorithm == "optimized":
        scheduler = OptimizedDPScheduler(data)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print(f"Running {algorithm} algorithm...")
    result = scheduler.schedule(planning_days, verbose=verbose)
    return result


def main():
    """Main entry point."""
    args = parse_args()

    print_header()

    # Load data
    print(f"Loading data from: {args.data}")
    try:
        data = load_maintenance_data(args.data)
    except FileNotFoundError:
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print_data_summary(data)

    # Determine planning horizon
    if args.years:
        planning_years = args.years
    else:
        planning_years = data.config.total_years

    planning_days = planning_years * 365
    print(f"Planning horizon: {planning_years} years ({planning_days} days)")
    print()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        # Compare all algorithms
        print("Comparing all algorithms...")
        print("=" * 60)
        results = compare_schedulers(data, planning_days, verbose=args.verbose)

        # Print comparison table
        print("\n" + "=" * 60)
        print("Algorithm Comparison")
        print("=" * 60)
        print(f"{'Algorithm':<15} {'A-checks':<10} {'C-checks':<10} {'D-checks':<10} {'Total':<10} {'Wasted':<15} {'Time':<10}")
        print("-" * 80)

        for name, result in results.items():
            total = result.total_a_checks + result.total_c_checks + result.total_d_checks
            print(f"{name:<15} {result.total_a_checks:<10} {result.total_c_checks:<10} {result.total_d_checks:<10} {total:<10} {result.total_wasted:<15.1f} {result.computation_time:<10.2f}s")

        # Save best result
        best_name = min(results.keys(), key=lambda k: results[k].total_wasted)
        best_result = results[best_name]
        print(f"\nBest algorithm: {best_name} (lowest wasted interval)")

        # Save schedules
        for name, result in results.items():
            csv_path = output_dir / f"schedule_{name}.csv"
            save_schedule_csv(result, str(csv_path), data)

        # Generate visualizations
        print("\nGenerating visualizations...")
        for name, result in results.items():
            try:
                gantt_path = output_dir / f"gantt_{name}.png"
                plot_gantt_chart(result, data, str(gantt_path))
            except Exception as e:
                print(f"Warning: Could not generate Gantt chart for {name}: {e}")

        # Generate comparison report
        report_path = output_dir / "comparison_report.txt"
        generate_report(results, data, str(report_path))

    else:
        # Run single algorithm
        result = run_single_algorithm(data, args.algorithm, planning_days, args.verbose)
        print_result_summary(result, args.algorithm)

        # Save schedule
        csv_path = output_dir / f"schedule_{args.algorithm}.csv"
        save_schedule_csv(result, str(csv_path), data)

        # Generate visualizations
        print("\nGenerating visualizations...")
        try:
            gantt_path = output_dir / f"gantt_{args.algorithm}.png"
            plot_gantt_chart(result, data, str(gantt_path))

            dist_path = output_dir / f"distribution_{args.algorithm}.png"
            plot_check_distribution(result, data, str(dist_path))

            capacity_path = output_dir / f"capacity_{args.algorithm}.png"
            plot_capacity_utilization(result, data, str(capacity_path))
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")

    print("\nDone!")
    print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()


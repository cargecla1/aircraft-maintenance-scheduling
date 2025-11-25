"""
Visualization module for Aircraft Maintenance Check Scheduling.

Generates Gantt charts, distribution plots, and reports.
"""

from datetime import date, timedelta
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np

from .data_loader import MaintenanceData
from .solver import ScheduleResult, ScheduledCheck


# Color scheme for check types
CHECK_COLORS = {
    "A": "#4CAF50",  # Green
    "C": "#2196F3",  # Blue
    "D": "#F44336",  # Red
}

# Fleet colors
FLEET_COLORS = {
    "A319": "#E91E63",
    "A320": "#9C27B0",
    "A321": "#3F51B5",
}


def plot_gantt_chart(
    result: ScheduleResult,
    data: MaintenanceData,
    output_path: str,
    max_aircraft: int = 20,
    show_legend: bool = True,
):
    """
    Generate a Gantt chart of the maintenance schedule.
    
    Args:
        result: Schedule result
        data: Maintenance data
        output_path: Path to save the plot
        max_aircraft: Maximum number of aircraft to show
        show_legend: Whether to show legend
    """
    # Group checks by aircraft
    checks_by_aircraft = defaultdict(list)
    for check in result.checks:
        checks_by_aircraft[check.tail].append(check)

    # Select aircraft to display
    aircraft_list = sorted(checks_by_aircraft.keys())[:max_aircraft]

    if not aircraft_list:
        print("No checks to visualize")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(8, len(aircraft_list) * 0.4)))

    # Get date range
    all_dates = [c.start_date for c in result.checks]
    if not all_dates:
        return

    min_date = min(all_dates)
    max_date = max(all_dates)
    for c in result.checks:
        if c.end_date:
            max_date = max(max_date, c.end_date)

    # Plot bars for each aircraft
    y_positions = {}
    for i, tail in enumerate(aircraft_list):
        y_positions[tail] = i
        checks = checks_by_aircraft[tail]

        for check in checks:
            start = check.start_date
            end = check.end_date or start
            duration = (end - start).days + 1 if end != start else 1

            color = CHECK_COLORS.get(check.check_type, "#9E9E9E")

            ax.barh(
                y=i,
                width=duration,
                left=mdates.date2num(start),
                height=0.6,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.8,
            )

    # Configure axes
    ax.set_yticks(range(len(aircraft_list)))
    ax.set_yticklabels(aircraft_list)
    ax.set_xlabel("Date")
    ax.set_ylabel("Aircraft")
    ax.set_title("Aircraft Maintenance Schedule - Gantt Chart")

    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")

    # Set x limits
    ax.set_xlim(mdates.date2num(min_date - timedelta(days=7)),
                mdates.date2num(max_date + timedelta(days=7)))

    # Add legend
    if show_legend:
        legend_patches = [
            mpatches.Patch(color=CHECK_COLORS["A"], label="A-check"),
            mpatches.Patch(color=CHECK_COLORS["C"], label="C-check"),
            mpatches.Patch(color=CHECK_COLORS["D"], label="D-check"),
        ]
        ax.legend(handles=legend_patches, loc="upper right")

    # Add grid
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Gantt chart saved to: {output_path}")


def plot_check_distribution(
    result: ScheduleResult,
    data: MaintenanceData,
    output_path: str,
):
    """
    Plot distribution of checks over time.
    
    Args:
        result: Schedule result
        data: Maintenance data
        output_path: Path to save the plot
    """
    if not result.checks:
        print("No checks to visualize")
        return

    # Group checks by month
    checks_by_month = defaultdict(lambda: {"A": 0, "C": 0, "D": 0})
    for check in result.checks:
        month_key = check.start_date.strftime("%Y-%m")
        checks_by_month[month_key][check.check_type] += 1

    # Sort months
    months = sorted(checks_by_month.keys())
    a_counts = [checks_by_month[m]["A"] for m in months]
    c_counts = [checks_by_month[m]["C"] for m in months]
    d_counts = [checks_by_month[m]["D"] for m in months]

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(months))
    width = 0.8

    ax.bar(x, a_counts, width, label="A-check", color=CHECK_COLORS["A"])
    ax.bar(x, c_counts, width, bottom=a_counts, label="C-check", color=CHECK_COLORS["C"])
    ax.bar(x, d_counts, width, bottom=[a + c for a, c in zip(a_counts, c_counts)],
           label="D-check", color=CHECK_COLORS["D"])

    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Checks")
    ax.set_title("Maintenance Check Distribution Over Time")

    # Show every 3rd month label
    ax.set_xticks(x[::3])
    ax.set_xticklabels(months[::3], rotation=45, ha="right")

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Distribution plot saved to: {output_path}")


def plot_capacity_utilization(
    result: ScheduleResult,
    data: MaintenanceData,
    output_path: str,
):
    """
    Plot hangar capacity utilization over time.
    
    Args:
        result: Schedule result
        data: Maintenance data
        output_path: Path to save the plot
    """
    if not result.checks:
        print("No checks to visualize")
        return

    # Calculate daily C-check slot usage
    c_slot_usage = defaultdict(int)
    a_slot_usage = defaultdict(int)

    for check in result.checks:
        if check.check_type in ("C", "D"):
            start = check.start_date
            end = check.end_date or start
            current = start
            while current <= end:
                c_slot_usage[current] += 1
                current += timedelta(days=1)
        elif check.check_type == "A":
            a_slot_usage[check.start_date] += 1

    if not c_slot_usage:
        print("No C-checks to visualize")
        return

    # Get date range
    all_dates = sorted(set(c_slot_usage.keys()) | set(a_slot_usage.keys()))
    min_date = min(all_dates)
    max_date = max(all_dates)

    # Create time series
    dates = []
    c_usage = []
    a_usage = []

    current = min_date
    while current <= max_date:
        dates.append(current)
        c_usage.append(c_slot_usage.get(current, 0))
        a_usage.append(a_slot_usage.get(current, 0))
        current += timedelta(days=1)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # C-check capacity
    ax1.fill_between(dates, c_usage, alpha=0.7, color=CHECK_COLORS["C"], label="C/D-check slots used")
    ax1.axhline(y=data.config.max_c_check, color="red", linestyle="--", label=f"Capacity ({data.config.max_c_check})")
    ax1.set_ylabel("C-check Slots")
    ax1.set_title("Hangar Capacity Utilization")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # A-check capacity (weekly moving average)
    window = 7
    a_usage_smooth = np.convolve(a_usage, np.ones(window)/window, mode='same')
    ax2.fill_between(dates, a_usage_smooth, alpha=0.7, color=CHECK_COLORS["A"], label="A-check slots (7-day avg)")
    ax2.axhline(y=data.config.max_a_check, color="red", linestyle="--", label=f"Capacity ({data.config.max_a_check})")
    ax2.set_ylabel("A-check Slots")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Capacity utilization plot saved to: {output_path}")


def plot_wasted_interval_analysis(
    result: ScheduleResult,
    data: MaintenanceData,
    output_path: str,
):
    """
    Plot wasted interval analysis.
    
    Args:
        result: Schedule result
        data: Maintenance data
        output_path: Path to save the plot
    """
    if not result.checks:
        print("No checks to visualize")
        return

    # Separate checks by type
    a_wasted = [c.wasted_interval for c in result.checks if c.check_type == "A"]
    c_wasted = [c.wasted_interval for c in result.checks if c.check_type == "C"]
    d_wasted = [c.wasted_interval for c in result.checks if c.check_type == "D"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # A-check histogram
    if a_wasted:
        axes[0].hist(a_wasted, bins=20, color=CHECK_COLORS["A"], edgecolor="black", alpha=0.7)
        axes[0].axvline(np.mean(a_wasted), color="red", linestyle="--", label=f"Mean: {np.mean(a_wasted):.1f}")
        axes[0].set_title(f"A-check Wasted Interval\n(n={len(a_wasted)})")
        axes[0].set_xlabel("Wasted Interval")
        axes[0].set_ylabel("Count")
        axes[0].legend()

    # C-check histogram
    if c_wasted:
        axes[1].hist(c_wasted, bins=20, color=CHECK_COLORS["C"], edgecolor="black", alpha=0.7)
        axes[1].axvline(np.mean(c_wasted), color="red", linestyle="--", label=f"Mean: {np.mean(c_wasted):.1f}")
        axes[1].set_title(f"C-check Wasted Interval\n(n={len(c_wasted)})")
        axes[1].set_xlabel("Wasted Interval")
        axes[1].set_ylabel("Count")
        axes[1].legend()

    # D-check histogram
    if d_wasted:
        axes[2].hist(d_wasted, bins=20, color=CHECK_COLORS["D"], edgecolor="black", alpha=0.7)
        axes[2].axvline(np.mean(d_wasted), color="red", linestyle="--", label=f"Mean: {np.mean(d_wasted):.1f}")
        axes[2].set_title(f"D-check Wasted Interval\n(n={len(d_wasted)})")
        axes[2].set_xlabel("Wasted Interval")
        axes[2].set_ylabel("Count")
        axes[2].legend()

    plt.suptitle("Wasted Interval Distribution by Check Type")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Wasted interval analysis saved to: {output_path}")


def generate_report(
    results: dict[str, ScheduleResult],
    data: MaintenanceData,
    output_path: str,
):
    """
    Generate a text report comparing algorithms.
    
    Args:
        results: Dictionary of algorithm name to result
        data: Maintenance data
        output_path: Path to save the report
    """
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("AIRCRAFT MAINTENANCE CHECK SCHEDULING OPTIMIZATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("DATA SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of aircraft: {len(data.aircraft_tails)}\n")
        f.write(f"Planning start date: {data.config.begin_day}\n")
        f.write(f"Planning horizon: {data.config.total_years} years\n")
        f.write(f"C-check capacity: {data.config.max_c_check}\n")
        f.write(f"A-check capacity: {data.config.max_a_check}\n\n")

        f.write("ALGORITHM COMPARISON\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Algorithm':<15} {'A-checks':<10} {'C-checks':<10} {'D-checks':<10} {'Wasted':<15} {'Time':<10}\n")
        f.write("-" * 70 + "\n")

        for name, result in results.items():
            f.write(f"{name:<15} {result.total_a_checks:<10} {result.total_c_checks:<10} "
                   f"{result.total_d_checks:<10} {result.total_wasted:<15.1f} {result.computation_time:<10.2f}s\n")

        # Find best algorithm
        best_name = min(results.keys(), key=lambda k: results[k].total_wasted)
        baseline = results.get("greedy", list(results.values())[0])
        best = results[best_name]

        f.write("\n" + "=" * 70 + "\n")
        f.write("OPTIMIZATION RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best algorithm: {best_name}\n\n")

        if baseline != best:
            wasted_reduction = baseline.total_wasted - best.total_wasted
            wasted_pct = (wasted_reduction / baseline.total_wasted) * 100 if baseline.total_wasted > 0 else 0
            total_baseline = baseline.total_a_checks + baseline.total_c_checks + baseline.total_d_checks
            total_best = best.total_a_checks + best.total_c_checks + best.total_d_checks
            check_reduction = total_baseline - total_best
            check_pct = (check_reduction / total_baseline) * 100 if total_baseline > 0 else 0

            f.write(f"Wasted interval reduction: {wasted_reduction:.1f} ({wasted_pct:.1f}%)\n")
            f.write(f"Total check difference: {check_reduction} ({check_pct:.1f}%)\n\n")

        # Detailed breakdown for best algorithm
        f.write("DETAILED BREAKDOWN (Best Algorithm)\n")
        f.write("-" * 40 + "\n")

        a_checks = [c for c in best.checks if c.check_type == "A"]
        c_checks = [c for c in best.checks if c.check_type == "C"]
        d_checks = [c for c in best.checks if c.check_type == "D"]

        if a_checks:
            avg_a_wasted = sum(c.wasted_interval for c in a_checks) / len(a_checks)
            f.write(f"A-checks: {len(a_checks)}, Avg wasted: {avg_a_wasted:.1f}\n")

        if c_checks:
            avg_c_wasted = sum(c.wasted_interval for c in c_checks) / len(c_checks)
            f.write(f"C-checks: {len(c_checks)}, Avg wasted: {avg_c_wasted:.1f}\n")

        if d_checks:
            avg_d_wasted = sum(c.wasted_interval for c in d_checks) / len(d_checks)
            f.write(f"D-checks: {len(d_checks)}, Avg wasted: {avg_d_wasted:.1f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")

    print(f"Report saved to: {output_path}")


def plot_fleet_timeline(
    result: ScheduleResult,
    data: MaintenanceData,
    output_path: str,
    aircraft_subset: Optional[list[str]] = None,
):
    """
    Plot timeline for specific aircraft or fleet type.
    
    Args:
        result: Schedule result
        data: Maintenance data
        output_path: Path to save the plot
        aircraft_subset: Optional list of aircraft to include
    """
    if aircraft_subset is None:
        # Show first 10 aircraft
        aircraft_subset = data.aircraft_tails[:10]

    checks_by_aircraft = defaultdict(list)
    for check in result.checks:
        if check.tail in aircraft_subset:
            checks_by_aircraft[check.tail].append(check)

    if not checks_by_aircraft:
        print("No checks found for specified aircraft")
        return

    fig, ax = plt.subplots(figsize=(16, len(aircraft_subset) * 0.6 + 2))

    y_labels = []
    for i, tail in enumerate(aircraft_subset):
        fleet = data.d_initial[tail].fleet
        y_labels.append(f"{tail} ({fleet})")

        for check in checks_by_aircraft[tail]:
            start = check.start_date
            end = check.end_date or start
            duration = (end - start).days + 1

            color = CHECK_COLORS.get(check.check_type, "#9E9E9E")

            ax.barh(
                y=i,
                width=duration,
                left=mdates.date2num(start),
                height=0.5,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.8,
            )

            # Add check type label for C and D checks
            if check.check_type in ("C", "D") and duration >= 5:
                mid_date = mdates.date2num(start) + duration / 2
                ax.text(mid_date, i, check.check_type, ha="center", va="center",
                       fontsize=8, color="white", fontweight="bold")

    ax.set_yticks(range(len(aircraft_subset)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Date")
    ax.set_title("Aircraft Maintenance Timeline")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha="right")

    legend_patches = [
        mpatches.Patch(color=CHECK_COLORS["A"], label="A-check"),
        mpatches.Patch(color=CHECK_COLORS["C"], label="C-check"),
        mpatches.Patch(color=CHECK_COLORS["D"], label="D-check"),
    ]
    ax.legend(handles=legend_patches, loc="upper right")

    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Fleet timeline saved to: {output_path}")


if __name__ == "__main__":
    from .data_loader import load_maintenance_data
    from .solver import DPScheduler

    print("Loading data...")
    data = load_maintenance_data("../Sep 25 2017 (Case Study 2018-2021).xlsx")

    print("Running DP scheduler...")
    scheduler = DPScheduler(data)
    result = scheduler.schedule(365, verbose=False)

    print("\nGenerating visualizations...")
    plot_gantt_chart(result, data, "gantt_test.png")
    plot_check_distribution(result, data, "distribution_test.png")
    plot_capacity_utilization(result, data, "capacity_test.png")
    plot_wasted_interval_analysis(result, data, "wasted_test.png")

    print("\nDone!")


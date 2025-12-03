#!/usr/bin/env python3
"""
Aircraft Maintenance Check Scheduling Optimizer

Faithful reproduction of Deng et al. (2020):
"A practical dynamic programming based methodology for aircraft 
maintenance check scheduling optimization"
European Journal of Operational Research, 281(2), 256-273.
"""

import argparse
import sys
from pathlib import Path

from aircraft_scheduler import (
    Scheduler,
    load_maintenance_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aircraft Maintenance Check Scheduler (Deng et al. 2020)",
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
        help="Planning horizon in years (default: from data file).",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=200,
        help="Beam width for search (higher = slower but better).",
    )
    parser.add_argument(
        "--discretization",
        type=float,
        default=100.0,
        help="Histogram bin size (paper's u parameter).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Aircraft Maintenance Check Scheduler")
    print("Deng et al. (2020) - EJOR")
    print("=" * 60)
    print()

    # Load data
    print(f"Loading: {args.data}")
    try:
        data = load_maintenance_data(args.data)
    except FileNotFoundError:
        print(f"Error: file not found: {args.data}")
        sys.exit(1)
    
    print(f"Fleet: {len(data.aircraft_tails)} aircraft")
    
    # Determine planning horizon
    planning_years = args.years if args.years else data.config.total_years
    planning_days = planning_years * 365
    print(f"Horizon: {planning_years} years ({planning_days} days)")
    print()

    # Run scheduler
    scheduler = Scheduler(
        data,
        beam_width=args.beam_width,
        u=args.discretization,
    )
    
    result = scheduler.schedule(planning_days, verbose=not args.quiet)

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"C-checks: {result['n_c_checks']}")
    print(f"A-checks: {result['n_a_checks']}")
    print(f"Tolerance events: {result['tolerance_events']}")
    print(f"Total cost: {result['total_cost']:.0f}")
    print(f"Computation time: {result['computation_time']:.1f}s")


if __name__ == "__main__":
    main()

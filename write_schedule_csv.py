#!/usr/bin/env python3
"""
Helper script to write ScheduleResult to CSV format for poster visualizations.

Usage:
    python write_schedule_csv.py --result <pickle_file> --output output/schedule_deng.csv
    OR
    python write_schedule_csv.py --run --years 4 --output output/schedule_deng.csv
"""

import argparse
import sys
import pickle
import pandas as pd
from pathlib import Path

from aircraft_scheduler import Scheduler, load_maintenance_data, ScheduleResult


def result_to_csv(result: ScheduleResult, output_path: str):
    """Convert ScheduleResult to CSV format matching poster_visualizations expectations."""
    rows = []
    for check in result.checks:
        rows.append({
            'Aircraft': check.tail,
            'Fleet': '',  # Would need to get from data
            'Check Type': check.check_type,
            'Start Date': check.start_date.strftime('%Y-%m-%d'),
            'End Date': check.end_date.strftime('%Y-%m-%d') if check.end_date else check.start_date.strftime('%Y-%m-%d'),
            'Duration (days)': check.duration,
            'Unused FH (Eq. 28)': check.total_unused_fh,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} checks to {output_path}")
    print(f"Date range: {df['Start Date'].min()} to {df['Start Date'].max()}")


def run_and_save(years: int, output_path: str, data_file: str = None, beam_width: int = 200):
    """Run scheduler and save results to CSV."""
    if data_file is None:
        data_file = "Sep 25 2017 (Case Study 2018-2021).xlsx"
    
    print(f"Loading data from {data_file}...")
    data = load_maintenance_data(data_file)
    
    planning_days = years * 365
    print(f"Running scheduler for {years} years ({planning_days} days)...")
    
    scheduler = Scheduler(data, beam_width=beam_width, u=100.0)
    result = scheduler.schedule(planning_days, verbose=True)
    
    print(f"\nResults:")
    print(f"  C-checks: {result.total_c_checks}")
    print(f"  A-checks: {result.total_a_checks}")
    print(f"  Total unused FH: {result.total_unused_fh:.1f}")
    print(f"  Computation time: {result.computation_time:.1f}s")
    
    # Save to CSV
    result_to_csv(result, output_path)
    
    # Also save result object for future use
    result_path = Path(output_path).with_suffix('.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved result object to {result_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Write ScheduleResult to CSV for poster visualizations"
    )
    parser.add_argument(
        '--result',
        type=str,
        help='Path to pickled ScheduleResult file'
    )
    parser.add_argument(
        '--run',
        action='store_true',
        help='Run scheduler instead of loading result'
    )
    parser.add_argument(
        '--years',
        type=int,
        default=4,
        help='Planning horizon in years (default: 4)'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to Excel data file'
    )
    parser.add_argument(
        '--beam-width',
        type=int,
        default=200,
        help='Beam width for scheduler (default: 200)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/schedule_deng.csv',
        help='Output CSV path (default: output/schedule_deng.csv)'
    )
    
    args = parser.parse_args()
    
    if args.run:
        run_and_save(args.years, args.output, args.data, args.beam_width)
    elif args.result:
        print(f"Loading result from {args.result}...")
        with open(args.result, 'rb') as f:
            result = pickle.load(f)
        result_to_csv(result, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()


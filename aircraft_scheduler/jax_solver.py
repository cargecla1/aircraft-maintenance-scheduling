"""
Aircraft Maintenance Check Scheduling Optimizer

A faithful reproduction of:
    Deng, Q., Santos, B.F., Curran, R. (2020). "A practical dynamic programming 
    based methodology for aircraft maintenance check scheduling optimization."
    European Journal of Operational Research, 281(2), 256-273.

IMPORTANT: Despite the paper's title, the algorithm is NOT Bellman DP.
It is forward beam search with state aggregation and dominance pruning.

Key Paper Elements Implemented:
-------------------------------
1. COST FUNCTION (Equation 28):
   C_t = Σ[ χ·(I^FH - FH) + (1-χ)·P_a·θ + P_d·η ]
   
   Where:
   - χ = 1 if check performed, 0 otherwise
   - I^FH - FH = unused flight hours (primary objective)
   - P_a = tolerance penalty (check done past interval, requires approval)
   - θ = 1 if check uses tolerance, 0 otherwise
   - P_d = ground-waiting penalty (aircraft overdue, waiting for slot)
   - η = 1 if ground-waiting, 0 otherwise

2. SEVEN DAILY ACTIONS (Table 2):
   (n_a, n_c) ∈ {(0,0), (1,0), (0,1), (1,1), (0,2), (1,2), (2,0)}
   
3. STATE AGGREGATION (Section 4.4):
   States aggregated by utilization histogram to reduce state space.
   
4. THRIFTY PRUNING (Section 4.4.1):
   Keep only lowest-cost state per histogram bucket.

5. PRIORITY ORDERING (Equations 36-40):
   Earliest deadline first for aircraft selection.

Usage:
------
    from aircraft_scheduler import Scheduler, load_maintenance_data
    
    data = load_maintenance_data("data.xlsx")
    scheduler = Scheduler(data, beam_width=200)
    result = scheduler.schedule(4 * 365, verbose=True)
"""

import jax.numpy as jnp
from jax import jit
from functools import partial
from collections import defaultdict
import numpy as np
import time
import sys

from .data_loader import MaintenanceData


# =============================================================================
# PAPER'S 7 DAILY ACTIONS (Table 2)
# =============================================================================
# Format: (n_a, n_c) = number of A-checks, number of C-checks
PAPER_ACTIONS = [
    (0, 0),  # Do nothing
    (1, 0),  # 1 A-check only
    (0, 1),  # 1 C-check only
    (1, 1),  # 1 A-check + 1 C-check
    (0, 2),  # 2 C-checks
    (1, 2),  # 1 A-check + 2 C-checks
    (2, 0),  # 2 A-checks (capped to capacity)
]


# =============================================================================
# STATE REPRESENTATION
# =============================================================================

def create_fleet_state(data: MaintenanceData) -> dict:
    """
    Create initial fleet state from maintenance data.
    
    State variables per aircraft (paper's Table 1):
    - dy_c, fh_c, fc_c: Days/FH/FC since last C-check
    - dy_a, fh_a, fc_a: Days/FH/FC since last A-check
    - c_ci_dy/fh/fc: C-check interval in DY/FH/FC
    - a_ci_dy/fh/fc: A-check interval in DY/FH/FC
    - grounded_days: Days remaining in maintenance
    """
    n = len(data.aircraft_tails)
    
    state = {
        # C-check utilization (days/FH/FC since last check)
        'dy_c': np.array([data.c_initial[t].dy_c for t in data.aircraft_tails], dtype=np.float32),
        'fh_c': np.array([data.c_initial[t].fh_c for t in data.aircraft_tails], dtype=np.float32),
        'fc_c': np.array([data.c_initial[t].fc_c for t in data.aircraft_tails], dtype=np.float32),
        
        # A-check utilization
        'dy_a': np.array([data.a_initial[t].dy_a for t in data.aircraft_tails], dtype=np.float32),
        'fh_a': np.array([data.a_initial[t].fh_a for t in data.aircraft_tails], dtype=np.float32),
        'fc_a': np.array([data.a_initial[t].fc_a for t in data.aircraft_tails], dtype=np.float32),
        
        # Grounding state
        'grounded_days': np.zeros(n, dtype=np.float32),
    }
    
    return state


def create_static_params(data: MaintenanceData) -> dict:
    """
    Create static parameters (intervals, rates, etc.) that don't change.
    """
    n = len(data.aircraft_tails)
    
    static = {
        # C-check intervals
        'c_ci_dy': np.array([data.c_initial[t].c_ci_dy for t in data.aircraft_tails], dtype=np.float32),
        'c_ci_fh': np.array([data.c_initial[t].c_ci_fh for t in data.aircraft_tails], dtype=np.float32),
        'c_ci_fc': np.array([data.c_initial[t].c_ci_fc for t in data.aircraft_tails], dtype=np.float32),
        
        # A-check intervals
        'a_ci_dy': np.array([data.a_initial[t].a_ci_dy for t in data.aircraft_tails], dtype=np.float32),
        'a_ci_fh': np.array([data.a_initial[t].a_ci_fh for t in data.aircraft_tails], dtype=np.float32),
        'a_ci_fc': np.array([data.a_initial[t].a_ci_fc for t in data.aircraft_tails], dtype=np.float32),
        
        # Daily flight rates
        'daily_fh': np.array([
            sum(data.flight_rates[t].dfh.values()) / 12 if t in data.flight_rates else 10.0
            for t in data.aircraft_tails
        ], dtype=np.float32),
        'daily_fc': np.array([
            sum(data.flight_rates[t].dfc.values()) / 12 if t in data.flight_rates else 4.0
            for t in data.aircraft_tails
        ], dtype=np.float32),
        
        # C-check turnaround time (TAT)
        'c_tat': np.array([
            data.c_check_info[t].elapsed_times[0] if t in data.c_check_info else data.config.c_e_t
            for t in data.aircraft_tails
        ], dtype=np.float32),
    }
    
    return static


# =============================================================================
# COST FUNCTION (Equation 28)
# =============================================================================

def compute_check_cost(state, static, c_mask, a_mask, p_a, p_d):
    """
    Compute cost for one day's checks per paper's Equation 28.
    
    Cost = Σ(unused_FH) + P_a × θ + P_d × η
    
    Where:
    - unused_FH = interval_FH - actual_FH at check (only for checked aircraft)
    - θ = number of checks performed while in tolerance zone
    - η = number of aircraft overdue but not receiving maintenance
    
    Returns:
        cost: Total cost for this step
        tolerance_events: Number of checks done in tolerance (θ)
        n_c_checks: Number of C-checks performed
        n_a_checks: Number of A-checks performed
    """
    grounded = state['grounded_days'] > 0
    
    # Count checks
    n_c_checks = np.sum(c_mask)
    n_a_checks = np.sum(a_mask)
    
    # --- UNUSED FH COMPONENT ---
    # C-checks: unused FH for C-check + merged A-check (paper's Assumption A.6)
    c_unused_fh = np.maximum(static['c_ci_fh'] - state['fh_c'], 0.0)
    a_unused_fh_in_c = np.maximum(static['a_ci_fh'] - state['fh_a'], 0.0)
    c_cost = np.sum(c_mask * (c_unused_fh + a_unused_fh_in_c))
    
    # A-checks: unused FH
    a_unused_fh = np.maximum(static['a_ci_fh'] - state['fh_a'], 0.0)
    a_cost = np.sum(a_mask * a_unused_fh)
    
    # --- TOLERANCE PENALTY (P_a × θ) ---
    # Check is in tolerance if DY > interval OR FH > interval OR FC > interval
    c_in_tolerance = (
        (state['dy_c'] > static['c_ci_dy']) |
        (state['fh_c'] > static['c_ci_fh']) |
        (state['fc_c'] > static['c_ci_fc'])
    )
    a_in_tolerance = (
        (state['dy_a'] > static['a_ci_dy']) |
        (state['fh_a'] > static['a_ci_fh']) |
        (state['fc_a'] > static['a_ci_fc'])
    )
    
    c_tolerance_events = np.sum(c_mask & c_in_tolerance)
    a_tolerance_events = np.sum(a_mask & a_in_tolerance)
    tolerance_events = c_tolerance_events + a_tolerance_events
    tolerance_penalty = tolerance_events * p_a
    
    # --- GROUND-WAITING PENALTY (P_d × η) ---
    # Aircraft is overdue if past interval + tolerance buffer
    # For simplicity, we check if past interval (tolerance zone entry = overdue for penalty)
    c_overdue = c_in_tolerance  # Past interval = needs regulatory approval
    a_overdue = a_in_tolerance
    overdue = c_overdue | a_overdue
    
    # Ground-waiting: overdue but not grounded and not being checked
    checked = c_mask | a_mask
    ground_waiting = overdue & ~grounded & ~checked
    ground_waiting_penalty = np.sum(ground_waiting) * p_d
    
    total_cost = c_cost + a_cost + tolerance_penalty + ground_waiting_penalty
    
    return total_cost, int(tolerance_events), int(n_c_checks), int(n_a_checks)


# =============================================================================
# STATE TRANSITIONS
# =============================================================================

def apply_action(state, static, c_mask, a_mask):
    """
    Apply checks and advance one day.
    
    Implements:
    - C-check resets both C and A counters (paper's Assumption A.6: A merges into C)
    - A-check resets A counters only
    - Grounding for check duration (TAT)
    - Daily aging for non-grounded aircraft
    """
    new_state = {k: v.copy() for k, v in state.items()}
    
    # --- APPLY C-CHECKS ---
    # Reset C-check counters
    new_state['dy_c'] = np.where(c_mask, 0.0, new_state['dy_c'])
    new_state['fh_c'] = np.where(c_mask, 0.0, new_state['fh_c'])
    new_state['fc_c'] = np.where(c_mask, 0.0, new_state['fc_c'])
    
    # Reset A-check counters (A merges into C per Assumption A.6)
    new_state['dy_a'] = np.where(c_mask, 0.0, new_state['dy_a'])
    new_state['fh_a'] = np.where(c_mask, 0.0, new_state['fh_a'])
    new_state['fc_a'] = np.where(c_mask, 0.0, new_state['fc_a'])
    
    # Ground aircraft for C-check TAT
    new_state['grounded_days'] = np.where(c_mask, static['c_tat'], new_state['grounded_days'])
    
    # --- APPLY A-CHECKS ---
    # Only for aircraft not getting C-check
    a_only_mask = a_mask & ~c_mask
    new_state['dy_a'] = np.where(a_only_mask, 0.0, new_state['dy_a'])
    new_state['fh_a'] = np.where(a_only_mask, 0.0, new_state['fh_a'])
    new_state['fc_a'] = np.where(a_only_mask, 0.0, new_state['fc_a'])
    
    # --- ADVANCE ONE DAY ---
    # Decrement grounding countdown
    new_state['grounded_days'] = np.maximum(new_state['grounded_days'] - 1, 0)
    
    # Age non-grounded aircraft
    active = new_state['grounded_days'] <= 0
    new_state['dy_c'] += active.astype(np.float32)
    new_state['dy_a'] += active.astype(np.float32)
    new_state['fh_c'] += active * static['daily_fh']
    new_state['fh_a'] += active * static['daily_fh']
    new_state['fc_c'] += active * static['daily_fc']
    new_state['fc_a'] += active * static['daily_fc']
    
    return new_state


# =============================================================================
# PRIORITY ORDERING (Equations 36-40)
# =============================================================================

def compute_remaining_days(state, static, check_type):
    """
    Compute remaining days until check for each aircraft.
    
    Paper's priority ordering uses earliest deadline first.
    Remaining days = min(remaining by DY, remaining by FH, remaining by FC)
    """
    if check_type == 'C':
        remaining_dy = static['c_ci_dy'] - state['dy_c']
        remaining_fh = (static['c_ci_fh'] - state['fh_c']) / np.maximum(static['daily_fh'], 0.1)
        remaining_fc = (static['c_ci_fc'] - state['fc_c']) / np.maximum(static['daily_fc'], 0.1)
    else:  # A-check
        remaining_dy = static['a_ci_dy'] - state['dy_a']
        remaining_fh = (static['a_ci_fh'] - state['fh_a']) / np.maximum(static['daily_fh'], 0.1)
        remaining_fc = (static['a_ci_fc'] - state['fc_a']) / np.maximum(static['daily_fc'], 0.1)
    
    return np.minimum(np.minimum(remaining_dy, remaining_fh), remaining_fc)


def select_aircraft(state, static, n_select, check_type, exclude_mask=None):
    """
    Select aircraft for checks using earliest deadline first (Eq. 36-40).
    
    Returns boolean mask of selected aircraft.
    """
    n = len(state['dy_c'])
    grounded = state['grounded_days'] > 0
    
    remaining = compute_remaining_days(state, static, check_type)
    
    # Exclude grounded aircraft and already-selected
    remaining = np.where(grounded, np.inf, remaining)
    if exclude_mask is not None:
        remaining = np.where(exclude_mask, np.inf, remaining)
    
    # Select n_select most urgent (lowest remaining days)
    # Use rank-based selection for JAX compatibility
    ranks = np.sum(remaining[:, None] > remaining[None, :], axis=1)
    mask = (ranks < n_select) & (n_select > 0) & ~grounded
    if exclude_mask is not None:
        mask = mask & ~exclude_mask
    
    return mask


# =============================================================================
# STATE AGGREGATION (Section 4.4)
# =============================================================================

def compute_histogram(state, static, u):
    """
    Compute utilization histogram for state aggregation.
    
    Paper's Equation 44 uses mean utilization; we use histograms for
    more precise state distinction (conservative choice).
    
    Args:
        state: Current fleet state
        static: Static parameters
        u: Discretization step (paper's parameter u)
    
    Returns:
        Hashable tuple representing aggregated state
    """
    grounded = state['grounded_days'] > 0
    active = ~grounded
    
    # C-check FH bins
    c_bins = (state['fh_c'] / u).astype(int)
    
    # A-check FH bins (use u/2 for finer resolution - A-check intervals are shorter)
    a_bins = (state['fh_a'] / (u / 2)).astype(int)
    
    # Count tolerance usage
    c_in_tol = (state['dy_c'] > static['c_ci_dy']) | \
               (state['fh_c'] > static['c_ci_fh']) | \
               (state['fc_c'] > static['c_ci_fc'])
    a_in_tol = (state['dy_a'] > static['a_ci_dy']) | \
               (state['fh_a'] > static['a_ci_fh']) | \
               (state['fc_a'] > static['a_ci_fc'])
    tolerance_count = int(np.sum((c_in_tol | a_in_tol) & active))
    grounded_count = int(np.sum(grounded))
    
    # Create histogram tuples
    c_unique, c_counts = np.unique(c_bins[active], return_counts=True)
    a_unique, a_counts = np.unique(a_bins[active], return_counts=True)
    
    c_hist = tuple(sorted(zip(c_unique.tolist(), c_counts.tolist())))
    a_hist = tuple(sorted(zip(a_unique.tolist(), a_counts.tolist())))
    
    return (c_hist, a_hist, tolerance_count, grounded_count)


# =============================================================================
# THRIFTY PRUNING (Section 4.4.1)
# =============================================================================

def thrifty_prune(frontier, beam_width):
    """
    Apply thrifty dominance pruning (paper's Section 4.4.1).
    
    Strategy:
    1. Group states by histogram (aggregated state)
    2. Keep only lowest-cost state per histogram
    3. Apply beam width limit
    
    This is a simplification of full dominance checking, which would
    require O(n²) comparisons. The paper notes this is acceptable
    for tractability.
    """
    # Group by histogram
    by_histogram = defaultdict(list)
    for entry in frontier:
        histogram = entry[5]  # (state, cost, tol, n_c, n_a, histogram)
        by_histogram[histogram].append(entry)
    
    # Keep best (lowest cost) per histogram
    pruned = []
    for hist, entries in by_histogram.items():
        best = min(entries, key=lambda x: x[1])  # Sort by cost
        pruned.append(best)
    
    # Beam width limit
    pruned.sort(key=lambda x: x[1])
    return pruned[:beam_width]


# =============================================================================
# MAIN SCHEDULER
# =============================================================================

class Scheduler:
    """
    Forward beam search scheduler implementing Deng et al. (2020).
    
    This is the canonical implementation following the paper's methodology:
    - Forward simulation (NOT Bellman DP)
    - 7 daily actions from Table 2
    - Equation 28 cost function
    - Histogram-based state aggregation
    - Thrifty dominance pruning
    - Earliest deadline first priority ordering
    
    Args:
        data: MaintenanceData from load_maintenance_data()
        beam_width: Maximum states to keep per day (paper uses ~500-1000)
        u: Discretization step for histograms (paper's u parameter)
        p_a: Tolerance penalty (paper's P_a, should be large, e.g., 10^8)
        p_d: Ground-waiting penalty (paper's P_d, should be large, e.g., 10^8)
    """
    
    def __init__(
        self,
        data: MaintenanceData,
        beam_width: int = 200,
        u: float = 100.0,
        p_a: float = 1e8,
        p_d: float = 1e8,
    ):
        self.data = data
        self.beam_width = beam_width
        self.u = u
        self.p_a = p_a
        self.p_d = p_d
        
        # Initialize state and parameters
        self.initial_state = create_fleet_state(data)
        self.static = create_static_params(data)
        self.n_aircraft = len(data.aircraft_tails)
        
        # Capacity constraints
        self.max_c = data.config.max_c_check
        self.max_a = data.config.max_a_check
    
    def schedule(self, planning_days: int, verbose: bool = False) -> dict:
        """
        Run the forward beam search algorithm.
        
        Args:
            planning_days: Number of days to plan (paper uses 4 years = 1460 days)
            verbose: Print progress information
        
        Returns:
            Dictionary with:
            - total_cost: Paper's objective value (Eq. 28)
            - tolerance_events: Number of checks done in tolerance
            - n_c_checks: Total C-checks scheduled
            - n_a_checks: Total A-checks scheduled
            - computation_time: Seconds elapsed
        """
        start_time = time.time()
        
        if verbose:
            print(f"[Deng] Starting {planning_days}-day schedule, beam={self.beam_width}")
            print(f"[Deng] Fleet: {self.n_aircraft} aircraft, P_a={self.p_a:.0e}, P_d={self.p_d:.0e}")
            sys.stdout.flush()
        
        # Initialize frontier: (state, cost, tol_events, n_c, n_a, histogram)
        init_hist = compute_histogram(self.initial_state, self.static, self.u)
        frontier = [(self.initial_state.copy(), 0.0, 0, 0, 0, init_hist)]
        
        for day in range(planning_days):
            # Progress reporting
            if verbose and day % 50 == 0:
                elapsed = time.time() - start_time
                best = min(frontier, key=lambda x: x[1])
                n_unique = len(set(f[5] for f in frontier))
                eta = (elapsed / max(day, 1)) * (planning_days - day) / 60 if day > 0 else 0
                print(f"[Deng] Day {day:4d}/{planning_days} | "
                      f"states={len(frontier):3d} ({n_unique:3d} unique) | "
                      f"cost={best[1]:.0f} | C={best[3]} A={best[4]} | "
                      f"time={elapsed:.0f}s ETA={eta:.1f}m")
                sys.stdout.flush()
            
            # Expand frontier
            next_frontier = []
            
            for state, cum_cost, cum_tol, cum_c, cum_a, _ in frontier:
                # Evaluate all 7 actions
                for n_a, n_c in PAPER_ACTIONS:
                    # Respect capacity constraints
                    n_c = min(n_c, self.max_c)
                    n_a = min(n_a, self.max_a)
                    
                    # Select aircraft (earliest deadline first)
                    c_mask = select_aircraft(state, self.static, n_c, 'C')
                    a_mask = select_aircraft(state, self.static, n_a, 'A', exclude_mask=c_mask)
                    
                    # Compute cost (Equation 28)
                    cost, tol, nc, na = compute_check_cost(
                        state, self.static, c_mask, a_mask, self.p_a, self.p_d
                    )
                    
                    # Apply action and advance one day
                    new_state = apply_action(state, self.static, c_mask, a_mask)
                    
                    # Compute histogram for aggregation
                    new_hist = compute_histogram(new_state, self.static, self.u)
                    
                    next_frontier.append((
                        new_state,
                        cum_cost + cost,
                        cum_tol + tol,
                        cum_c + nc,
                        cum_a + na,
                        new_hist,
                    ))
            
            # Thrifty pruning (Section 4.4.1)
            frontier = thrifty_prune(next_frontier, self.beam_width)
            
            if not frontier:
                break
        
        # Extract best solution
        best = min(frontier, key=lambda x: x[1])
        computation_time = time.time() - start_time
        
        if verbose:
            print(f"[Deng] Complete in {computation_time:.1f}s")
            print(f"[Deng] C-checks: {best[3]}, A-checks: {best[4]}")
            print(f"[Deng] Tolerance events: {best[2]}")
            print(f"[Deng] Total cost: {best[1]:.0f}")
        
        return {
            'total_cost': best[1],
            'tolerance_events': best[2],
            'n_c_checks': best[3],
            'n_a_checks': best[4],
            'computation_time': computation_time,
            'final_state': best[0],
        }


# Backward compatibility aliases
DengScheduler = Scheduler
JaxDPScheduler = Scheduler


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    from .data_loader import load_maintenance_data
    
    print("Aircraft Maintenance Scheduler - Deng et al. (2020)")
    print("=" * 60)
    
    data = load_maintenance_data("Sep 25 2017 (Case Study 2018-2021).xlsx")
    print(f"Loaded {len(data.aircraft_tails)} aircraft")
    
    scheduler = Scheduler(data, beam_width=200)
    result = scheduler.schedule(4 * 365, verbose=True)
    
    print("\n" + "=" * 60)
    print("RESULTS (4-year horizon)")
    print("=" * 60)
    print(f"C-checks: {result['n_c_checks']}")
    print(f"A-checks: {result['n_a_checks']}")
    print(f"Tolerance events: {result['tolerance_events']}")
    print(f"Computation time: {result['computation_time']:.1f}s")

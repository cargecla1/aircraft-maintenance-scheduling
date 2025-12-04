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

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional
from collections import defaultdict
import numpy as np
import time
import sys

from .data_loader import MaintenanceData


# =============================================================================
# PAPER'S 7 DAILY ACTIONS (Table 2)
# =============================================================================

PAPER_ACTIONS = [
    (0, 0),  # Do nothing
    (1, 0),  # 1 A-check only
    (0, 1),  # 1 C-check only
    (1, 1),  # 1 A-check + 1 C-check
    (0, 2),  # 2 C-checks
    (1, 2),  # 1 A-check + 2 C-checks
    (2, 0),  # 2 A-checks
]


# =============================================================================
# RESULT DATA STRUCTURES
# =============================================================================

@dataclass
class FleetAction:
    """A concrete fleet action specifying which aircraft get which checks."""
    a_checks: list[str]  # Tail numbers getting A-checks
    c_checks: list[str]  # Tail numbers getting C-checks
    
    @property
    def n_a(self) -> int:
        return len(self.a_checks)
    
    @property
    def n_c(self) -> int:
        return len(self.c_checks)
    
    def is_null(self) -> bool:
        return self.n_a == 0 and self.n_c == 0


@dataclass
class ScheduledCheck:
    """A scheduled maintenance check with paper's eq. 28 metrics."""
    tail: str
    check_type: str  # 'A', 'C', or 'D'
    start_date: date
    end_date: Optional[date] = None
    duration: int = 1
    fh_at_check: float = 0.0
    unused_fh: float = 0.0
    used_tolerance_fh: float = 0.0
    is_tolerance_event: bool = False
    a_fh_at_check: float = 0.0
    a_unused_fh: float = 0.0

    @property
    def total_unused_fh(self) -> float:
        return self.unused_fh + self.a_unused_fh


@dataclass
class Table3KPIs:
    """KPIs matching Table 3 in Deng et al. paper."""
    n_a_checks: int
    n_c_checks: int
    n_d_checks: int
    avg_fh_at_a_checks: float
    std_fh_at_a_checks: float
    avg_fh_at_c_checks: float
    std_fh_at_c_checks: float
    total_unused_fh: float
    avg_unused_fh_a: float
    avg_unused_fh_c: float
    total_fh_tolerance_used: float
    total_tolerance_events: int
    
    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class ScheduleResult:
    """Result of the scheduling optimization with paper's eq. 28 metrics."""
    checks: list[ScheduledCheck]
    total_a_checks: int
    total_c_checks: int
    total_d_checks: int
    computation_time: float
    total_unused_fh: float = 0.0
    total_tolerance_penalty: float = 0.0
    total_tolerance_events: int = 0
    _kpis: Optional[Table3KPIs] = field(default=None, repr=False)

    @property
    def objective(self) -> float:
        return self.total_unused_fh + self.total_tolerance_penalty

    def get_checks_for_aircraft(self, tail: str) -> list[ScheduledCheck]:
        return [c for c in self.checks if c.tail == tail]

    def compute_kpis(self) -> Table3KPIs:
        if self._kpis is not None:
            return self._kpis
        
        a_checks = [c for c in self.checks if c.check_type == "A"]
        c_checks = [c for c in self.checks if c.check_type == "C"]
        d_checks = [c for c in self.checks if c.check_type == "D"]
        
        a_fh = [c.fh_at_check for c in a_checks]
        c_fh = [c.fh_at_check for c in c_checks + d_checks]
        
        self._kpis = Table3KPIs(
            n_a_checks=len(a_checks),
            n_c_checks=len(c_checks),
            n_d_checks=len(d_checks),
            avg_fh_at_a_checks=float(np.mean(a_fh)) if a_fh else 0.0,
            std_fh_at_a_checks=float(np.std(a_fh)) if a_fh else 0.0,
            avg_fh_at_c_checks=float(np.mean(c_fh)) if c_fh else 0.0,
            std_fh_at_c_checks=float(np.std(c_fh)) if c_fh else 0.0,
            total_unused_fh=sum(c.unused_fh for c in a_checks) + sum(c.total_unused_fh for c in c_checks + d_checks),
            avg_unused_fh_a=sum(c.unused_fh for c in a_checks) / len(a_checks) if a_checks else 0.0,
            avg_unused_fh_c=sum(c.total_unused_fh for c in c_checks + d_checks) / len(c_checks + d_checks) if (c_checks + d_checks) else 0.0,
            total_fh_tolerance_used=sum(c.used_tolerance_fh for c in self.checks),
            total_tolerance_events=sum(1 for c in self.checks if c.is_tolerance_event),
        )
        return self._kpis

    def summary(self) -> str:
        return "\n".join([
            "Schedule Summary",
            "================",
            f"Checks: {self.total_a_checks} A / {self.total_c_checks} C / {self.total_d_checks} D",
            f"Unused FH: {self.total_unused_fh:.1f}",
            f"Tolerance events: {self.total_tolerance_events}",
            f"Time: {self.computation_time:.2f}s",
        ])


# =============================================================================
# STATE REPRESENTATION
# =============================================================================

def create_fleet_state(data: MaintenanceData) -> dict:
    """Create initial fleet state from maintenance data."""
    n = len(data.aircraft_tails)
    return {
        'dy_c': np.array([data.c_initial[t].dy_c for t in data.aircraft_tails], dtype=np.float32),
        'fh_c': np.array([data.c_initial[t].fh_c for t in data.aircraft_tails], dtype=np.float32),
        'fc_c': np.array([data.c_initial[t].fc_c for t in data.aircraft_tails], dtype=np.float32),
        'dy_a': np.array([data.a_initial[t].dy_a for t in data.aircraft_tails], dtype=np.float32),
        'fh_a': np.array([data.a_initial[t].fh_a for t in data.aircraft_tails], dtype=np.float32),
        'fc_a': np.array([data.a_initial[t].fc_a for t in data.aircraft_tails], dtype=np.float32),
        'grounded_days': np.zeros(n, dtype=np.float32),
    }


def create_static_params(data: MaintenanceData) -> dict:
    """Create static parameters (intervals, rates, etc.)."""
    return {
        'c_ci_dy': np.array([data.c_initial[t].c_ci_dy for t in data.aircraft_tails], dtype=np.float32),
        'c_ci_fh': np.array([data.c_initial[t].c_ci_fh for t in data.aircraft_tails], dtype=np.float32),
        'c_ci_fc': np.array([data.c_initial[t].c_ci_fc for t in data.aircraft_tails], dtype=np.float32),
        'a_ci_dy': np.array([data.a_initial[t].a_ci_dy for t in data.aircraft_tails], dtype=np.float32),
        'a_ci_fh': np.array([data.a_initial[t].a_ci_fh for t in data.aircraft_tails], dtype=np.float32),
        'a_ci_fc': np.array([data.a_initial[t].a_ci_fc for t in data.aircraft_tails], dtype=np.float32),
        'daily_fh': np.array([
            sum(data.flight_rates[t].dfh.values()) / 12 if t in data.flight_rates else 10.0
            for t in data.aircraft_tails
        ], dtype=np.float32),
        'daily_fc': np.array([
            sum(data.flight_rates[t].dfc.values()) / 12 if t in data.flight_rates else 4.0
            for t in data.aircraft_tails
        ], dtype=np.float32),
        'c_tat': np.array([
            data.c_check_info[t].elapsed_times[0] if t in data.c_check_info else data.config.c_e_t
            for t in data.aircraft_tails
        ], dtype=np.float32),
    }


# =============================================================================
# COST FUNCTION (Equation 28)
# =============================================================================

def compute_check_cost(state, static, c_mask, a_mask, p_a, p_d):
    """Compute cost for one day's checks per paper's Equation 28."""
    grounded = state['grounded_days'] > 0
    n_c_checks = np.sum(c_mask)
    n_a_checks = np.sum(a_mask)
    
    # Unused FH
    c_unused_fh = np.maximum(static['c_ci_fh'] - state['fh_c'], 0.0)
    a_unused_fh_in_c = np.maximum(static['a_ci_fh'] - state['fh_a'], 0.0)
    c_cost = np.sum(c_mask * (c_unused_fh + a_unused_fh_in_c))
    a_cost = np.sum(a_mask * np.maximum(static['a_ci_fh'] - state['fh_a'], 0.0))
    
    # Tolerance penalty
    c_in_tol = (state['dy_c'] > static['c_ci_dy']) | (state['fh_c'] > static['c_ci_fh']) | (state['fc_c'] > static['c_ci_fc'])
    a_in_tol = (state['dy_a'] > static['a_ci_dy']) | (state['fh_a'] > static['a_ci_fh']) | (state['fc_a'] > static['a_ci_fc'])
    tolerance_events = np.sum(c_mask & c_in_tol) + np.sum(a_mask & a_in_tol)
    
    # Ground-waiting penalty
    overdue = c_in_tol | a_in_tol
    ground_waiting = overdue & ~grounded & ~(c_mask | a_mask)
    
    total_cost = c_cost + a_cost + tolerance_events * p_a + np.sum(ground_waiting) * p_d
    return total_cost, int(tolerance_events), int(n_c_checks), int(n_a_checks)


# =============================================================================
# STATE TRANSITIONS
# =============================================================================

def apply_action(state, static, c_mask, a_mask):
    """Apply checks and advance one day."""
    new_state = {k: v.copy() for k, v in state.items()}
    
    # C-checks reset both C and A counters
    for key in ['dy_c', 'fh_c', 'fc_c', 'dy_a', 'fh_a', 'fc_a']:
        new_state[key] = np.where(c_mask, 0.0, new_state[key])
    new_state['grounded_days'] = np.where(c_mask, static['c_tat'], new_state['grounded_days'])
    
    # A-checks (only if not getting C-check)
    a_only = a_mask & ~c_mask
    for key in ['dy_a', 'fh_a', 'fc_a']:
        new_state[key] = np.where(a_only, 0.0, new_state[key])
    
    # Advance one day
    new_state['grounded_days'] = np.maximum(new_state['grounded_days'] - 1, 0)
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

def select_aircraft(state, static, n_select, check_type, exclude_mask=None):
    """Select aircraft using earliest deadline first."""
    grounded = state['grounded_days'] > 0
    
    if check_type == 'C':
        remaining = np.minimum(
            static['c_ci_dy'] - state['dy_c'],
            np.minimum(
                (static['c_ci_fh'] - state['fh_c']) / np.maximum(static['daily_fh'], 0.1),
                (static['c_ci_fc'] - state['fc_c']) / np.maximum(static['daily_fc'], 0.1)
            )
        )
    else:
        remaining = np.minimum(
            static['a_ci_dy'] - state['dy_a'],
            np.minimum(
                (static['a_ci_fh'] - state['fh_a']) / np.maximum(static['daily_fh'], 0.1),
                (static['a_ci_fc'] - state['fc_a']) / np.maximum(static['daily_fc'], 0.1)
            )
        )
    
    remaining = np.where(grounded, np.inf, remaining)
    if exclude_mask is not None:
        remaining = np.where(exclude_mask, np.inf, remaining)
    
    ranks = np.sum(remaining[:, None] > remaining[None, :], axis=1)
    mask = (ranks < n_select) & (n_select > 0) & ~grounded
    if exclude_mask is not None:
        mask = mask & ~exclude_mask
    return mask


# =============================================================================
# STATE AGGREGATION (Section 4.4)
# =============================================================================

def compute_histogram(state, static, u):
    """Compute utilization histogram for state aggregation."""
    grounded = state['grounded_days'] > 0
    active = ~grounded
    
    c_bins = (state['fh_c'] / u).astype(int)
    a_bins = (state['fh_a'] / (u / 2)).astype(int)
    
    c_in_tol = (state['dy_c'] > static['c_ci_dy']) | (state['fh_c'] > static['c_ci_fh']) | (state['fc_c'] > static['c_ci_fc'])
    a_in_tol = (state['dy_a'] > static['a_ci_dy']) | (state['fh_a'] > static['a_ci_fh']) | (state['fc_a'] > static['a_ci_fc'])
    
    c_unique, c_counts = np.unique(c_bins[active], return_counts=True)
    a_unique, a_counts = np.unique(a_bins[active], return_counts=True)
    
    return (
        tuple(sorted(zip(c_unique.tolist(), c_counts.tolist()))),
        tuple(sorted(zip(a_unique.tolist(), a_counts.tolist()))),
        int(np.sum((c_in_tol | a_in_tol) & active)),
        int(np.sum(grounded))
    )


# =============================================================================
# THRIFTY PRUNING (Section 4.4.1)
# =============================================================================

def thrifty_prune(frontier, beam_width):
    """Apply thrifty dominance pruning.
    
    Entry format: (state, cost, tol, n_c, n_a, histogram, history)
    """
    by_histogram = defaultdict(list)
    for entry in frontier:
        by_histogram[entry[5]].append(entry)  # entry[5] is histogram
    
    pruned = [min(entries, key=lambda x: x[1]) for entries in by_histogram.values()]
    pruned.sort(key=lambda x: x[1])
    return pruned[:beam_width]


# =============================================================================
# MAIN SCHEDULER
# =============================================================================

class Scheduler:
    """
    Forward beam search scheduler implementing Deng et al. (2020).
    
    Args:
        data: MaintenanceData from load_maintenance_data()
        beam_width: Maximum states to keep per day (default 200)
        u: Discretization step for histograms (default 100.0)
        p_a: Tolerance penalty (default 10^8)
        p_d: Ground-waiting penalty (default 10^8)
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
        self.initial_state = create_fleet_state(data)
        self.static = create_static_params(data)
        self.n_aircraft = len(data.aircraft_tails)
        self.max_c = data.config.max_c_check
        self.max_a = data.config.max_a_check
    
    def schedule(self, planning_days: int, verbose: bool = False) -> ScheduleResult:
        """
        Run the forward beam search algorithm.
        
        Returns ScheduleResult with full schedule (checks with dates and aircraft).
        """
        start_time = time.time()
        start_date = self.data.config.begin_day
        tails = self.data.aircraft_tails
        
        if verbose:
            print(f"[Deng] Starting {planning_days}-day schedule, beam={self.beam_width}")
            print(f"[Deng] Fleet: {self.n_aircraft} aircraft, P_a={self.p_a:.0e}, P_d={self.p_d:.0e}")
            sys.stdout.flush()
        
        init_hist = compute_histogram(self.initial_state, self.static, self.u)
        # Frontier entry: (state, cost, tol, n_c, n_a, histogram, history)
        # history is list of (day, c_indices, a_indices, state_snapshot) tuples
        frontier = [(self.initial_state.copy(), 0.0, 0, 0, 0, init_hist, [])]
        
        for day in range(planning_days):
            if verbose and day % 50 == 0:
                elapsed = time.time() - start_time
                best = min(frontier, key=lambda x: x[1])
                eta = (elapsed / max(day, 1)) * (planning_days - day) / 60 if day > 0 else 0
                print(f"[Deng] Day {day:4d}/{planning_days} | "
                      f"states={len(frontier):3d} | cost={best[1]:.0f} | "
                      f"C={best[3]} A={best[4]} | time={elapsed:.0f}s ETA={eta:.1f}m")
                sys.stdout.flush()
            
            next_frontier = []
            for state, cum_cost, cum_tol, cum_c, cum_a, _, history in frontier:
                for n_a, n_c in PAPER_ACTIONS:
                    n_c = min(n_c, self.max_c)
                    n_a = min(n_a, self.max_a)
                    
                    c_mask = select_aircraft(state, self.static, n_c, 'C')
                    a_mask = select_aircraft(state, self.static, n_a, 'A', exclude_mask=c_mask)
                    
                    cost, tol, nc, na = compute_check_cost(state, self.static, c_mask, a_mask, self.p_a, self.p_d)
                    
                    # Record action if any checks scheduled
                    if nc > 0 or na > 0:
                        c_indices = tuple(np.where(c_mask)[0].tolist())
                        a_indices = tuple(np.where(a_mask)[0].tolist())
                        # Snapshot state at time of check for FH/tolerance info
                        state_snap = (state['fh_c'].copy(), state['fh_a'].copy(),
                                      state['dy_c'].copy(), state['dy_a'].copy())
                        new_history = history + [(day, c_indices, a_indices, state_snap)]
                    else:
                        new_history = history
                    
                    new_state = apply_action(state, self.static, c_mask, a_mask)
                    new_hist = compute_histogram(new_state, self.static, self.u)
                    
                    next_frontier.append((new_state, cum_cost + cost, cum_tol + tol, 
                                         cum_c + nc, cum_a + na, new_hist, new_history))
            
            frontier = thrifty_prune(next_frontier, self.beam_width)
            if not frontier:
                break
        
        best = min(frontier, key=lambda x: x[1])
        computation_time = time.time() - start_time
        
        if verbose:
            print(f"[Deng] Complete in {computation_time:.1f}s")
            print(f"[Deng] C-checks: {best[3]}, A-checks: {best[4]}, Tolerance: {best[2]}")
        
        # Convert history to ScheduledCheck objects
        checks = []
        for day, c_indices, a_indices, state_snap in best[6]:
            fh_c, fh_a, dy_c, dy_a = state_snap
            check_date = start_date + timedelta(days=day)
            
            for idx in c_indices:
                tail = tails[idx]
                c_interval_fh = self.static['c_ci_fh'][idx]
                c_interval_dy = self.static['c_ci_dy'][idx]
                a_interval_fh = self.static['a_ci_fh'][idx]
                fh_at = float(fh_c[idx])
                unused = max(0.0, c_interval_fh - fh_at)
                a_fh_at = float(fh_a[idx])
                a_unused = max(0.0, a_interval_fh - a_fh_at)
                is_tol = (dy_c[idx] > c_interval_dy) or (fh_at > c_interval_fh)
                tat = int(self.static['c_tat'][idx])
                
                checks.append(ScheduledCheck(
                    tail=tail,
                    check_type='C',
                    start_date=check_date,
                    end_date=check_date + timedelta(days=tat - 1),
                    duration=tat,
                    fh_at_check=fh_at,
                    unused_fh=unused,
                    is_tolerance_event=is_tol,
                    a_fh_at_check=a_fh_at,
                    a_unused_fh=a_unused,
                ))
            
            for idx in a_indices:
                tail = tails[idx]
                a_interval_fh = self.static['a_ci_fh'][idx]
                a_interval_dy = self.static['a_ci_dy'][idx]
                fh_at = float(fh_a[idx])
                unused = max(0.0, a_interval_fh - fh_at)
                is_tol = (dy_a[idx] > a_interval_dy) or (fh_at > a_interval_fh)
                
                checks.append(ScheduledCheck(
                    tail=tail,
                    check_type='A',
                    start_date=check_date,
                    end_date=check_date,
                    duration=1,
                    fh_at_check=fh_at,
                    unused_fh=unused,
                    is_tolerance_event=is_tol,
                ))
        
        return ScheduleResult(
            checks=checks,
            total_a_checks=best[4],
            total_c_checks=best[3],
            total_d_checks=0,
            computation_time=computation_time,
            total_unused_fh=sum(c.total_unused_fh for c in checks),
            total_tolerance_events=best[2],
        )

"""
Shared data structures for Aircraft Maintenance Check Scheduling.

For the scheduler implementation, see jax_solver.py which provides
the paper-faithful Deng et al. (2020) forward beam search algorithm.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional
import numpy as np


# =============================================================================
# Paper's 7 Daily Actions (Table 2 in Deng et al.)
# =============================================================================

PAPER_ACTIONS = [
    (0, 0),  # No maintenance
    (1, 0),  # 1 A-check only
    (0, 1),  # 1 C-check only
    (1, 1),  # 1 A-check + 1 C-check
    (0, 2),  # 2 C-checks
    (1, 2),  # 1 A-check + 2 C-checks
    (2, 0),  # 2 A-checks
]


# =============================================================================
# Result Data Structures
# =============================================================================

@dataclass
class FleetAction:
    """
    A concrete fleet action specifying which aircraft get which checks.
    """
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
    
    # FH metrics (paper's objective)
    fh_at_check: float = 0.0
    unused_fh: float = 0.0
    
    # Tolerance tracking
    used_tolerance_fh: float = 0.0
    is_tolerance_event: bool = False
    
    # For C/D checks: merged A-check data
    a_fh_at_check: float = 0.0
    a_unused_fh: float = 0.0

    @property
    def total_unused_fh(self) -> float:
        """Total unused FH including merged A-check for C/D checks."""
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
        return {
            'n_a_checks': self.n_a_checks,
            'n_c_checks': self.n_c_checks,
            'n_d_checks': self.n_d_checks,
            'avg_fh_at_a_checks': self.avg_fh_at_a_checks,
            'std_fh_at_a_checks': self.std_fh_at_a_checks,
            'avg_fh_at_c_checks': self.avg_fh_at_c_checks,
            'std_fh_at_c_checks': self.std_fh_at_c_checks,
            'total_unused_fh': self.total_unused_fh,
            'avg_unused_fh_a': self.avg_unused_fh_a,
            'avg_unused_fh_c': self.avg_unused_fh_c,
            'total_fh_tolerance_used': self.total_fh_tolerance_used,
            'total_tolerance_events': self.total_tolerance_events,
        }


@dataclass
class ScheduleResult:
    """Result of the scheduling optimization with paper's eq. 28 metrics."""
    checks: list[ScheduledCheck]
    total_a_checks: int
    total_c_checks: int
    total_d_checks: int
    computation_time: float
    
    # Paper's objective (eq. 28)
    total_unused_fh: float = 0.0
    total_tolerance_penalty: float = 0.0
    total_tolerance_events: int = 0
    
    _kpis: Optional[Table3KPIs] = field(default=None, repr=False)

    @property
    def objective(self) -> float:
        """Paper's objective value: unused FH + tolerance penalties."""
        return self.total_unused_fh + self.total_tolerance_penalty

    def get_checks_for_aircraft(self, tail: str) -> list[ScheduledCheck]:
        return [c for c in self.checks if c.tail == tail]

    def compute_kpis(self) -> Table3KPIs:
        """Compute Table 3 KPIs from the scheduled checks."""
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
            "",
            "Paper's Objective (eq. 28):",
            f"  Unused FH: {self.total_unused_fh:.1f}",
            f"  Tolerance penalty: {self.total_tolerance_penalty:.1f}",
            f"  Total: {self.objective:.1f}",
            f"  Tolerance events: {self.total_tolerance_events}",
            "",
            f"Time: {self.computation_time:.2f}s",
        ])

    def paper_summary(self) -> str:
        """Generate Table 3 compatible summary."""
        kpis = self.compute_kpis()
        return "\n".join([
            "Table 3 KPIs:",
            f"  A-checks: {kpis.n_a_checks} (avg FH: {kpis.avg_fh_at_a_checks:.1f} ± {kpis.std_fh_at_a_checks:.1f})",
            f"  C-checks: {kpis.n_c_checks} (avg FH: {kpis.avg_fh_at_c_checks:.1f} ± {kpis.std_fh_at_c_checks:.1f})",
            f"  D-checks: {kpis.n_d_checks}",
            f"  Total unused FH: {kpis.total_unused_fh:.1f}",
            f"  Tolerance events: {kpis.total_tolerance_events}",
        ])

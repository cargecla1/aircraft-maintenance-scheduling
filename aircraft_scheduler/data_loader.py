"""
Data loader module for Aircraft Maintenance Check Scheduling.

Loads and parses the Excel dataset with fleet maintenance data.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class AircraftDInitial:
    """D-check initial state for an aircraft."""
    fleet: str
    tail: str
    d_cycle: int  # Current cycle in D-check
    dy_d: int  # Days since last D-check
    d_ci_dy: int  # D-check interval in days
    d_tol_dy: int  # Tolerance allowed for D-check
    d_tolu_dy: int  # Tolerance used in previous D-check
    d_max: int  # Max cycles of C-check in a D-check cycle
    d_end: Optional[date]  # End date of previous D-check
    operation: date  # Aircraft only ages after this date


@dataclass
class AircraftCInitial:
    """C-check initial state for an aircraft."""
    fleet: str
    tail: str
    c_sn: float  # Check code of previous C-check
    c_max: int  # Total C-checks in a cycle
    dy_c: int  # Days since last C-check
    fh_c: float  # Flight hours since last C-check
    fc_c: float  # Flight cycles since last C-check
    c_ci_dy: int  # C-check interval (days)
    c_ci_fh: float  # C-check interval (flight hours)
    c_ci_fc: float  # C-check interval (flight cycles)
    c_tol_dy: int  # Tolerance allowed (days)
    c_tol_fh: float  # Tolerance allowed (flight hours)
    c_tol_fc: float  # Tolerance allowed (flight cycles)
    c_tolu_dy: int  # Tolerance used (days)
    c_tolu_fh: float  # Tolerance used (flight hours)
    c_tolu_fc: float  # Tolerance used (flight cycles)
    c_start: Optional[date]  # Start date of previous C-check
    c_end: Optional[date]  # End date of previous C-check


@dataclass
class AircraftAInitial:
    """A-check initial state for an aircraft."""
    fleet: str
    tail: str
    a_sn: float  # Check code of previous A-check
    a_max: int  # Total A-checks in a cycle
    dy_a: int  # Days since last A-check
    fh_a: float  # Flight hours since last A-check
    fc_a: float  # Flight cycles since last A-check
    a_ci_dy: int  # A-check interval (days)
    a_ci_fh: float  # A-check interval (flight hours)
    a_ci_fc: float  # A-check interval (flight cycles)
    a_tol_dy: int  # Tolerance allowed (days)
    a_tol_fh: float  # Tolerance allowed (flight hours)
    a_tol_fc: float  # Tolerance allowed (flight cycles)
    a_tolu_dy: int  # Tolerance used (days)
    a_tolu_fh: float  # Tolerance used (flight hours)
    a_tolu_fc: float  # Tolerance used (flight cycles)
    a_end: Optional[date]  # End date of previous A-check


@dataclass
class FlightRates:
    """Daily flight hours and cycles for an aircraft by month."""
    fleet: str
    tail: str
    dfh: dict[int, float]  # Month (1-12) -> daily flight hours
    dfc: dict[int, float]  # Month (1-12) -> daily flight cycles


@dataclass
class CCheckInfo:
    """Future C-check codes and elapsed times for an aircraft."""
    fleet: str
    tail: str
    check_codes: list[float]  # Next 5 C-check codes
    elapsed_times: list[int]  # Elapsed time (TAT) for next 5 C-checks


@dataclass
class DateRange:
    """A date range with start and end dates."""
    begin: date
    end: date

    def contains(self, d: date) -> bool:
        return self.begin <= d <= self.end


@dataclass
class PeakSeason:
    """Commercial peak season when C-checks should be avoided."""
    year: int
    begin: date
    end: date


@dataclass
class ReservedMaintenance:
    """Reserved maintenance activity."""
    fleet: str
    tail: str
    reserved_type: str
    begin: date
    end: date
    time: int
    slot: str


@dataclass
class AdditionalSlots:
    """Additional maintenance slots."""
    begin: date
    end: date
    total_slots: int


@dataclass
class Config:
    """Configuration parameters for optimization.
    
    Per paper's Equation 28, the cost function has three terms:
    - Unused FH at checks (primary objective)
    - P_a * theta: tolerance penalty (tolerance_penalty)
    - P_d * eta: ground-waiting penalty (ground_waiting_penalty)
    """
    begin_year: int
    total_years: int
    begin_day: date
    m_cost: float  # Legacy: unified penalty (deprecated, use tolerance_penalty/ground_waiting_penalty)
    c_cost: float  # Penalty for additional C-check slot
    max_c_check: int  # Hangar capacity for C-checks
    max_a_check: int  # Hangar capacity for A-checks
    start_day_interval: int  # Min days between C-check starts
    c_e_t: int  # Default C-check elapsed time
    max_tol_dy_c: int
    max_tol_fh_c: float
    max_tol_fc_c: float
    max_tol_dy_a: int
    max_tol_fh_a: float
    max_tol_fc_a: float
    max_tol_dy_d: int
    dtc_min: int  # Lower bound to prevent scheduling C-check too often
    dta_min: int  # Lower bound to prevent scheduling A-check too often
    c_task_types: dict[str, int]  # Task type -> slot code
    a_task_types: dict[str, int]  # Task type -> slot code
    # Paper's Eq. 28 penalties (P_a and P_d)
    tolerance_penalty: float = 0.0  # P_a: penalty when tolerance is used (requires regulatory approval)
    ground_waiting_penalty: float = 0.0  # P_d: penalty when aircraft is grounded waiting for slot
    d_check_min_tat: int = 21  # Minimum TAT (days) for D-checks (heavy C-checks)
    
    def __post_init__(self):
        """Set penalty defaults from m_cost if not explicitly provided."""
        if self.tolerance_penalty == 0.0:
            self.tolerance_penalty = self.m_cost
        if self.ground_waiting_penalty == 0.0:
            self.ground_waiting_penalty = self.m_cost


@dataclass
class MaintenanceData:
    """Complete maintenance data from the Excel file."""
    d_initial: dict[str, AircraftDInitial]  # tail -> D-check initial
    c_initial: dict[str, AircraftCInitial]  # tail -> C-check initial
    a_initial: dict[str, AircraftAInitial]  # tail -> A-check initial
    flight_rates: dict[str, FlightRates]  # tail -> flight rates
    c_check_info: dict[str, CCheckInfo]  # tail -> C-check codes/times
    config: Config
    reserved: list[ReservedMaintenance]
    c_peak: list[PeakSeason]
    c_not_allowed: list[DateRange]
    a_not_allowed: list[date]
    more_c_slots: list[AdditionalSlots]
    more_a_slots: dict[date, int]  # date -> additional slots
    public_holidays: list[date]
    aircraft_tails: list[str]  # Ordered list of aircraft


def _parse_date(val) -> Optional[date]:
    """Parse a date value from Excel, handling -1 as None."""
    if pd.isna(val) or val == -1:
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        return datetime.strptime(val, "%Y-%m-%d").date()
    return None


def load_maintenance_data(excel_path: str) -> MaintenanceData:
    """Load all maintenance data from the Excel file."""
    xl = pd.ExcelFile(excel_path)

    # Load D_INITIAL
    df_d = pd.read_excel(xl, "D_INITIAL")
    d_initial = {}
    for _, row in df_d.iterrows():
        tail = row["A/C TAIL"]
        d_initial[tail] = AircraftDInitial(
            fleet=row["FLEET"],
            tail=tail,
            d_cycle=int(row["D-CYCLE"]),
            dy_d=int(row["DY-D"]),
            d_ci_dy=int(row["D-CI-DY"]),
            d_tol_dy=int(row["D-TOL-DY"]),
            d_tolu_dy=int(row["D-TOLU-DY"]),
            d_max=int(row["D-MAX"]),
            d_end=_parse_date(row["D-END"]),
            operation=_parse_date(row["OPERATION"]) or date(2017, 9, 25),
        )

    # Load C_INITIAL
    df_c = pd.read_excel(xl, "C_INITIAL")
    c_initial = {}
    for _, row in df_c.iterrows():
        tail = row["A/C TAIL"]
        c_initial[tail] = AircraftCInitial(
            fleet=row["FLEET"],
            tail=tail,
            c_sn=float(row["C-SN"]),
            c_max=int(row["C-MAX"]),
            dy_c=int(row["DY-C"]),
            fh_c=float(row["FH-C"]),
            fc_c=float(row["FC-C"]),
            c_ci_dy=int(row["C-CI-DY"]),
            c_ci_fh=float(row["C-CI-FH"]),
            c_ci_fc=float(row["C-CI-FC"]),
            c_tol_dy=int(row["C-TOL-DY"]),
            c_tol_fh=float(row["C-TOL-FH"]),
            c_tol_fc=float(row["C-TOL-FC"]),
            c_tolu_dy=int(row["C-TOLU-DY"]),
            c_tolu_fh=float(row["C-TOLU-FH"]),
            c_tolu_fc=float(row["C-TOLU-FC"]),
            c_start=_parse_date(row["C-START"]),
            c_end=_parse_date(row["C-END"]),
        )

    # Load A_INITIAL
    df_a = pd.read_excel(xl, "A_INITIAL")
    a_initial = {}
    for _, row in df_a.iterrows():
        tail = row["A/C TAIL"]
        a_initial[tail] = AircraftAInitial(
            fleet=row["FLEET"],
            tail=tail,
            a_sn=float(row["A-SN"]),
            a_max=int(row["A-MAX"]),
            dy_a=int(row["DY-A"]),
            fh_a=float(row["FH-A"]),
            fc_a=float(row["FC-A"]),
            a_ci_dy=int(row["A-CI-DY"]),
            a_ci_fh=float(row["A-CI-FH"]),
            a_ci_fc=float(row["A-CI-FC"]),
            a_tol_dy=int(row["A-TOL-DY"]),
            a_tol_fh=float(row["A-TOL-FH"]),
            a_tol_fc=float(row["A-TOL-FC"]),
            a_tolu_dy=int(row["A-TOLU-DY"]),
            a_tolu_fh=float(row["A-TOLU-FH"]),
            a_tolu_fc=float(row["A-TOLU-FC"]),
            a_end=_parse_date(row["A-END"]),
        )

    # Load DFH and DFC
    df_dfh = pd.read_excel(xl, "DFH")
    df_dfc = pd.read_excel(xl, "DFC")
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

    flight_rates = {}
    for _, row in df_dfh.iterrows():
        tail = row["A/C TAIL"]
        dfh = {i + 1: float(row[m]) for i, m in enumerate(months)}
        flight_rates[tail] = FlightRates(
            fleet=row["FLEET"],
            tail=tail,
            dfh=dfh,
            dfc={},  # Will be filled from DFC
        )

    for _, row in df_dfc.iterrows():
        tail = row["A/C TAIL"]
        dfc = {i + 1: float(row[m]) for i, m in enumerate(months)}
        if tail in flight_rates:
            flight_rates[tail].dfc = dfc

    # Load C_CHECK_CODE and C_ELAPSED_TIME
    df_code = pd.read_excel(xl, "C_CHECK_CODE")
    df_time = pd.read_excel(xl, "C_ELAPSED_TIME")

    c_check_info = {}
    code_cols = [c for c in df_code.columns if c.startswith("C CODE")]
    time_cols = [c for c in df_time.columns if c.startswith("TAT")]

    for idx, row in df_code.iterrows():
        tail = row["A/C TAIL"]
        codes = [float(row[c]) for c in code_cols]
        times_row = df_time.iloc[idx]
        times = [int(times_row[c]) for c in time_cols]
        c_check_info[tail] = CCheckInfo(
            fleet=row["FLEET"],
            tail=tail,
            check_codes=codes,
            elapsed_times=times,
        )

    # Load ADDITIONAL config
    df_add = pd.read_excel(xl, "ADDITIONAL")
    config_dict = {}
    c_task_types = {}
    a_task_types = {}

    for _, row in df_add.iterrows():
        key = str(row.iloc[0]).strip()
        val = row.iloc[1]

        if key == "C TASK TYPE":
            c_task_types[str(val)] = int(row.iloc[2]) if pd.notna(row.iloc[2]) else 0
        elif key == "A TASK TYPE":
            a_task_types[str(val)] = int(row.iloc[2]) if pd.notna(row.iloc[2]) else 0
        else:
            config_dict[key] = val

    begin_day = _parse_date(config_dict.get("BEGIN DAY")) or date(2017, 9, 25)

    config = Config(
        begin_year=int(config_dict.get("BEGIN YEAR", 2017)),
        total_years=int(config_dict.get("TOTAL YEARS", 4)),
        begin_day=begin_day,
        m_cost=float(config_dict.get("M COST", 100000000)),
        c_cost=float(config_dict.get("C COST", 100000)),
        max_c_check=int(config_dict.get("MAX C CHECK", 3)),
        max_a_check=int(config_dict.get("MAX A CHECK", 1)),
        start_day_interval=int(config_dict.get("START DAY INTERVAL", 3)),
        c_e_t=int(config_dict.get("C_E_T", 23)),
        max_tol_dy_c=int(config_dict.get("MAX TOL DY C", 0)),
        max_tol_fh_c=float(config_dict.get("MAX TOL FH C", 0)),
        max_tol_fc_c=float(config_dict.get("MAX TOL FC C", 0)),
        max_tol_dy_a=int(config_dict.get("MAX TOL DY A", 0)),
        max_tol_fh_a=float(config_dict.get("MAX TOL FH A", 0)),
        max_tol_fc_a=float(config_dict.get("MAX TOL FC A", 0)),
        max_tol_dy_d=int(config_dict.get("MAX TOL DY D", 0)),
        dtc_min=int(config_dict.get("DTC MIN", 365)),
        dta_min=int(config_dict.get("DTA MIN", 21)),
        c_task_types=c_task_types,
        a_task_types=a_task_types,
    )

    # Load RESERVED
    df_res = pd.read_excel(xl, "RESERVED")
    reserved = []
    for _, row in df_res.iterrows():
        if pd.notna(row["A/C TAIL"]):
            reserved.append(
                ReservedMaintenance(
                    fleet=row["FLEET"],
                    tail=row["A/C TAIL"],
                    reserved_type=row["RESERVED TYPE"],
                    begin=_parse_date(row["RESERVED BEGIN"]),
                    end=_parse_date(row["RESERVED END"]),
                    time=int(row["RESERVED TIME"]),
                    slot=row["RESERVED SLOT"],
                )
            )

    # Load C_PEAK
    df_peak = pd.read_excel(xl, "C_PEAK")
    c_peak = []
    for _, row in df_peak.iterrows():
        c_peak.append(
            PeakSeason(
                year=int(row["YEAR"]),
                begin=_parse_date(row["PEAK BEGIN"]),
                end=_parse_date(row["PEAK END"]),
            )
        )

    # Load C_NOT_ALLOWED
    df_cna = pd.read_excel(xl, "C_NOT_ALLOWED")
    c_not_allowed = []
    for _, row in df_cna.iterrows():
        c_not_allowed.append(
            DateRange(
                begin=_parse_date(row["BEGIN"]),
                end=_parse_date(row["END"]),
            )
        )

    # Load A_NOT_ALLOWED
    df_ana = pd.read_excel(xl, "A_NOT_ALLOWED")
    a_not_allowed = [_parse_date(row["DATE"]) for _, row in df_ana.iterrows()]

    # Load MORE_C_SLOTS
    df_mcs = pd.read_excel(xl, "MORE_C_SLOTS")
    more_c_slots = []
    for _, row in df_mcs.iterrows():
        more_c_slots.append(
            AdditionalSlots(
                begin=_parse_date(row["BEGIN"]),
                end=_parse_date(row["END"]),
                total_slots=int(row["TOTAL SLOTS"]),
            )
        )

    # Load MORE_A_SLOTS
    df_mas = pd.read_excel(xl, "MORE_A_SLOTS")
    more_a_slots = {}
    for _, row in df_mas.iterrows():
        d = _parse_date(row["DATE"])
        if d:
            more_a_slots[d] = int(row["TOTAL SLOTS"])

    # Load PUBLIC_HOLIDAYS
    df_ph = pd.read_excel(xl, "PUBLIC_HOLIDAYS")
    public_holidays = [_parse_date(row["DATE"]) for _, row in df_ph.iterrows()]

    # Get ordered list of aircraft tails
    aircraft_tails = list(d_initial.keys())

    return MaintenanceData(
        d_initial=d_initial,
        c_initial=c_initial,
        a_initial=a_initial,
        flight_rates=flight_rates,
        c_check_info=c_check_info,
        config=config,
        reserved=reserved,
        c_peak=c_peak,
        c_not_allowed=c_not_allowed,
        a_not_allowed=a_not_allowed,
        more_c_slots=more_c_slots,
        more_a_slots=more_a_slots,
        public_holidays=public_holidays,
        aircraft_tails=aircraft_tails,
    )


if __name__ == "__main__":
    # Test loading
    data = load_maintenance_data("Sep 25 2017 (Case Study 2018-2021).xlsx")
    print(f"Loaded {len(data.aircraft_tails)} aircraft")
    print(f"Config: {data.config}")
    print(f"First aircraft: {data.aircraft_tails[0]}")
    print(f"  D-initial: {data.d_initial[data.aircraft_tails[0]]}")
    print(f"  C-initial: {data.c_initial[data.aircraft_tails[0]]}")
    print(f"  A-initial: {data.a_initial[data.aircraft_tails[0]]}")


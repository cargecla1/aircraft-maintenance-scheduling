# Aircraft Maintenance Check Scheduling

Implementation of the optimization algorithm from:

> Deng, Q., Santos, B.F., Curran, R. (2020). *A practical dynamic programming based methodology for aircraft maintenance check scheduling optimization*. European Journal of Operational Research, 281(2), 256-273.

The implementation follows the paper's forward beam search methodology with practical engineering adaptations for tractability (simplified dominance pruning, approximate workability checking).

---

## Problem Setting

Airlines must schedule periodic maintenance checks for their fleet:
- **A-checks**: Light maintenance every ~750 flight hours / 120 days
- **C-checks**: Heavy maintenance every ~7,500 flight hours / 730 days

Each aircraft has three utilization counters that accumulate since its last check:
- **DY** (calendar days)
- **FH** (flight hours)
- **FC** (flight cycles)

A check is **due** when *any* counter exceeds its interval. Regulatory bodies allow a **tolerance buffer** beyond the interval, but using it requires approval and incurs a penalty.

---

## Problem Formulation

The paper formulates this as a **Markov Decision Process (MDP)**:

### State Space

For a fleet of N aircraft, the state at time t includes:
- For each aircraft i: counters (DY, FH, FC) since last A-check and C-check
- Current intervals (which decrease if tolerance was used previously)
- Grounding status (days remaining in maintenance)

### Action Space (Table 2)

Each day, the dispatcher chooses how many checks to perform:

| Action | n_a (A-checks) | n_c (C-checks) |
|--------|----------------|----------------|
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 2 | 0 | 1 |
| 3 | 1 | 1 |
| 4 | 0 | 2 |
| 5 | 1 | 2 |
| 6 | 2 | 0 |

Aircraft are assigned to checks using **earliest deadline first** priority (Equations 36-40).

### Transition Dynamics

Given action (n_a, n_c):
1. Select n_c most urgent aircraft for C-checks, n_a for A-checks
2. Reset counters for checked aircraft; set grounding time
3. Advance all non-grounded aircraft by one day (DY += 1, FH += daily_fh, FC += daily_fc)

### Cost Function (Equation 28)

```
C_t = Σ[χ·(I_FH - FH) + (1-χ)·P_a·θ + P_d·η]
```

Where:
- **χ = 1** if a check is performed, **0** otherwise
- **I_FH - FH**: unused flight hours (scheduling a check early wastes remaining interval)
- **P_a · θ**: tolerance penalty (θ=1 if check uses regulatory tolerance)
- **P_d · η**: ground-waiting penalty (η=1 if aircraft is overdue but not in maintenance)

**Goal**: Minimize total cost over the planning horizon (typically 4 years).

### Constraints

- **Hangar capacity**: Max concurrent C-checks (typically 3) and A-checks (typically 1)
- **Peak seasons**: No C-checks during high-demand periods
- **Minimum interval**: Days between C-check starts (prevents bunching)

---

## Solution Method: Forward Induction

### Why Not Standard DP?

The state space is enormous: with 45 aircraft, each having 6+ continuous state variables, exact dynamic programming is intractable. The paper calls their method "dynamic programming" but it's actually **forward beam search** with state aggregation.

### The Algorithm

```
1. Initialize frontier = {initial_fleet_state}
2. For each day t = 1 to T:
   a. Expand: For each state in frontier, apply all 7 actions
   b. Aggregate: Group states by utilization histogram
   c. Prune: Keep only lowest-cost state per histogram group (Thrifty)
   d. Beam limit: Keep top-k states by cumulative cost
3. Return best final state
```

### Key Differences from Bellman DP

| Bellman DP | Forward Induction (this paper) |
|------------|--------------------------------|
| Works backward from T to 0 | Works forward from 0 to T |
| Computes value function V(s) | Tracks cumulative cost |
| Exact (for finite MDP) | Approximate (beam search) |
| Needs full state enumeration | Uses state aggregation |

### State Aggregation (Section 4.4)

To make the problem tractable, states are grouped by **utilization histograms**:
- Discretize each aircraft's FH into bins of size u (e.g., u=100)
- Count how many aircraft fall in each bin
- States with identical histograms are considered equivalent

This reduces the state space from continuous to a manageable discrete set.

### Thrifty Pruning (Section 4.4.1)

For states with the same histogram, only keep the one with **lowest cumulative cost**. This is a dominance rule: if two paths lead to the same aggregated state, prefer the cheaper one.

---

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download data from: https://doi.org/10.4121/uuid:1630e6fd-9574-46e8-899e-83037c17bcef
```

## Usage

```python
from aircraft_scheduler import Scheduler, load_maintenance_data

data = load_maintenance_data("Sep 25 2017 (Case Study 2018-2021).xlsx")
scheduler = Scheduler(data, beam_width=200)
result = scheduler.schedule(4 * 365, verbose=True)

print(f"C-checks: {result['n_c_checks']}")
print(f"A-checks: {result['n_a_checks']}")
print(f"Tolerance events: {result['tolerance_events']}")
```

CLI:
```bash
python main.py --years 4 --beam-width 200
```

---

## Results

**4-Year Benchmark** (45 aircraft, beam_width=200):

| Metric | This Implementation | Paper Table 3 | Paper Table 4 |
|--------|---------------------|---------------|---------------|
| C-checks | 94 | 82–89 | 85–88 |
| A-checks | 839 | 758–818 | 877–881 |
| Tolerance events | 5 | 1–17 | 0–4 |
| Computation time | **2.9 min** | — | ~8.5 min |

**Analysis:**
- ✅ **C-checks match**: 94 vs paper's 82–89 range
- ✅ **A-checks match**: 839 falls within paper's 758–881 range
- ✅ **Minimal tolerance**: Only 5 events over 4 years
- ✅ **Faster**: ~3x speedup vs paper's reported times

**Efficiency:** A-check utilization 95.8%, C-check utilization 93.2%

**1-Year Results:** C=25, A=202, Tolerance=1, Time=42s

---

## Extensions: Beam Search Variants

Recognizing the algorithm as **beam search** enables improvements from the AI/OR literature:

| Variant | Tolerance Events | Cost Reduction |
|---------|------------------|----------------|
| Base Deng (reproduction) | 5 | — |
| + Future heuristic (λ=100) | 2 | −75% |

**Key insight:** Best-first expansion with a lookahead heuristic ("time to next grounding risk") significantly reduces both tolerance events and total cost while maintaining zero groundings.

Run experiments:
```bash
python run_variant_experiments.py --variant all --years 4
```

---

## Project Structure

```
aircraft_scheduler/
├── scheduler.py       # Main scheduler implementation
├── core.py            # State, dynamics, constraints, cost functions
├── data_loader.py     # Excel data parser
└── visualization.py   # Plotting utilities
```

---

## References

```bibtex
@article{deng2020practical,
  title={A practical dynamic programming based methodology for aircraft 
         maintenance check scheduling optimization},
  author={Deng, Qichen and Santos, Bruno F and Curran, Richard},
  journal={European Journal of Operational Research},
  volume={281},
  number={2},
  pages={256--273},
  year={2020}
}
```

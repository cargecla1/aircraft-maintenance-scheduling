# Aircraft Maintenance Check Scheduling

Dynamic programming approach to optimize long-term aircraft maintenance schedules, minimizing wasted intervals between checks.

Based on: Deng, Q., Santos, B.F., Curran, R. (2020). *A practical dynamic programming based methodology for aircraft maintenance check scheduling optimization*. European Journal of Operational Research, 281(2), 256-273.

## Problem Setting

Airlines must schedule periodic A-checks (every ~120 days/750 FH) and C-checks (every ~730 days/7500 FH) for their fleet. Checks are triggered by calendar days (DY), flight hours (FH), or flight cycles (FC)—whichever limit is reached first.

**Constraints:**
- Limited hangar capacity (e.g., 3 concurrent C-checks, 1 A-check)
- No C-checks during peak commercial seasons
- Minimum days between C-check starts
- A-checks can merge into C-checks

**Objective:** Minimize *wasted intervals*—the gap between prescribed check intervals and actual usage at check time—thereby reducing total checks and increasing aircraft availability.

## MDP Formulation

**State** `s_t`: For each aircraft, track cumulative DY/FH/FC since last A-check and C-check, plus tolerance used from previous checks.

**Action** `x_t`: Binary decisions for each aircraft—schedule A-check, C-check, or neither.

**Transition** `s_{t+1} = f(s_t, x_t)`:
- If no check: aircraft ages by 1 day + daily FH/FC
- If check scheduled: reset counters, ground aircraft for check duration

**Cost** `c(s_t, x_t)`: Sum of wasted intervals for scheduled checks:
```
wasted = (interval_DY - actual_DY) + (interval_FH - actual_FH) + (interval_FC - actual_FC)
```

## Solution Method

**Forward Induction DP** with rolling horizon:

1. At each time step, evaluate feasible actions given capacity constraints
2. Compute immediate cost (wasted interval) + estimated future cost
3. Select action minimizing total cost
4. Advance fleet state by one day

Key techniques from the paper:
- State aggregation by tolerance signature to reduce state space
- Greedy priority ordering (earliest deadline first)
- Mandatory check detection with lookahead

## Usage

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download data from https://doi.org/10.4121/uuid:1630e6fd-9574-46e8-899e-83037c17bcef

python main.py --algorithm dp --years 4
python main.py --compare  # Compare greedy vs DP
```

## Results

On the paper's dataset (45 aircraft, 4-year horizon):

| Metric | Greedy | DP |
|--------|--------|-----|
| C+D checks | 106 | 93 (-12%) |
| Wasted interval | 430,555 | 372,174 (-14%) |
| Computation | 4.1s | 4.1s |

## Files

- `main.py` - CLI entry point
- `solver.py` - Greedy and DP scheduling algorithms  
- `state.py` - Aircraft/fleet state representation
- `dynamics.py` - State transitions and aging
- `cost.py` - Wasted interval calculations
- `constraints.py` - Capacity and holiday constraints
- `data_loader.py` - Excel parser
- `visualization.py` - Gantt charts and plots

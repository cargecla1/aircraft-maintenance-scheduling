# Aircraft Maintenance Check Scheduling

Implementation of the forward beam search algorithm from Deng, Santos & Curran (2020), with practical adaptations for tractability.

> Deng, Q., Santos, B.F., Curran, R. (2020). A practical dynamic programming based methodology for aircraft maintenance check scheduling optimization. *European Journal of Operational Research*, 281(2), 256-273.

## Problem

Airlines schedule A-checks (~750 FH, 1 day) and C-checks (~7500 FH, 10-30 days) for their fleet. Each aircraft tracks three counters since its last check: calendar days (DY), flight hours (FH), and flight cycles (FC). A check is due when any counter exceeds its interval.

The objective (Equation 28) minimizes unused flight hours plus penalties for tolerance usage and ground-waiting:

```
C_t = Σ[χ·(I_FH - FH) + P_a·θ + P_d·η]
```

## Algorithm

The paper's "practical DP" is forward beam search with state aggregation. Each day, expand all 7 actions from Table 2, group successor states by utilization histogram, keep only the lowest-cost state per histogram bucket, and advance.

```
aircraft_scheduler/
├── scheduler.py       # Beam search implementation
├── core.py            # State, dynamics, cost functions
├── data_loader.py     # Excel parser
└── visualization.py   # Plotting
```

## Usage

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py --years 4 --beam-width 200
```

Data: https://doi.org/10.4121/uuid:1630e6fd-9574-46e8-899e-83037c17bcef

## Results

4-year benchmark (45 aircraft, beam_width=200):

| Metric | Implementation | Paper |
|--------|----------------|-------|
| C-checks | 94 | 82-89 |
| A-checks | 839 | 758-881 |
| Tolerance events | 5 | 0-17 |
| Time | 2.9 min | ~8.5 min |

Efficiency: A-check 95.8%, C-check 93.2%.

Adding a future-risk heuristic reduces tolerance events from 5 to 2 and cost by 75%:

```bash
python run_variant_experiments.py --variant all --years 4
```

## Implementation Notes

The paper's algorithm requires several expensive operations that we approximate for tractability.

**State aggregation.** Beam search generates many successor states each day. To bound memory, the paper groups "similar" states and keeps only one representative per group. The paper's grouping criterion is mean fleet utilization: for each check type, compute ū = (1/N) Σ (FH_i / I_FH), round to the nearest 0.1, and use the pair (ū_A, ū_C) as the group key. With 11 possible values per check type, this yields at most 121 groups.

The problem is that mean utilization discards distributional information. Consider two fleets of 10 aircraft. Fleet A has all aircraft at 50% C-check utilization. Fleet B has 5 aircraft at 0% and 5 at 100%. Both have mean 50%, so the paper treats them as equivalent. But Fleet B has 5 aircraft about to exceed their interval, while Fleet A has none. These fleets need different scheduling decisions.

We instead group by histogram: discretize each aircraft's utilization into bins, count how many aircraft fall in each bin, and use this count vector as the group key. Fleet A has histogram {50%: 10}, Fleet B has histogram {0%: 5, 100%: 5}. These are different keys, so the states remain separate. The tradeoff is more groups (larger state space), but we preserve information about fleet heterogeneity that affects scheduling.

**Dominance pruning.** When multiple states fall in the same histogram bucket, we must decide which to keep. The paper's "Thrifty" rule compares states bin-by-bin: state A dominates B if A has higher utilization in every bin and lower cost. This requires O(n²) comparisons per bucket. We simplify to keeping only the lowest-cost state per bucket, which is faster but may discard states that would lead to better solutions.

**Workability check.** A state is "workable" if some policy can reach the horizon without grounding any aircraft. The paper defines this existentially: does any feasible action sequence exist? Checking this exactly is expensive. We instead simulate a single greedy policy (always check the most urgent aircraft) for a configurable lookahead horizon. If greedy fails, we reject the state. This is conservative: we may reject workable states, but we never accept unworkable ones.

**A-check resolution.** A-checks have a 750 FH interval versus 7500 FH for C-checks. Using the same discretization step u for both loses detail for A-checks. We use u/2 for A-check histograms to preserve resolution at the shorter interval.

**Penalty values.** The paper distinguishes P_a (tolerance penalty) from P_d (ground-waiting penalty) conceptually, but both are set to 10^8 in the case study. We use the same value for both.

## Reference

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

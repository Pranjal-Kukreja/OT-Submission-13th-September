# Transportation Problem — VAM (Initial BFS) + MODI (Optimality & Improvement)

A Python implementation that solves a classic **Transportation Problem**
in two required stages:

1. **Vogel's Approximation Method (VAM)** — builds a good initial basic
   feasible solution (BFS).
2. **MODI (Modified Distribution) Method** — tests that BFS for
   optimality using row/column dual values (`u_i`, `v_j`), and
   iteratively improves it (stepping-stone reallocation around a
   closed loop) until no improvement is possible.

## 1. Problem Statement

A company has **3 factories** (sources) shipping a single product to
**4 warehouses** (destinations). Supplies, demands, and per-unit
shipping costs are:

|          | D1 | D2 | D3 | D4 | Supply |
|----------|----|----|----|----|--------|
| **S1**   | 19 | 30 | 50 | 10 | 7      |
| **S2**   | 70 | 30 | 40 | 60 | 9      |
| **S3**   | 40 | 8  | 70 | 20 | 18     |
| Demand   | 5  | 8  | 7  | 14 | **34** |

Total supply (34) = total demand (34), so the problem is already
**balanced**. (The program also handles unbalanced cases automatically
by adding a zero-cost dummy source/destination.)

**Goal:** find the shipment plan `x_ij` (units shipped from factory i
to warehouse j) that minimizes total transportation cost
`Σ c_ij · x_ij`, subject to each factory's supply and each warehouse's
demand being exactly met.

## 2. Repository Contents

| File | Description |
|---|---|
| `transportation_vam_modi.py` | Main program. Generic, reusable functions for VAM, degeneracy handling, and MODI, applied to the case-study data. Prints every VAM allocation step and every MODI iteration (dual values, opportunity costs, closed loop, reallocation). |
| `verify_with_scipy.py` | Independent cross-check of the result by formulating the transportation problem as an equality-constrained LP and solving with `scipy.optimize.linprog` (HiGHS). |
| `sample_output.txt` | Saved console output of a full run, for quick reference without re-running the code. |
| `requirements.txt` | Python dependencies. |

## 3. Method Details

### Step 1 — VAM (initial BFS)
At every step:
- Compute the **penalty** for each remaining row/column = difference
  between the two lowest unit costs among its still-active cells.
- Pick the row/column with the **largest penalty**.
- Allocate as much as possible (`min(supply, demand)`) to the
  **cheapest cell** in that row/column.
- Cross out whichever supply or demand is now exhausted, and repeat
  until all supply/demand is allocated.

### Step 2 — MODI (optimality test & improvement)
- Compute dual values `u_i`, `v_j` from the basic cells using
  `u_i + v_j = c_ij` (set `u_1 = 0` and propagate — the basic cells
  form a spanning tree over the sources/destinations).
- Compute the opportunity cost `d_ij = c_ij − (u_i + v_j)` for every
  **non-basic** cell.
- **Optimal** when every `d_ij ≥ 0`.
- Otherwise, the most negative `d_ij` cell **enters** the basis. Its
  unique closed loop through the current basic cells is found
  (alternating `+`/`−` signs), and `θ = min` allocation among the
  `−` cells is shifted around the loop (stepping-stone). The cell
  that hits zero **leaves** the basis. Repeat.
- **Degeneracy** (fewer than `m + n − 1` basic cells) is detected and
  fixed automatically by adding a tiny (`ε = 1e-6`) allocation to a
  cell chosen (via union-find) so the basis stays a spanning tree.

## 4. How to Run

```bash
pip install -r requirements.txt

python transportation_vam_modi.py     # runs VAM, then MODI, printing every step
python verify_with_scipy.py           # independent check against SciPy's linprog
```

## 5. Result

**VAM initial solution:** total cost = 779

**After one MODI iteration** (S2,D2 enters; S2,D4 leaves), the
solution becomes optimal:

|          | D1 | D2 | D3 | D4 |
|----------|----|----|----|----|
| **S1**   | 5  | 0  | 0  | 2  |
| **S2**   | 0  | 2  | 7  | 0  |
| **S3**   | 0  | 6  | 0  | 12 |

**Minimum total transportation cost = 743**

This exactly matches the independent verification against
`scipy.optimize.linprog` (both the allocation and the total cost),
confirming the correctness of the VAM + MODI implementation.

## 6. Notes

- The code was additionally stress-tested on an **unbalanced** problem
  and a **degenerate** problem (both included in development, not
  part of the case-study run) — in both cases the VAM+MODI result
  matched SciPy's LP solution exactly.
- `vam_initial_solution`, `fix_degeneracy`, and `modi_solve` are
  written generically, so the same code can be reused for any other
  `m x n` transportation problem by simply changing `supply`,
  `demand`, and `cost` in `solve_case_study()`.

## Author

Submitted as part of a class assignment on Optimization Techniques /
Operations Research — Transportation Problem (VAM + MODI Method).

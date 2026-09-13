# Big-M Simplex Method — The Diet Problem

A Python implementation of the **Big-M Simplex Method** used to solve a
well-known constrained optimization problem: the classic **Diet
Problem** from Linear Programming, formulated, converted to standard
form (with slack / surplus / artificial variables), and solved
computationally end-to-end.

## 1. Problem Statement

Minimize the cost of a two-food diet that must meet two minimum
nutrient requirements:

```
Minimize   Z  = 2x1 + 3x2                (total cost)

subject to
    x1 + 2x2 >= 40      (minimum requirement of Nutrient A)
    x1 +  x2 >= 30      (minimum requirement of Nutrient B)
    x1, x2 >= 0
```

Because both constraints are of the `>=` type, an identity starting
basis is not directly available from slack variables alone. The
**Big-M Method** handles this by:

1. Subtracting a **surplus variable** from each `>=` constraint to
   convert it to an equality.
2. Adding an **artificial variable** to each equality to obtain a
   ready starting basic feasible solution.
3. Penalizing every artificial variable with a very large cost `M` in
   the objective function, so the simplex algorithm is forced to drive
   them out of the basis (to zero) as it searches for optimality.

### Standard form used by the solver

```
x1 + 2x2 - s1 + a1                = 40
x1 +  x2       - s2 + a2          = 30

Minimize  Z = 2x1 + 3x2 + 0·s1 + 0·s2 + M·a1 + M·a2
x1, x2, s1, s2, a1, a2 >= 0
```

## 2. Repository Contents

| File | Description |
|---|---|
| `big_m_simplex.py` | Main program. Contains a **general-purpose** `BigMSimplex` tableau solver (handles any mix of `<=`, `>=`, `=` constraints and `min`/`max` objectives) and applies it to the Diet Problem, printing every simplex iteration. |
| `verify_with_scipy.py` | Independent cross-check of the result using `scipy.optimize.linprog` (HiGHS solver). |
| `sample_output.txt` | Saved console output from a full run (all iterations + final answer + SciPy verification), for quick reference without re-running the code. |
| `requirements.txt` | Python dependencies. |

## 3. How the Solver Works

`BigMSimplex` builds the full simplex tableau (decision, slack,
surplus and artificial variable columns), then repeats the standard
simplex loop:

1. Compute the **`cj - zj`** (net evaluation) row.
2. **Optimality test:** stop when no `cj - zj` value is positive
   (problem is treated internally as a maximization; a `min` problem
   is solved by maximizing `-Z` and negating the result at the end).
3. **Entering variable:** the column with the most positive `cj - zj`.
4. **Leaving variable:** minimum-ratio test (`RHS / pivot column`,
   only over rows with a positive pivot-column entry).
5. **Pivot** (Gauss-Jordan row reduction) and repeat.
6. At optimality, verify all artificial variables are `0` — otherwise
   the original LPP is infeasible.

## 4. How to Run

```bash
pip install -r requirements.txt

python big_m_simplex.py          # runs the Big-M simplex solver, prints every iteration
python verify_with_scipy.py      # independent check against SciPy's linprog
```

## 5. Result

| Variable | Optimal value |
|---|---|
| x1 | 20 |
| x2 | 10 |
| **Z\*** (minimum cost) | **70** |

**Constraint check at the optimum:**
`x1 + 2x2 = 40` ✅ (exactly meets requirement) &nbsp;&nbsp;
`x1 + x2 = 30` ✅ (exactly meets requirement)

This matches the independent verification against `scipy.optimize.linprog`
(HiGHS solver), confirming the correctness of the Big-M simplex
implementation.

## 6. Notes

- `M` is represented numerically as `1e6` inside the program (a
  conventional practical choice for the Big-M method; it can be
  changed in `big_m_simplex.py` via the `BIG_M` constant).
- The `BigMSimplex` class is written generically, so it can be reused
  for other LPPs (any combination of `<=`, `>=`, `=` constraints, and
  either `min` or `max` objectives) by simply changing `c`, `A`, `b`,
  `constraint_types`, and `sense`.

## Author

Submitted as part of a class assignment on Optimization Techniques /
Operations Research — Big-M Simplex Method.

"""
Big-M Simplex Method - Computational Solver
=============================================

Case Study : THE DIET PROBLEM (a classic, well-known LPP)
------------------------------------------------------------
A person wants to design the cheapest possible diet using two food
items (x1, x2) while satisfying minimum daily nutrient requirements.

    Minimize   Z  = 2*x1 + 3*x2                (cost, Rs./unit)

    subject to
        x1 + 2*x2  >= 40          (minimum Vitamin-A requirement)
        x1 +   x2  >= 30          (minimum Vitamin-B requirement)
        x1, x2 >= 0

Because both structural constraints are of the ">=" type, simple
slack variables alone cannot provide a ready (identity) starting
basis. We therefore introduce SURPLUS variables (to convert ">=" to
"=") and ARTIFICIAL variables (to obtain an initial basic feasible
solution), and penalize the artificial variables with a very large
positive cost M in the objective function -> this is exactly the
"Big-M Method".

Standard form used internally
------------------------------
    x1 + 2x2 - s1 + a1              = 40
    x1 +  x2       - s2 + a2        = 30
    x1,x2,s1,s2,a1,a2 >= 0

    Minimize Z = 2x1 + 3x2 + 0.s1 + 0.s2 + M.a1 + M.a2

The program below implements a GENERAL Big-M Simplex tableau solver
(works for any mix of <=, >=, = constraints, min or max objective)
and then applies it to the Diet Problem as the worked case study.
It prints every simplex iteration (tableau, entering/leaving
variables, ratio test) and finally reports the optimal decision
variable values and the optimal objective value.

Author : <your name>
Course : Operations Research / Optimization Techniques
"""

import numpy as np
import pandas as pd

pd.set_option("display.float_format", lambda x: f"{x:10.3f}")
pd.set_option("display.width", 140)

BIG_M = 1e6          # numerical value used to represent the symbolic "M"
TOL = 1e-9            # numerical tolerance for optimality / ratio tests


class BigMSimplex:
    """
    A general purpose Big-M Simplex solver.

    Parameters
    ----------
    c : list[float]
        Objective function coefficients of the *original* decision
        variables, in the ORIGINAL sense given by `sense`.
    A : list[list[float]]
        Constraint coefficient matrix (rows = constraints).
    b : list[float]
        RHS values (must be made non-negative internally).
    constraint_types : list[str]
        One of '<=', '>=', '=' for every row of A.
    sense : str
        'min' or 'max' -> sense of the ORIGINAL objective function.
    var_names : list[str], optional
        Names for the original decision variables (for pretty printing).
    """

    def __init__(self, c, A, b, constraint_types, sense="min", var_names=None):
        self.n_orig = len(c)
        self.m = len(A)
        self.sense = sense.lower()
        self.constraint_types = constraint_types
        self.var_names = var_names or [f"x{i+1}" for i in range(self.n_orig)]

        # Work internally as a MAXIMIZATION problem.
        # If original sense is 'min', maximize (-Z) instead; at the end
        # multiply the optimal value by -1 to recover the true minimum.
        sign = 1.0 if self.sense == "max" else -1.0
        self.obj_sign = sign
        c_internal = [sign * ci for ci in c]

        A = [row[:] for row in A]
        b = list(b)

        # Make every RHS non-negative (flip the row if needed)
        for i in range(self.m):
            if b[i] < 0:
                b[i] = -b[i]
                A[i] = [-a for a in A[i]]
                flip = {"<=": ">=", ">=": "<=", "=": "="}
                constraint_types[i] = flip[constraint_types[i]]

        self.col_names = list(self.var_names)
        self.cost = list(c_internal)

        n_rows = self.m
        tableau_cols = []
        for row in A:
            tableau_cols.append(list(row))

        artificial_rows = []

        for i, ctype in enumerate(constraint_types):
            if ctype == "<=":
                # + slack
                name = f"s{i+1}"
                self.col_names.append(name)
                self.cost.append(0.0)
                for r in range(n_rows):
                    tableau_cols[r].append(1.0 if r == i else 0.0)
            elif ctype == ">=":
                # - surplus, + artificial
                sname = f"s{i+1}"
                self.col_names.append(sname)
                self.cost.append(0.0)
                for r in range(n_rows):
                    tableau_cols[r].append(-1.0 if r == i else 0.0)

                aname = f"a{i+1}"
                self.col_names.append(aname)
                self.cost.append(-BIG_M)          # heavy penalty (max problem)
                for r in range(n_rows):
                    tableau_cols[r].append(1.0 if r == i else 0.0)
                artificial_rows.append((i, aname))
            elif ctype == "=":
                aname = f"a{i+1}"
                self.col_names.append(aname)
                self.cost.append(-BIG_M)
                for r in range(n_rows):
                    tableau_cols[r].append(1.0 if r == i else 0.0)
                artificial_rows.append((i, aname))
            else:
                raise ValueError("constraint_types must be one of '<=','>=','='")

        n_total = len(self.col_names)
        self.T = np.zeros((n_rows, n_total))
        for r in range(n_rows):
            self.T[r, :] = tableau_cols[r]
        self.rhs = np.array(b, dtype=float)
        self.cost = np.array(self.cost, dtype=float)

        # initial basis: slack where available, else artificial
        self.basis = [None] * n_rows
        for i, ctype in enumerate(constraint_types):
            if ctype == "<=":
                self.basis[i] = self.col_names.index(f"s{i+1}")
            else:
                self.basis[i] = self.col_names.index(f"a{i+1}")

        self.iteration = 0
        self.history = []

    # ------------------------------------------------------------------
    def _tableau_df(self):
        df = pd.DataFrame(self.T, columns=self.col_names)
        df.insert(0, "Basis", [self.col_names[i] for i in self.basis])
        df["RHS"] = self.rhs
        return df

    def _cj_zj_row(self):
        cB = self.cost[self.basis]
        zj = cB @ self.T
        cj_minus_zj = self.cost - zj
        z_value = cB @ self.rhs
        return cj_minus_zj, z_value

    def _print_iteration(self, entering=None, leaving=None, pivot_col=None,
                          pivot_row=None, ratios=None):
        cj_zj, z_val = self._cj_zj_row()
        print(f"\n--- Iteration {self.iteration} ---")
        df = self._tableau_df()
        print(df.to_string(index=False))
        cz = pd.Series(cj_zj, index=self.col_names, name="cj - zj")
        print(cz.to_frame().T.to_string(index=False))
        obj_display = z_val if self.sense == "max" else -z_val
        print(f"Current objective value (in original '{self.sense}' sense): "
              f"{obj_display:.4f}")
        if ratios is not None:
            r_series = pd.Series(ratios, index=[f"Row{i+1}" for i in range(self.m)])
            print("Ratio test (RHS / pivot column, only rows with positive entry shown):")
            print(r_series.dropna().to_string())
        if entering is not None:
            print(f"Entering variable : {self.col_names[pivot_col]}")
        if leaving is not None:
            print(f"Leaving variable  : {self.col_names[self.basis[pivot_row]]}")

    # ------------------------------------------------------------------
    def solve(self, verbose=True, max_iter=50):
        if verbose:
            print("=" * 90)
            print("INITIAL TABLEAU (starting basic feasible solution using artificial vars)")
            print("=" * 90)
            self._print_iteration()

        for _ in range(max_iter):
            cj_zj, _ = self._cj_zj_row()

            # Optimality check (maximization): stop when no cj-zj > 0
            if np.all(cj_zj <= TOL):
                break

            pivot_col = int(np.argmax(cj_zj))

            col = self.T[:, pivot_col]
            ratios = np.full(self.m, np.nan)
            for r in range(self.m):
                if col[r] > TOL:
                    ratios[r] = self.rhs[r] / col[r]

            if np.all(np.isnan(ratios)):
                raise RuntimeError("Problem is unbounded.")

            pivot_row = int(np.nanargmin(ratios))

            self.iteration += 1
            if verbose:
                self._print_iteration(entering=True, leaving=True,
                                       pivot_col=pivot_col, pivot_row=pivot_row,
                                       ratios=ratios)

            # ---- pivot operation ----
            pivot_element = self.T[pivot_row, pivot_col]
            self.T[pivot_row, :] /= pivot_element
            self.rhs[pivot_row] /= pivot_element

            for r in range(self.m):
                if r != pivot_row and abs(self.T[r, pivot_col]) > TOL:
                    factor = self.T[r, pivot_col]
                    self.T[r, :] -= factor * self.T[pivot_row, :]
                    self.rhs[r] -= factor * self.rhs[pivot_row]

            self.basis[pivot_row] = pivot_col

            if verbose:
                print("\n>>> Tableau AFTER pivoting:")
                self._print_iteration()

        else:
            raise RuntimeError("Max iterations exceeded without reaching optimality.")

        # ---- Feasibility check: artificial variables must be 0 in final basis ----
        for i, bi in enumerate(self.basis):
            name = self.col_names[bi]
            if name.startswith("a") and self.rhs[i] > TOL:
                raise RuntimeError(
                    "No feasible solution exists: an artificial variable "
                    f"({name}) remains positive ({self.rhs[i]:.4f}) in the optimal basis."
                )

        solution = {name: 0.0 for name in self.col_names}
        for i, bi in enumerate(self.basis):
            solution[self.col_names[bi]] = self.rhs[i]

        cB = self.cost[self.basis]
        z_internal = cB @ self.rhs
        optimal_value = z_internal if self.sense == "max" else -z_internal

        decision_values = {self.var_names[i]: solution.get(self.var_names[i], 0.0)
                            for i in range(self.n_orig)}

        return {
            "status": "optimal",
            "iterations": self.iteration,
            "decision_variables": decision_values,
            "objective_value": optimal_value,
            "full_solution": solution,
        }


# ==========================================================================
# CASE STUDY : THE DIET PROBLEM
# ==========================================================================
def solve_diet_problem(verbose=True):
    """
    Minimize   Z  = 2 x1 + 3 x2
    subject to
        x1 + 2 x2 >= 40
        x1 +   x2 >= 30
        x1, x2 >= 0
    """
    c = [2, 3]
    A = [
        [1, 2],
        [1, 1],
    ]
    b = [40, 30]
    constraint_types = [">=", ">="]

    solver = BigMSimplex(c, A, b, constraint_types, sense="min",
                          var_names=["x1", "x2"])
    result = solver.solve(verbose=verbose)
    return result


if __name__ == "__main__":
    print("#" * 90)
    print("BIG-M SIMPLEX METHOD  -  Case Study: THE DIET PROBLEM (a classic LPP)")
    print("#" * 90)
    print("""
Problem formulation
--------------------
Minimize   Z  = 2 x1 + 3 x2                       (cost of the diet)
subject to
    x1 + 2 x2 >= 40      (minimum requirement of Nutrient A)
    x1 +   x2 >= 30      (minimum requirement of Nutrient B)
    x1, x2 >= 0

Standard form (Big-M)
-----------------------
    x1 + 2x2 - s1 + a1              = 40
    x1 +  x2       - s2 + a2        = 30
    Minimize Z = 2x1 + 3x2 + 0.s1 + 0.s2 + M.a1 + M.a2
""")

    result = solve_diet_problem(verbose=True)

    print("\n" + "=" * 90)
    print("FINAL RESULT")
    print("=" * 90)
    print(f"Number of simplex iterations : {result['iterations']}")
    print("Optimal decision variable values:")
    for k, v in result["decision_variables"].items():
        print(f"    {k} = {v:.4f}")
    print(f"Optimal objective value Z* = {result['objective_value']:.4f}")

    print("\nVerification of constraints at the optimal point:")
    x1 = result["decision_variables"]["x1"]
    x2 = result["decision_variables"]["x2"]
    print(f"    x1 + 2x2 = {x1 + 2*x2:.4f}  (>= 40 required)")
    print(f"    x1 +  x2 = {x1 + x2:.4f}  (>= 30 required)")

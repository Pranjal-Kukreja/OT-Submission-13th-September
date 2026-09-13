"""
Transportation Problem  -  VAM (initial BFS)  +  MODI (optimality test
& iterative improvement)
=========================================================================

Case Study : A CLASSIC TRANSPORTATION PROBLEM
------------------------------------------------
A company has 3 factories (sources) supplying goods to 4 warehouses
(destinations). Each factory has a limited supply, each warehouse has
a fixed demand, and the unit shipping cost from every factory to
every warehouse is known. The problem is BALANCED
(total supply = total demand).

        Warehouse
Factory   D1   D2   D3   D4  | Supply
  S1      19   30   50   10  |   7
  S2      70   30   40   60  |   9
  S3      40    8   70   20  |  18
Demand     5    8    7   14  |  34   (= total supply)

Goal: find the shipment plan (how many units from each factory to
each warehouse) that MINIMIZES the total transportation cost.

Method (exactly as required by the assignment)
------------------------------------------------
STEP 1 - Vogel's Approximation Method (VAM)
    Produces a good initial basic feasible solution (BFS) by, at every
    step, allocating to the cheapest cell in the row/column that has
    the largest "penalty" (difference between the two smallest costs
    in that row/column).

STEP 2 - MODI (Modified Distribution) Method
    Starting from the VAM solution, MODI computes row/column dual
    values (u_i, v_j) from the basic cells (u_i + v_j = c_ij), derives
    the opportunity cost  d_ij = c_ij - (u_i + v_j)  for every
    non-basic cell, and:
        - if every d_ij >= 0  -> current solution is OPTIMAL, stop.
        - otherwise           -> bring the most negative d_ij cell
          into the basis, trace its unique closed loop through the
          current basic cells, reallocate along the loop
          (stepping-stone), and repeat.

The program below is written generically (works for any m x n
balanced transportation table) and applied to the case-study data.
Every VAM allocation step and every MODI iteration is printed.
"""

import itertools
import numpy as np
import pandas as pd

pd.set_option("display.width", 140)
TOL = 1e-7
EPS = 1e-6  # tiny allocation used only to resolve degenerate basic solutions


# =========================================================================
# Small utility: Disjoint-Set (Union-Find) used to detect / fix degeneracy
# =========================================================================
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb
            return True
        return False


# =========================================================================
# STEP 0 : balance the problem (add a dummy row/column if needed)
# =========================================================================
def balance_problem(supply, demand, cost):
    supply = list(supply)
    demand = list(demand)
    cost = [row[:] for row in cost]
    total_s, total_d = sum(supply), sum(demand)
    if abs(total_s - total_d) < TOL:
        return supply, demand, cost, None
    if total_s < total_d:
        supply.append(total_d - total_s)
        for row in cost:
            pass
        cost.append([0] * len(demand))
        return supply, demand, cost, "dummy_source_added"
    else:
        demand.append(total_s - total_d)
        for row in cost:
            row.append(0)
        return supply, demand, cost, "dummy_destination_added"


# =========================================================================
# STEP 1 : VOGEL'S APPROXIMATION METHOD  (initial basic feasible solution)
# =========================================================================
def vam_initial_solution(supply, demand, cost, verbose=True):
    m, n = len(supply), len(demand)
    supply = supply[:]
    demand = demand[:]
    allocation = {}                          # (i, j) -> quantity allocated
    active_rows = set(range(m))
    active_cols = set(range(n))
    step = 0

    if verbose:
        print("\n" + "=" * 90)
        print("STEP 1 : VOGEL'S APPROXIMATION METHOD (VAM) - Initial Basic Feasible Solution")
        print("=" * 90)

    while active_rows and active_cols:
        step += 1

        # ---- Row that has only one active column left -> forced allocation
        forced_cell = None
        for i in list(active_rows):
            cols = [j for j in active_cols]
            if len(cols) == 1:
                forced_cell = (i, cols[0])
                break
        if forced_cell is None:
            for j in list(active_cols):
                rows = [i for i in active_rows]
                if len(rows) == 1:
                    forced_cell = (rows[0], j)
                    break

        if forced_cell is not None:
            i, j = forced_cell
            qty = min(supply[i], demand[j])
        else:
            # ---- Compute row penalties
            row_pen = {}
            for i in active_rows:
                costs = sorted(cost[i][j] for j in active_cols)
                row_pen[i] = (costs[1] - costs[0]) if len(costs) > 1 else costs[0]

            # ---- Compute column penalties
            col_pen = {}
            for j in active_cols:
                costs = sorted(cost[i][j] for i in active_rows)
                col_pen[j] = (costs[1] - costs[0]) if len(costs) > 1 else costs[0]

            if verbose:
                pen_df = pd.DataFrame(
                    {"Row": [f"S{i+1}" for i in active_rows],
                     "Penalty": [row_pen[i] for i in active_rows]}
                )
                print(f"\n-- VAM Step {step}: Row penalties:")
                print(pen_df.to_string(index=False))
                pen_df2 = pd.DataFrame(
                    {"Column": [f"D{j+1}" for j in active_cols],
                     "Penalty": [col_pen[j] for j in active_cols]}
                )
                print("   Column penalties:")
                print(pen_df2.to_string(index=False))

            best_row = max(row_pen, key=lambda k: row_pen[k])
            best_col = max(col_pen, key=lambda k: col_pen[k])

            if row_pen[best_row] >= col_pen[best_col]:
                i = best_row
                j = min(active_cols, key=lambda c: (cost[i][c], c))
            else:
                j = best_col
                i = min(active_rows, key=lambda r: (cost[r][j], r))
            qty = min(supply[i], demand[j])

        allocation[(i, j)] = allocation.get((i, j), 0) + qty
        supply[i] -= qty
        demand[j] -= qty

        if verbose:
            print(f"   -> Allocate {qty} units to cell (S{i+1}, D{j+1}) "
                  f"[cost = {cost[i][j]}]  (remaining supply S{i+1}={supply[i]}, "
                  f"remaining demand D{j+1}={demand[j]})")

        if abs(supply[i]) < TOL:
            active_rows.discard(i)
        if abs(demand[j]) < TOL:
            active_cols.discard(j)

    if verbose:
        print("\nVAM initial allocation complete.")
    return allocation


# =========================================================================
# Fix degeneracy: ensure exactly (m + n - 1) basic cells before running MODI
# =========================================================================
def fix_degeneracy(allocation, cost, m, n, verbose=True):
    needed = m + n - 1
    dsu = DSU(m + n)
    for (i, j) in allocation:
        dsu.union(i, m + j)

    if len(allocation) == needed:
        return allocation

    if verbose:
        print(f"\n[Degeneracy check] Basic cells = {len(allocation)}, "
              f"required = {needed}. Adding epsilon allocations to "
              f"non-basic cells that keep the basis a spanning tree...")

    candidates = sorted(
        ((cost[i][j], i, j) for i in range(m) for j in range(n)
         if (i, j) not in allocation),
        key=lambda t: t[0]
    )
    for c, i, j in candidates:
        if len(allocation) == needed:
            break
        if dsu.find(i) != dsu.find(m + j):
            allocation[(i, j)] = EPS
            dsu.union(i, m + j)
            if verbose:
                print(f"   -> add epsilon allocation at (S{i+1}, D{j+1})")
    return allocation


# =========================================================================
# STEP 2 : MODI (MODIFIED DISTRIBUTION) METHOD
# =========================================================================
def compute_uv(allocation, cost, m, n):
    u = [None] * m
    v = [None] * n
    u[0] = 0
    changed = True
    while changed:
        changed = False
        for (i, j) in allocation:
            if u[i] is not None and v[j] is None:
                v[j] = cost[i][j] - u[i]
                changed = True
            elif v[j] is not None and u[i] is None:
                u[i] = cost[i][j] - v[j]
                changed = True
    # Any remaining None (can happen only if disconnected, shouldn't after
    # degeneracy fix) default to 0.
    u = [x if x is not None else 0 for x in u]
    v = [x if x is not None else 0 for x in v]
    return u, v


def find_closed_loop(allocation, entering_cell, m, n):
    """
    Find the unique closed loop formed when `entering_cell` is added to
    the current set of basic cells. Uses the standard elimination
    technique: repeatedly drop any cell (other than the entering cell)
    that is alone in its row or column among the current candidate set;
    what remains is exactly the loop. The loop is then ordered by
    alternating row/column moves starting at the entering cell.
    """
    candidates = set(allocation.keys()) | {entering_cell}

    changed = True
    while changed:
        changed = False
        row_count = {}
        col_count = {}
        for (i, j) in candidates:
            row_count[i] = row_count.get(i, 0) + 1
            col_count[j] = col_count.get(j, 0) + 1
        for cell in list(candidates):
            i, j = cell
            if cell == entering_cell:
                continue
            if row_count[i] == 1 or col_count[j] == 1:
                candidates.remove(cell)
                changed = True

    # order the loop by alternating row-move / column-move
    path = [entering_cell]
    visited = {entering_cell}
    current = entering_cell
    direction = "row"
    remaining = candidates - visited
    while remaining:
        i, j = current
        if direction == "row":
            nxt = [c for c in remaining if c[0] == i]
        else:
            nxt = [c for c in remaining if c[1] == j]
        if not nxt:
            break
        current = nxt[0]
        path.append(current)
        visited.add(current)
        remaining = candidates - visited
        direction = "col" if direction == "row" else "row"

    return path


def modi_solve(allocation, cost, m, n, verbose=True):
    allocation = dict(allocation)
    iteration = 0

    if verbose:
        print("\n" + "=" * 90)
        print("STEP 2 : MODI (MODIFIED DISTRIBUTION) METHOD - Optimality test & improvement")
        print("=" * 90)

    while True:
        iteration += 1
        u, v = compute_uv(allocation, cost, m, n)

        opp_cost = {}
        for i in range(m):
            for j in range(n):
                if (i, j) not in allocation:
                    opp_cost[(i, j)] = cost[i][j] - (u[i] + v[j])

        if verbose:
            print(f"\n--- MODI Iteration {iteration} ---")
            alloc_df = pd.DataFrame(
                [[allocation.get((i, j), "") for j in range(n)] for i in range(m)],
                index=[f"S{i+1}" for i in range(m)],
                columns=[f"D{j+1}" for j in range(n)],
            )
            print("Current allocation:")
            print(alloc_df.to_string())
            print("u_i :", {f"S{i+1}": round(u[i], 3) for i in range(m)})
            print("v_j :", {f"D{j+1}": round(v[j], 3) for j in range(n)})
            opp_df = pd.DataFrame(
                [[round(opp_cost.get((i, j), 0), 3) if (i, j) not in allocation else "basic"
                  for j in range(n)] for i in range(m)],
                index=[f"S{i+1}" for i in range(m)],
                columns=[f"D{j+1}" for j in range(n)],
            )
            print("Opportunity costs d_ij = c_ij - (u_i+v_j)  for non-basic cells:")
            print(opp_df.to_string())

        if not opp_cost or min(opp_cost.values()) >= -TOL:
            if verbose:
                print("\nAll opportunity costs >= 0  ->  current solution is OPTIMAL.")
            break

        entering = min(opp_cost, key=lambda k: opp_cost[k])
        if verbose:
            print(f"\nMost negative opportunity cost at {('S'+str(entering[0]+1), 'D'+str(entering[1]+1))} "
                  f"= {opp_cost[entering]:.3f}  ->  this cell ENTERS the basis.")

        loop = find_closed_loop(allocation, entering, m, n)
        signs = ["+" if k % 2 == 0 else "-" for k in range(len(loop))]

        if verbose:
            loop_str = " -> ".join(f"{s}(S{i+1},D{j+1})" for s, (i, j) in zip(signs, loop))
            print(f"Closed loop: {loop_str}")

        minus_cells = [loop[k] for k in range(len(loop)) if signs[k] == "-"]
        theta = min(allocation.get(c, 0) for c in minus_cells)

        if verbose:
            print(f"theta (max units shiftable) = min allocation among '-' cells = {theta:.3f}")

        for cell, sign in zip(loop, signs):
            if sign == "+":
                allocation[cell] = allocation.get(cell, 0) + theta
            else:
                allocation[cell] = allocation.get(cell, 0) - theta

        # remove the (a) leaving cell(s) that hit exactly zero
        leaving_candidates = [c for c in minus_cells if abs(allocation[c]) < 1e-9]
        # only drop ONE leaving cell (keep basis size = m+n-1) - standard rule
        if leaving_candidates:
            leave_cell = leaving_candidates[0]
            del allocation[leave_cell]
            if verbose:
                print(f"Leaving cell (drops to 0): (S{leave_cell[0]+1}, D{leave_cell[1]+1})")

        # drop any other cells whose value is (numerically) zero and not the entering cell
        for c in list(allocation.keys()):
            if c != entering and abs(allocation[c]) < 1e-9 and c not in [entering]:
                if c in allocation and c != leave_cell:
                    pass  # keep at least m+n-1 basic cells; do not over-remove

        if verbose:
            print(f"New allocation after reallocating {theta:.3f} units around the loop:")

    return allocation


# =========================================================================
# Reporting helpers
# =========================================================================
def total_cost(allocation, cost):
    return sum(qty * cost[i][j] for (i, j), qty in allocation.items() if qty > TOL)


def print_final_plan(allocation, cost, m, n, supply_names, demand_names, dummy_info):
    print("\n" + "=" * 90)
    print("FINAL RESULT - OPTIMAL SHIPMENT PLAN")
    print("=" * 90)
    plan_df = pd.DataFrame(
        [[allocation.get((i, j), 0) if allocation.get((i, j), 0) > TOL else 0
          for j in range(n)] for i in range(m)],
        index=supply_names, columns=demand_names,
    )
    print(plan_df.to_string())

    print("\nShipments (source -> destination : units, cost):")
    grand_total = 0.0
    for (i, j), qty in sorted(allocation.items()):
        if qty > TOL:
            is_dummy = (dummy_info == "dummy_source_added" and i == m - 1) or \
                       (dummy_info == "dummy_destination_added" and j == n - 1)
            line_cost = qty * cost[i][j]
            grand_total += line_cost
            tag = "  (dummy - unshipped / unmet, cost 0)" if is_dummy else ""
            print(f"   {supply_names[i]} -> {demand_names[j]} : {qty:.3f} units "
                  f"x {cost[i][j]} = {line_cost:.3f}{tag}")

    print(f"\nMINIMUM TOTAL TRANSPORTATION COST = {grand_total:.3f}")
    return grand_total


# =========================================================================
# CASE STUDY DATA
# =========================================================================
def solve_case_study(verbose=True):
    supply = [7, 9, 18]
    demand = [5, 8, 7, 14]
    cost = [
        [19, 30, 50, 10],
        [70, 30, 40, 60],
        [40, 8, 70, 20],
    ]
    supply_names = [f"S{i+1}" for i in range(len(supply))]
    demand_names = [f"D{j+1}" for j in range(len(demand))]

    supply_b, demand_b, cost_b, dummy_info = balance_problem(supply, demand, cost)
    if dummy_info and verbose:
        print(f"\n[Balancing] {dummy_info.replace('_', ' ')} (supply={sum(supply_b)}, "
              f"demand={sum(demand_b)}).")
        if dummy_info == "dummy_source_added":
            supply_names.append("Dummy")
        else:
            demand_names.append("Dummy")

    m, n = len(supply_b), len(demand_b)

    print("Cost matrix (rows = sources, columns = destinations):")
    print(pd.DataFrame(cost_b, index=supply_names, columns=demand_names).to_string())
    print("Supply:", dict(zip(supply_names, supply_b)))
    print("Demand:", dict(zip(demand_names, demand_b)))

    vam_alloc = vam_initial_solution(supply_b, demand_b, cost_b, verbose=verbose)

    print("\nVAM Initial Basic Feasible Solution:")
    vam_df = pd.DataFrame(
        [[vam_alloc.get((i, j), 0) for j in range(n)] for i in range(m)],
        index=supply_names, columns=demand_names,
    )
    print(vam_df.to_string())
    vam_cost = total_cost(vam_alloc, cost_b)
    print(f"Initial (VAM) total transportation cost = {vam_cost:.3f}")

    vam_alloc = fix_degeneracy(vam_alloc, cost_b, m, n, verbose=verbose)

    final_alloc = modi_solve(vam_alloc, cost_b, m, n, verbose=verbose)

    grand_total = print_final_plan(final_alloc, cost_b, m, n, supply_names, demand_names, dummy_info)
    return final_alloc, grand_total


if __name__ == "__main__":
    print("#" * 90)
    print("TRANSPORTATION PROBLEM  -  VAM (initial BFS) + MODI (optimality & improvement)")
    print("#" * 90)
    print("""
Problem formulation
--------------------
3 factories (S1,S2,S3) ship a single product to 4 warehouses
(D1,D2,D3,D4). Unit shipping costs, factory supplies and warehouse
demands are given below. Find the shipment plan that minimizes total
transportation cost (total supply = total demand = 34, balanced).
""")
    solve_case_study(verbose=True)

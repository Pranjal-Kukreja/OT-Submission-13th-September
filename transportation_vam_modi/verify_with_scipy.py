"""
Independent cross-check of the VAM + MODI result using
scipy.optimize.linprog (HiGHS solver), formulating the balanced
transportation problem directly as an equality-constrained LP:

    minimize   sum_i sum_j c_ij * x_ij
    subject to sum_j x_ij = supply_i      for every source i
               sum_i x_ij = demand_j      for every destination j
               x_ij >= 0

This is NOT the assignment solution itself -- it only exists to
verify that the hand-built VAM + MODI implementation in
transportation_vam_modi.py produces the correct minimum cost.
"""

import numpy as np
from scipy.optimize import linprog

supply = [7, 9, 18]
demand = [5, 8, 7, 14]
cost = [
    [19, 30, 50, 10],
    [70, 30, 40, 60],
    [40, 8, 70, 20],
]

m, n = len(supply), len(demand)
c = np.array(cost).flatten()  # x_ij in row-major order: x[i*n + j]

A_eq = []
b_eq = []

# supply constraints: sum_j x_ij = supply_i
for i in range(m):
    row = np.zeros(m * n)
    row[i * n:(i + 1) * n] = 1
    A_eq.append(row)
    b_eq.append(supply[i])

# demand constraints: sum_i x_ij = demand_j
for j in range(n):
    row = np.zeros(m * n)
    for i in range(m):
        row[i * n + j] = 1
    A_eq.append(row)
    b_eq.append(demand[j])

result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")

x = result.x.reshape(m, n)
print("SciPy linprog verification")
print("-" * 50)
print(f"Status : {result.message}")
print("Optimal shipment plan (rows=S1..S3, cols=D1..D4):")
print(np.round(x, 3))
print(f"\nMinimum total cost (SciPy) = {result.fun:.3f}")
print("Expected from VAM + MODI implementation: 743.000")

"""
Independent cross-check of the Big-M Simplex result using
scipy.optimize.linprog (interior-point / HiGHS solver).

This is NOT the assignment solution itself -- it only exists to
verify that the hand-built Big-M simplex implementation in
big_m_simplex.py produces the correct optimum.
"""

from scipy.optimize import linprog

# Minimize Z = 2x1 + 3x2
c = [2, 3]

# scipy's linprog expects constraints in the form A_ub @ x <= b_ub,
# so we convert our ">=" constraints by multiplying both sides by -1.
#   x1 + 2x2 >= 40   ->   -x1 - 2x2 <= -40
#   x1 +  x2 >= 30   ->   -x1 -  x2 <= -30
A_ub = [[-1, -2],
        [-1, -1]]
b_ub = [-40, -30]

bounds = [(0, None), (0, None)]

result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

print("SciPy linprog verification")
print("-" * 40)
print(f"Status  : {result.message}")
print(f"x1      : {result.x[0]:.4f}")
print(f"x2      : {result.x[1]:.4f}")
print(f"Z*      : {result.fun:.4f}")
print("\nExpected from Big-M simplex: x1 = 20, x2 = 10, Z* = 70")

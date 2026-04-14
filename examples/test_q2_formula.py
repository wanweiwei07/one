import numpy as np
import one.utils.math as oum

# Check: what does the current q2 formula do?
# Line 226: q2 = float(psi - phi + np.pi/2)

# From debug output for q1=45°, q2=30°, q3=0°:
# phi_wrist = 112.21°
# psi = 21.18°
# q2_computed = 91.03°
# q2_expected = 30.00°

phi = np.radians(112.21)
psi = np.radians(21.18)

q2_old_formula = phi - psi  # Original formula
q2_new_formula = psi - phi + np.pi/2  # Current formula (line 226)

print("Testing q2 formula with debug output values:")
print(f"phi (angle to wrist in swing plane): {np.degrees(phi):.2f}°")
print(f"psi (elbow contribution): {np.degrees(psi):.2f}°")
print()
print(f"q2 (old: phi - psi): {np.degrees(q2_old_formula):.2f}°")
print(f"q2 (current: psi - phi + 90°): {np.degrees(q2_new_formula):.2f}°")
print(f"q2 expected: 30.00°")
print()

# Try various formula combinations
formulas = [
    ("phi - psi", phi - psi),
    ("psi - phi", psi - phi),
    ("phi - psi + 90°", phi - psi + np.pi/2),
    ("psi - phi + 90°", psi - phi + np.pi/2),
    ("phi + psi", phi + psi),
    ("-phi - psi + 90°", -phi - psi + np.pi/2),
    ("180° - phi - psi", np.pi - phi - psi),
]

print("="*80)
print("Trying different q2 formulas:")
for name, value in formulas:
    deg = np.degrees(value)
    wrapped = np.degrees(oum.wrap_to_pi(value))
    error = abs(wrapped - 30.0)
    print(f"{name:25s}: {deg:7.2f}° (wrapped: {wrapped:7.2f}°) error: {error:.2f}°")

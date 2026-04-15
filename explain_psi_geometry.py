import numpy as np
import matplotlib.pyplot as plt

# 2R planar mechanism geometry
# Imagine we're looking at the arm from the side (in the swing plane)
#
#     J2 (shoulder)
#      o----------o J3 (elbow)
#      |    l2    |\
#      |          | \ l3
#      |          |  \
#      |          |   o Wrist
#      |          | q3/
#      |          |  /
#      |__________|_/
#         d_swing

l2 = 0.455  # Link 2 length
l3 = 0.475  # Link 3 length
q3 = np.radians(41.42)  # From debug output

c3 = np.cos(q3)
s3 = np.sin(q3)

print("2R Planar Mechanism Geometry Analysis")
print("="*80)
print(f"l2 (shoulder to elbow): {l2:.3f} m")
print(f"l3 (elbow to wrist):    {l3:.3f} m")
print(f"q3 (elbow angle):       {np.degrees(q3):.2f}°")
print(f"cos(q3):                {c3:.4f}")
print(f"sin(q3):                {s3:.4f}")
print()

# When we have a 2R arm with joint angles q2 and q3:
# - J2 is at origin [0, 0]
# - J3 (elbow) is at [l2, 0] after rotating by q2
# - Wrist is at J3 + [l3*cos(q3), l3*sin(q3)] in the link's local frame

# But we need to relate q2 to the angle φ (phi) that points to the wrist
# and the elbow angle contribution ψ (psi)

# The triangle formed is:
#   - One side: l2 (shoulder to elbow)
#   - Second side: l3 (elbow to wrist) 
#   - Third side: d_swing (shoulder to wrist, straight line)
#   - Angle at elbow: π - q3 (interior angle)

# When we place J2 at origin and want to find the angle q2:
# We decompose the wrist position as:
#   wrist = [l2 + l3*cos(q3), l3*sin(q3)]  in link-2's frame
#
# The angle from J2 to wrist (in the frame where link-2 points along +X) is:
#   atan2(l3*sin(q3), l2 + l3*cos(q3))
#
# This is ψ (psi) - the "elbow contribution" to the pointing angle

psi = np.arctan2(l3 * s3, l2 + l3 * c3)

print("ψ (psi) - Elbow Angle Contribution")
print("-"*80)
print("Formula: ψ = atan2(l3*sin(q3), l2 + l3*cos(q3))")
print()
print(f"l3 * sin(q3) = {l3 * s3:.4f} m  ← This is the VERTICAL component")
print(f"l2 + l3*cos(q3) = {l2 + l3 * c3:.4f} m  ← This is the HORIZONTAL component")
print()
print(f"ψ = atan2({l3*s3:.4f}, {l2 + l3*c3:.4f}) = {np.degrees(psi):.2f}°")
print()

print("Geometric Interpretation:")
print("-"*80)
print("Imagine the 2R arm in a 2D plane:")
print()
print("     J2")
print("      •───────────→ l2 ──────────→ • J3")
print("       \                            |")
print("        \                           | q3 angle")
print("         \                          ↓")
print("          \                    l3 * sin(q3)")
print("           \                        |")
print("            \_____ d_swing _______• Wrist")
print("             \← ψ (psi)")
print()
print("l3 * sin(q3): The perpendicular offset from link-2's extension line")
print("              to the wrist. This is how much the elbow 'bends' the")
print("              arm away from a straight configuration.")
print()
print("l2 + l3*cos(q3): The distance along link-2's direction from J2 to the")
print("                 wrist's projection onto the extended link-2 axis.")
print()
print("ψ (psi): The angle that the straight line from J2 to wrist makes")
print("         with link-2's initial direction. This is the 'elbow")
print("         contribution' - how much the elbow angle affects the")
print("         overall pointing direction.")
print()

print("="*80)
print("Relationship to q2:")
print("-"*80)
print("φ (phi): Angle from reference direction to wrist in swing plane")
print("ψ (psi): Angle from link-2 direction to wrist (elbow contribution)")
print("q2:      Actual joint angle of J2")
print()
print("The relationship depends on the reference frame and joint convention.")
print("Common formulas:")
print("  q2 = φ - ψ       (if q2=0 means link-2 along reference)")
print("  q2 = φ + ψ       (alternative convention)")
print("  q2 = ψ - φ       (yet another convention)")
print()
print("The correct formula depends on:")
print("  1. What does q2=0 mean? (where does link-2 point at zero?)")
print("  2. What is the reference direction for φ?")
print("  3. Which direction is positive rotation?")

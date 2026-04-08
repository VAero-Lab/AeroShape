"""Check if the interpolated B-spline has a loop/overshoot at the TE."""
from aeroshape.geometry.airfoils import AirfoilProfile
from OCP.BRepAdaptor import BRepAdaptor_CompCurve
import numpy as np

prof = AirfoilProfile.from_naca4("2412", num_points=50)

# Build a wire in the default orientation (horizontal wing)
wire = prof.to_occ_wire(
    position=(0, 0, 0),
    twist_deg=0.0,
    local_chord=5.0,  # 5m chord
)

# Sample the wire densely
curve = BRepAdaptor_CompCurve(wire)
n_pts = 500
u_first = curve.FirstParameter()
u_last = curve.LastParameter()

pts = []
for i in range(n_pts + 1):
    u = u_first + (u_last - u_first) * i / n_pts
    p = curve.Value(u)
    pts.append((p.X(), p.Y(), p.Z()))

pts = np.array(pts)

# Find the TE region: points near max X
max_x = np.max(pts[:, 0])
te_mask = pts[:, 0] > max_x - 0.1  # within 0.1m of TE
te_pts = pts[te_mask]

print(f"Max X (TE): {max_x:.6f}")
print(f"Points near TE ({len(te_pts)}):")
for p in te_pts:
    print(f"  x={p[0]:.6f}, y={p[1]:.6f}, z={p[2]:.6f}")

# Check the Z spread at the TE
if len(te_pts) > 0:
    z_spread = np.max(te_pts[:, 2]) - np.min(te_pts[:, 2])
    print(f"\nZ spread at TE: {z_spread:.6f}")

# Check points near the start and end of the curve
print(f"\nFirst 5 points:")
for p in pts[:5]:
    print(f"  x={p[0]:.6f}, y={p[1]:.6f}, z={p[2]:.6f}")
print(f"\nLast 5 points:")
for p in pts[-5:]:
    print(f"  x={p[0]:.6f}, y={p[1]:.6f}, z={p[2]:.6f}")

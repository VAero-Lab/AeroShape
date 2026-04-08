"""Compare TE cusp behavior: horizontal wing vs winglet (with v_thickness_dir).

The hypothesis is that when v_thickness_dir rotates the airfoil into a
non-standard orientation, the B-spline interpolation may produce different 
cusp behavior near the TE closure point.
"""
from aeroshape.geometry.airfoils import AirfoilProfile
from OCP.BRepAdaptor import BRepAdaptor_CompCurve
import numpy as np

prof = AirfoilProfile.from_naca4("0010", num_points=50)

# Case 1: Horizontal (standard main wing)
wire_horiz = prof.to_occ_wire(
    position=(0, 5, 0),
    twist_deg=0.0,
    local_chord=3.0,
    v_chord_dir=(1, 0, 0),
    v_thickness_dir=(0, 0, 1),  # Standard Z-up
)

# Case 2: Vertical winglet (thickness in Y direction)
wire_vert = prof.to_occ_wire(
    position=(0, 5, 5),
    twist_deg=0.0,
    local_chord=3.0,
    v_chord_dir=(1, 0, 0),
    v_thickness_dir=(0, -1, 0),  # Thickness in -Y for a vertical winglet
)

def sample_te_region(wire, label):
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
    
    max_x = np.max(pts[:, 0])
    te_mask = pts[:, 0] > max_x - 0.05
    te_pts = pts[te_mask]
    
    print(f"\n=== {label} ===")
    print(f"Max X: {max_x:.6f}")
    print(f"Num TE points: {len(te_pts)}")
    if len(te_pts) > 0:
        y_spread = np.max(te_pts[:, 1]) - np.min(te_pts[:, 1])
        z_spread = np.max(te_pts[:, 2]) - np.min(te_pts[:, 2])
        print(f"Y spread at TE: {y_spread:.6f}")
        print(f"Z spread at TE: {z_spread:.6f}")
        # Show first, middle, last
        print(f"  Start: ({te_pts[0, 0]:.5f}, {te_pts[0, 1]:.5f}, {te_pts[0, 2]:.5f})")
        mid = len(te_pts) // 2
        print(f"  Mid:   ({te_pts[mid, 0]:.5f}, {te_pts[mid, 1]:.5f}, {te_pts[mid, 2]:.5f})")
        print(f"  End:   ({te_pts[-1, 0]:.5f}, {te_pts[-1, 1]:.5f}, {te_pts[-1, 2]:.5f})")

sample_te_region(wire_horiz, "Horizontal Wing")
sample_te_region(wire_vert, "Vertical Winglet")

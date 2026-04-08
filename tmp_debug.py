import sys
import numpy as np
from aeroshape.geometry.wings import MultiSegmentWing, AirfoilProfile
from aeroshape.geometry.curves import GuideCurve

print("Testing winglet generation...")

le0 = (0, 10, 0)
te0 = (5, 10, 0)

top_le = (0, 10, 10)
top_te = (3, 10, 10)

le = GuideCurve(start_point=le0)
le.add_line(end_point=top_le)

te = GuideCurve(start_point=te0)
te.add_line(end_point=top_te)

prof = AirfoilProfile.from_naca4("0020", num_points=50)

winglet = MultiSegmentWing.from_planform_curves(le, te, [(0, prof), (1, prof)], num_sections=5)
frames = winglet.get_section_frames()
mid_frame = frames[2]

print("Mid frame Z:", mid_frame["z_offset"])
print("Chord:", mid_frame["chord"])
print("v_chord_dir:", mid_frame["v_chord_dir"])
print("v_thickness_dir:", mid_frame["v_thickness_dir"])

wire = mid_frame["airfoil"].to_occ_wire(
    position=(mid_frame["x_offset"], mid_frame["y"], mid_frame["z_offset"]),
    twist_deg=mid_frame["twist_deg"],
    local_chord=mid_frame["chord"],
    v_chord_dir=mid_frame.get("v_chord_dir"),
    v_thickness_dir=mid_frame.get("v_thickness_dir")
)

from OCP.BRepAdaptor import BRepAdaptor_CompCurve
curve = BRepAdaptor_CompCurve(wire)
n_pts = 100
points = []
for i in range(n_pts + 1):
    u = curve.FirstParameter() + i / n_pts * (curve.LastParameter() - curve.FirstParameter())
    pnt = curve.Value(u)
    points.append((pnt.X(), pnt.Y(), pnt.Z()))
    
print("Wire points (num={}):".format(len(points)))
for i, p in enumerate(points):
    if i % 10 == 0:
        print(f"  Pt {i}: {p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}")
        
Y_vals = [p[1] for p in points]
print(f"Y min: {min(Y_vals):.4f}, Y max: {max(Y_vals):.4f}")

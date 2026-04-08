"""Write a tiny STEP file with just 3 lofted sections to visualize the TE seam.

We'll make two versions:
1. A horizontal loft (pure main-wing orientation)
2. A vertical loft (winglet orientation)
"""
from aeroshape.geometry.airfoils import AirfoilProfile
from aeroshape.nurbs.surfaces import NurbsSurfaceBuilder
from aeroshape.nurbs.export import NurbsExporter
import os

prof = AirfoilProfile.from_naca4("0012", num_points=50)

# Case 1: Horizontal wing
wires_h = []
for y in [0, 2, 4]:
    w = prof.to_occ_wire(
        position=(0, y, 0),
        twist_deg=0.0,
        local_chord=3.0,
        v_chord_dir=(1, 0, 0),
        v_thickness_dir=(0, 0, 1),
    )
    wires_h.append(w)

solid_h = NurbsSurfaceBuilder.loft(wires_h, solid=True, ruled=False)

# Case 2: Vertical winglet
wires_v = []
for z in [0, 2, 4]:
    w = prof.to_occ_wire(
        position=(0, 5, z),
        twist_deg=0.0,
        local_chord=3.0,
        v_chord_dir=(1, 0, 0),
        v_thickness_dir=(0, -1, 0),
    )
    wires_v.append(w)

solid_v = NurbsSurfaceBuilder.loft(wires_v, solid=True, ruled=False)

os.makedirs("Exports", exist_ok=True)
NurbsExporter.to_step(solid_h, "Exports/test_te_horizontal.step")
NurbsExporter.to_step(solid_v, "Exports/test_te_vertical.step")
print("Exported test_te_horizontal.step and test_te_vertical.step")
print("Open both in your CAD viewer and compare the trailing edges!")

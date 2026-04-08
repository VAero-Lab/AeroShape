"""Test BRepOffsetAPI_ThruSections with various settings to fix TE ribbon.

The theory: by refining how the lofter interpolates between wire sections 
at their seam points, we can eliminate the tessellation artifact.
"""
from aeroshape.geometry.airfoils import AirfoilProfile
from aeroshape.nurbs.export import NurbsExporter
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from build123d import Wire, Solid, Shape
import os

prof = AirfoilProfile.from_naca4("0012", num_points=50)

# Build 5 section wires for a vertical winglet
wires = []
for z in range(5):
    w = prof.to_occ_wire(
        position=(0, 5, z * 2),
        twist_deg=0.0,
        local_chord=3.0 - z * 0.3,  # Tapering
        v_chord_dir=(1, 0, 0),
        v_thickness_dir=(0, -1, 0),
    )
    wires.append(w)

os.makedirs("Exports", exist_ok=True)

# Method 1: Default (current behavior)
loft1 = BRepOffsetAPI_ThruSections(True, False)  # solid=True, ruled=False
for w in wires:
    loft1.AddWire(w)
loft1.Build()
NurbsExporter.to_step(Shape(loft1.Shape()), "Exports/test_te_default.step")
print("Exported test_te_default.step")

# Method 2: With CheckCompatibility disabled
loft2 = BRepOffsetAPI_ThruSections(True, False)
loft2.CheckCompatibility(False)  # Don't reorder wire vertices
for w in wires:
    loft2.AddWire(w)
loft2.Build()
NurbsExporter.to_step(Shape(loft2.Shape()), "Exports/test_te_no_compat.step")
print("Exported test_te_no_compat.step")

# Method 3: With SetSmoothing and higher parameters
loft3 = BRepOffsetAPI_ThruSections(True, False)
loft3.SetSmoothing(True)
loft3.SetMaxDegree(8)
for w in wires:
    loft3.AddWire(w)
loft3.Build()
NurbsExporter.to_step(Shape(loft3.Shape()), "Exports/test_te_smooth.step")
print("Exported test_te_smooth.step")

# Method 4: With SetCriteriumWeight
loft4 = BRepOffsetAPI_ThruSections(True, False)
loft4.SetSmoothing(True)
loft4.SetMaxDegree(8)
loft4.SetCriteriumWeight(0.1, 0.3, 0.6)  # Reduce torsion weight
for w in wires:
    loft4.AddWire(w)
loft4.Build()
NurbsExporter.to_step(Shape(loft4.Shape()), "Exports/test_te_weighted.step")
print("Exported test_te_weighted.step")

# Method 5: Ruled (linear interpolation between sections)
loft5 = BRepOffsetAPI_ThruSections(True, True)  # ruled=True
for w in wires:
    loft5.AddWire(w)
loft5.Build()
NurbsExporter.to_step(Shape(loft5.Shape()), "Exports/test_te_ruled.step")
print("Exported test_te_ruled.step")

print("\nAll test exports complete.")
print("Compare trailing edges in your CAD viewer!")

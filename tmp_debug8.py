"""Diagnose the trailing edge gap in the STEP lofted surface.

For a closed airfoil wire made from a single B-spline, the lofter creates
a single face. The TE corresponds to the parametric seam u=0 and u=1
on that face. If those two edges don't coincide, we get the fat ribbon.

Let's check by sampling the lofted surface at u=0 and u=1 (the TE seam).
"""
from examples.aircraft_commercial_airliner import create_wings
from aeroshape.nurbs.surfaces import NurbsSurfaceBuilder
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
import numpy as np

wings = create_wings()
wing = wings[0][0]
frames = wing.get_section_frames()

# Build just the winglet segment (last segment)
# The winglet is segment index 2
seg_frames = []
start = 0
for seg in wing.segments:
    n = seg.num_sections if hasattr(seg, 'num_sections') else 10
    start += n

# Just use the last few frames for the winglet
wires = []
for fr in frames[-10:]:  # Last 10 frames span the winglet
    wire = fr["airfoil"].to_occ_wire(
        position=(fr["x_offset"], fr["y"], fr["z_offset"]),
        twist_deg=fr["twist_deg"],
        local_chord=fr["chord"],
        v_chord_dir=fr.get("v_chord_dir"),
        v_thickness_dir=fr.get("v_thickness_dir")
    )
    wires.append(wire)

# Loft as shell for inspection
shell = NurbsSurfaceBuilder.loft(wires, solid=False, ruled=False)
shape = shell.wrapped if hasattr(shell, 'wrapped') else shell

# Get the face
explorer = TopExp_Explorer(shape, TopAbs_FACE)
face = TopoDS.Face_s(explorer.Current())
adaptor = BRepAdaptor_Surface(face)

u_min, u_max = adaptor.FirstUParameter(), adaptor.LastUParameter()
v_min, v_max = adaptor.FirstVParameter(), adaptor.LastVParameter()

print(f"Face U range: [{u_min:.6f}, {u_max:.6f}]")
print(f"Face V range: [{v_min:.6f}, {v_max:.6f}]")

# Sample at the seam (u=0 and u=1 or u=u_min and u=u_max)
# which should correspond to the TE
print(f"\n--- TE Seam check: u at start vs u at end ---")
for v_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
    v = v_min + v_frac * (v_max - v_min)
    pt_start = adaptor.Value(u_min, v)
    pt_end = adaptor.Value(u_max, v)
    gap = ((pt_start.X()-pt_end.X())**2 + (pt_start.Y()-pt_end.Y())**2 + (pt_start.Z()-pt_end.Z())**2)**0.5
    print(f"  V={v_frac:.2f}: start=({pt_start.X():.4f}, {pt_start.Y():.4f}, {pt_start.Z():.4f})")
    print(f"          end  =({pt_end.X():.4f}, {pt_end.Y():.4f}, {pt_end.Z():.4f})")
    print(f"          gap  = {gap:.6f}")

# Also check the airfoil closure
print(f"\n--- Airfoil point closure check ---")
fr = frames[-5]
px, pz = fr["airfoil"].x.copy(), fr["airfoil"].z.copy()
if fr["chord"] != fr["airfoil"].chord:
    scale = fr["chord"] / fr["airfoil"].chord
    px = px * scale
    pz = pz * scale
print(f"  First point: ({px[0]:.6f}, {pz[0]:.6f})")
print(f"  Last point:  ({px[-1]:.6f}, {pz[-1]:.6f})")
print(f"  TE gap in profile coords: {((px[0]-px[-1])**2 + (pz[0]-pz[-1])**2)**0.5:.8f}")

"""Compare shell vs solid face topology for the winglet loft."""
from examples.aircraft_commercial_airliner import create_wings
from aeroshape.nurbs.surfaces import NurbsSurfaceBuilder
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
import numpy as np

wings = create_wings()
wing = wings[0][0]
frames = wing.get_section_frames()

# Build wires for the winglet (last 10 frames)
wires = []
for fr in frames[-10:]:
    wire = fr["airfoil"].to_occ_wire(
        position=(fr["x_offset"], fr["y"], fr["z_offset"]),
        twist_deg=fr["twist_deg"],
        local_chord=fr["chord"],
        v_chord_dir=fr.get("v_chord_dir"),
        v_thickness_dir=fr.get("v_thickness_dir")
    )
    wires.append(wire)

print("=== SHELL (solid=False) ===")
shell = NurbsSurfaceBuilder.loft(wires, solid=False, ruled=False)
shape = shell.wrapped if hasattr(shell, 'wrapped') else shell
explorer = TopExp_Explorer(shape, TopAbs_FACE)
idx = 0
while explorer.More():
    face = TopoDS.Face_s(explorer.Current())
    gp = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, gp)
    adaptor = BRepAdaptor_Surface(face)
    u0, u1 = adaptor.FirstUParameter(), adaptor.LastUParameter()
    v0, v1 = adaptor.FirstVParameter(), adaptor.LastVParameter()
    print(f"  Face {idx}: U=[{u0:.3f},{u1:.3f}], V=[{v0:.3f},{v1:.3f}], area={gp.Mass():.4f}")
    idx += 1
    explorer.Next()
print(f"  Total faces: {idx}")

print("\n=== SOLID (solid=True) ===")
solid = NurbsSurfaceBuilder.loft(wires, solid=True, ruled=False)
shape_s = solid.wrapped if hasattr(solid, 'wrapped') else solid
explorer2 = TopExp_Explorer(shape_s, TopAbs_FACE)
idx2 = 0
while explorer2.More():
    face = TopoDS.Face_s(explorer2.Current())
    gp = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, gp)
    adaptor = BRepAdaptor_Surface(face)
    u0, u1 = adaptor.FirstUParameter(), adaptor.LastUParameter()
    v0, v1 = adaptor.FirstVParameter(), adaptor.LastVParameter()
    
    # Sample corners to understand shape
    p00 = adaptor.Value(u0, v0)
    p10 = adaptor.Value(u1, v0)
    p01 = adaptor.Value(u0, v1)
    p11 = adaptor.Value(u1, v1)
    print(f"  Face {idx2}: U=[{u0:.3f},{u1:.3f}], V=[{v0:.3f},{v1:.3f}], area={gp.Mass():.4f}")
    print(f"    corner(0,0)=({p00.X():.2f},{p00.Y():.2f},{p00.Z():.2f})")
    print(f"    corner(1,0)=({p10.X():.2f},{p10.Y():.2f},{p10.Z():.2f})")
    print(f"    corner(0,1)=({p01.X():.2f},{p01.Y():.2f},{p01.Z():.2f})")
    print(f"    corner(1,1)=({p11.X():.2f},{p11.Y():.2f},{p11.Z():.2f})")
    idx2 += 1
    explorer2.Next()
print(f"  Total faces: {idx2}")

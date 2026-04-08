"""Diagnose face structure of lofted shell for single vs multi-edge wires."""
from examples.aircraft_commercial_airliner import create_wings
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRep import BRep_Tool
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from aeroshape.nurbs.surfaces import NurbsSurfaceBuilder

wings = create_wings()
wing = wings[0][0]  # Main wing
frames = wing.get_section_frames()

# Build wires for the first 3 frames only (small test)
wires = []
for fr in frames[:5]:
    wire = fr["airfoil"].to_occ_wire(
        position=(fr["x_offset"], fr["y"], fr["z_offset"]),
        twist_deg=fr["twist_deg"],
        local_chord=fr["chord"],
        v_chord_dir=fr.get("v_chord_dir"),
        v_thickness_dir=fr.get("v_thickness_dir")
    )
    wires.append(wire)

# Count edges per wire
for i, w in enumerate(wires):
    edge_exp = TopExp_Explorer(w if not hasattr(w, 'wrapped') else w.wrapped, TopAbs_EDGE)
    count = 0
    while edge_exp.More():
        count += 1
        edge_exp.Next()
    print(f"Wire {i}: {count} edges")

# Loft as shell
shell = NurbsSurfaceBuilder.loft(wires, solid=False, ruled=False)
shape = shell.wrapped if hasattr(shell, 'wrapped') else shell

# Count faces
face_exp = TopExp_Explorer(shape, TopAbs_FACE)
face_count = 0
while face_exp.More():
    face = TopoDS.Face_s(face_exp.Current())
    adaptor = BRepAdaptor_Surface(face)
    u_min, u_max = adaptor.FirstUParameter(), adaptor.LastUParameter()
    v_min, v_max = adaptor.FirstVParameter(), adaptor.LastVParameter()
    
    gprops = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, gprops)
    area = gprops.Mass()
    
    print(f"Face {face_count}: U=[{u_min:.4f}, {u_max:.4f}], V=[{v_min:.4f}, {v_max:.4f}], Area={area:.4f}")
    face_count += 1
    face_exp.Next()

print(f"\nTotal faces: {face_count}")

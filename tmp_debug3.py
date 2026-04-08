from examples.aircraft_box_wing import create_box_wings
from examples.aircraft_box_wing import create_box_wings
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
import numpy as np

print("Generating Box Fin...")
wings = create_box_wings()
fin = wings[2][0]

segs = fin.to_occ_segments()
if not segs:
    print("No segments!")
    exit(1)
    
shape = segs[0]

# Tessellate 
BRepMesh_IncrementalMesh(shape.wrapped if hasattr(shape, 'wrapped') else shape, 0.1)

explorer = TopExp_Explorer(shape.wrapped if hasattr(shape, 'wrapped') else shape, TopAbs_FACE)
faces_data = []

while explorer.More():
    face = TopoDS.Face_s(explorer.Current())
    loc = TopLoc_Location()
    tri_data = BRep_Tool.Triangulation_s(face, loc)
    if tri_data is not None:
        verts = []
        for i in range(1, tri_data.NbNodes() + 1):
            p = tri_data.Node(i)
            p.Transform(loc.Transformation())
            verts.append([p.X(), p.Y(), p.Z()])
            
        faces = []
        for i in range(1, tri_data.NbTriangles() + 1):
            tri = tri_data.Triangle(i)
            faces.extend([3, tri.Value(1) - 1, tri.Value(2) - 1, tri.Value(3) - 1])
            
        faces_data.append((np.array(verts), np.array(faces)))
    explorer.Next()

print(f"Extracted {len(faces_data)} faces.")

all_v = []
all_f = []
offset = 0
for v, f in faces_data:
    all_v.append(v)
    f_mod = f.copy()
    for i in range(0, len(f), 4):
        f_mod[i+1] += offset
        f_mod[i+2] += offset
        f_mod[i+3] += offset
    all_f.append(f_mod)
    offset += len(v)
    
if all_v:
    V = np.vstack(all_v)
    F = np.concatenate(all_f)
    print(f"Total vertices: {len(V)}")
    print(f"Y min: {np.min(V[:, 1]):.4f}")
    print(f"Y max: {np.max(V[:, 1]):.4f}")
    print(f"Y spread (thickness): {np.max(V[:, 1]) - np.min(V[:, 1]):.4f}")
else:
    print("Failed to get mesh.")

from examples.aircraft_commercial_airliner import create_wings
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
import numpy as np

print("Generating winglet...")
wings_list = create_wings()
wing = wings_list[0][0] # First wing tuple
winglet = wing.segments[-1] 

# The MultiSegmentWing is returned. Let's just grab the whole shape.
shapes = wing.to_occ_segments()
# The last shape should be the winglet!
shape = shapes[-1]
if hasattr(shape, 'wrapped'):
    shape = shape.wrapped

# Analyze vertices near the trailing edge!
explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
verts = []
while explorer.More():
    v = TopoDS.Vertex_s(explorer.Current())
    p = BRep_Tool.Pnt_s(v)
    verts.append((p.X(), p.Y(), p.Z()))
    explorer.Next()
    
verts = np.array(verts)
print(f"Max X (TE region): {np.max(verts[:, 0])}")
te_verts = verts[verts[:, 0] > np.max(verts[:, 0]) - 0.5]
print("TE Vertices:")
for pt in te_verts:
    print(f"  {pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}")

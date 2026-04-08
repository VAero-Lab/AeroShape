from examples.aircraft_commercial_airliner import create_wings
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_WIRE, TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRep import BRep_Tool

wings = create_wings()
wing = wings[0][0]
shapes = wing.to_occ_segments()
winglet_shape = shapes[-1].wrapped if hasattr(shapes[-1], "wrapped") else shapes[-1]

print("Scanning edges of winglet...")
edge_explorer = TopExp_Explorer(winglet_shape, TopAbs_EDGE)
edge_lengths = []
while edge_explorer.More():
    edge = TopoDS.Edge_s(edge_explorer.Current())
    curve_adaptor = BRepAdaptor_Curve(edge)
    f = curve_adaptor.FirstParameter()
    l = curve_adaptor.LastParameter()
    
    # Just sample a few points to get an approx length
    pts = []
    for i in range(10):
        t = f + (l - f) * i / 9.0
        pt = curve_adaptor.Value(t)
        pts.append((pt.X(), pt.Y(), pt.Z()))
    
    length = 0
    for i in range(1, 10):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        dz = pts[i][2] - pts[i-1][2]
        length += (dx**2 + dy**2 + dz**2)**0.5
        
    edge_lengths.append(length)
    edge_explorer.Next()

edge_lengths.sort()
print("Top 10 smallest edge lengths:")
for l in edge_lengths[:10]:
    print(f"  {l:.6f}")

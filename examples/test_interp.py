"""Test lofting with GeomAPI_Interpolate to ensure no pole explosion."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from OCP.gp import gp_Pnt
from OCP.GeomAPI import GeomAPI_Interpolate
from OCP.TColgp import TColgp_HArray1OfPnt
from OCP.TColStd import TColStd_HArray1OfReal
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_BSplineSurface
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs

from aeroshape.geometry.wings import AirfoilProfile
from aeroshape.nurbs.surfaces import NurbsSurfaceBuilder
from build123d import Wire

airfoils = ["0025", "4418", "2412", "2410", "0009"] * 5 # 25 sections

n = 79 # 40*2-1
params = TColStd_HArray1OfReal(1, n)
for i in range(n):
    params.SetValue(i + 1, i / (n - 1))

wires = []
for idx, naca in enumerate(airfoils):
    af = AirfoilProfile.from_naca4(naca, num_points=40)
    px, pz = af.x.copy(), af.z.copy()
    px[-1], pz[-1] = px[0], pz[0]
    
    arr = TColgp_HArray1OfPnt(1, n)
    for i in range(n):
        arr.SetValue(i + 1, gp_Pnt(float(px[i]), float(idx * 2.0), float(pz[i])))
        
    interp = GeomAPI_Interpolate(arr, params, False, 1e-6)
    interp.Perform()
    
    if interp.IsDone():
        edge = BRepBuilderAPI_MakeEdge(interp.Curve()).Edge()
        wire = BRepBuilderAPI_MakeWire(edge).Wire()
        wires.append(Wire(wire))

print(f"Lofting {len(wires)} sections...")
t0 = time.time()
solid = NurbsSurfaceBuilder.loft(wires, solid=True, ruled=False)
t1 = time.time()
print(f"Loft time: {t1-t0:.2f} s")

occ = solid.wrapped
exp = TopExp_Explorer(occ, TopAbs_FACE)
while exp.More():
    face = TopoDS.Face_s(exp.Current())
    ad = BRepAdaptor_Surface(face)
    if ad.GetType() == GeomAbs_BSplineSurface:
        bs = ad.BSpline()
        print(f"Surface poles: {bs.NbUPoles()}x{bs.NbVPoles()} = {bs.NbUPoles()*bs.NbVPoles()}")
        print(f"Surface degree: ({bs.UDegree()},{bs.VDegree()})")
        print(f"U knots: {bs.NbUKnots()}")
        print(f"V knots: {bs.NbVKnots()}")
        
        wd = STEPControl_Writer()
        wd.Transfer(occ, STEPControl_AsIs)
        wd.Write("Exports/_tmp_interp.step")
        sz = os.path.getsize("Exports/_tmp_interp.step")
        print(f"STEP file size: {sz/1024:.0f} KB")
        break
    exp.Next()

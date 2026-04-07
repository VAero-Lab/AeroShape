"""Verify if parameterized GeomAPI_PointsToBSpline enforces identical knots."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from OCP.gp import gp_Pnt
from OCP.GeomAPI import GeomAPI_PointsToBSpline
from OCP.GeomAbs import GeomAbs_C2
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.TColStd import TColStd_Array1OfReal
from aeroshape.geometry.wings import AirfoilProfile

n = 79
params = TColStd_Array1OfReal(1, n)
for i in range(n):
    params.SetValue(i + 1, i / (n - 1))

print("Testing parameterized PointsToBSpline:")
for naca in ["0025", "4418", "2412", "2410", "0009"]:
    af = AirfoilProfile.from_naca4(naca, num_points=40)
    px, pz = af.x.copy(), af.z.copy()
    px[-1], pz[-1] = px[0], pz[0]
    arr = TColgp_Array1OfPnt(1, n)
    for i in range(n):
        arr.SetValue(i + 1, gp_Pnt(float(px[i]), 0.0, float(pz[i])))
    
    bs = GeomAPI_PointsToBSpline(arr, params, 3, 7, GeomAbs_C2, 1e-4)
    if bs.IsDone():
        c = bs.Curve()
        knots = [c.Knot(i) for i in range(1, c.NbKnots()+1)]
        knot_str = ", ".join(f"{k:.4f}" for k in knots[:5])
        print(f"  {naca:>10s} {c.NbPoles():>6d} {c.Degree():>7d} {c.NbKnots():>6d} [{knot_str}]")

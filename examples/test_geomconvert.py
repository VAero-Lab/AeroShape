"""Test builder logic for GeomConvert."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from OCP.gp import gp_Pnt, gp_Vec, gp_Dir
from OCP.Geom import Geom_Line, Geom_BezierCurve
from OCP.GeomConvert import GeomConvert_CompCurveToBSplineCurve
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.Geom import Geom_TrimmedCurve

# Line from (0,0,0) to (2,0,0)
p1 = gp_Pnt(0,0,0)
p2 = gp_Pnt(2,0,0)
vec = gp_Vec(p1, p2)
line = Geom_Line(p1, gp_Dir(vec))
# Trim the line: parameterization for Geom_Line is length along direction
line_seg = Geom_TrimmedCurve(line, 0.0, vec.Magnitude())

# Bezier curve out of the tangent
arr = TColgp_Array1OfPnt(1, 3)
arr.SetValue(1, p2)
arr.SetValue(2, gp_Pnt(4, 0, 0)) # Tangent pole
arr.SetValue(3, gp_Pnt(5, 5, 0))
bezier = Geom_BezierCurve(arr)

converter = GeomConvert_CompCurveToBSplineCurve(line_seg)
# Param: curve, tolerance, C1 constraint flag, C2 constraint flag
try:
    success = converter.Add(bezier, 1e-6)
    print(f"Added bezier: {success}")
except Exception as e:
    print(f"Failed Add: {e}")

bspline = converter.BSplineCurve()
print(f"B-Spline: First={bspline.FirstParameter()}, Last={bspline.LastParameter()}")
print(f"Poles={bspline.NbPoles()}, Degree={bspline.Degree()}")

for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
    t_eval = bspline.FirstParameter() + (t / 2.0) * (bspline.LastParameter() - bspline.FirstParameter())
    pt = bspline.Value(t_eval)
    print(f"t={t}: pt=({pt.X():.2f}, {pt.Y():.2f}, {pt.Z():.2f})")

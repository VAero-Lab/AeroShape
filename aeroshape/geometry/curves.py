"""Aerodynamic guide curve builder for planforms and lofts.

Provides an exact, mathematically robust API for constructing guide curves
(leading edges, trailing edges) using strict geometric constraints (lines, 
tangents, interpolations) without approximations causing wavy artifacts.
"""
from typing import List, Tuple
from OCP.gp import gp_Pnt, gp_Vec, gp_Dir
from OCP.Geom import Geom_Line, Geom_BezierCurve
from OCP.GeomConvert import GeomConvert_CompCurveToBSplineCurve
from OCP.Geom import Geom_TrimmedCurve
from OCP.TColgp import TColgp_Array1OfPnt, TColgp_HArray1OfPnt
from OCP.TColStd import TColStd_HArray1OfReal
from OCP.GeomAPI import GeomAPI_Interpolate, GeomAPI_PointsToBSpline
from OCP.GeomAbs import GeomAbs_C2

class GuideCurve:
    """Builder for mixed-analytical planform curves.
    
    Concatenates exact geometric segments (lines, beziers, custom NURBS)
    into a mathematically unified B-Spline curve. Very useful for avoiding
    approximation artifacts when sweeping geometries.
    """
    
    def __init__(self, start_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self._segments = []
        self._current_point = gp_Pnt(*start_point)
    
    def add_line(self, end_point: Tuple[float, float, float]):
        """Append a strictly collinear straight line.
        
        Guarantees that the resulting section of the wing will be perfectly straight.
        """
        p_end = gp_Pnt(*end_point)
        vec = gp_Vec(self._current_point, p_end)
        if vec.Magnitude() < 1e-8:
            return self
            
        line = Geom_Line(self._current_point, gp_Dir(vec))
        seg = Geom_TrimmedCurve(line, 0.0, vec.Magnitude())
        self._segments.append(seg)
        self._current_point = p_end
        return self
        
    def add_bezier(self, poles: List[Tuple[float, float, float]]):
        """Append an arbitrary piecewise Bezier curve defined by control points.
        
        The first implicit pole is forced to be the current location to preserve
        C0 continuity connection.
        
        Parameters
        ----------
        poles : list of tuple
            Remaining control points for the Bezier curve.
        """
        arr = TColgp_Array1OfPnt(1, len(poles) + 1)
        arr.SetValue(1, self._current_point)
        for i, pt in enumerate(poles):
            arr.SetValue(i + 2, gp_Pnt(*pt))
            
        bezier = Geom_BezierCurve(arr)
        self._segments.append(bezier)
        self._current_point = gp_Pnt(*poles[-1])
        return self
        
    def add_tangent_bezier(self, control_point: Tuple[float, float, float], end_point: Tuple[float, float, float]):
        """Append a quadratic Bezier ensuring tangent flow from the origin.
        
        This builds a 3-pole Bezier: [current_pos, control_point, end_point].
        If control_point is collinear with the incoming segment's end vector,
        this geometrically enforces exact global G1 transition constraints.
        """
        arr = TColgp_Array1OfPnt(1, 3)
        arr.SetValue(1, self._current_point)
        arr.SetValue(2, gp_Pnt(*control_point))
        arr.SetValue(3, gp_Pnt(*end_point))
        
        bezier = Geom_BezierCurve(arr)
        self._segments.append(bezier)
        self._current_point = gp_Pnt(*end_point)
        return self
        
    def add_interpolated_points(self, points: List[Tuple[float, float, float]]):
        """Append a segment that exactly interpolates through a list of points.
        
        Produces a C2 degree 3 B-spline with 0.00 deviation across points.
        """
        if not points:
            return self
            
        target_pts = [self._current_point] + [gp_Pnt(*p) for p in points]
        n_pts = len(target_pts)
        
        arr = TColgp_HArray1OfPnt(1, n_pts)
        params = TColStd_HArray1OfReal(1, n_pts)
        for i, pt in enumerate(target_pts):
            arr.SetValue(i + 1, pt)
            params.SetValue(i + 1, i / (n_pts - 1))
            
        interp = GeomAPI_Interpolate(arr, params, False, 1e-6)
        interp.Perform()
        if not interp.IsDone():
            raise RuntimeError("Failed to interpolate constraint points.")
            
        self._segments.append(interp.Curve())
        self._current_point = target_pts[-1]
        return self
        
    def add_fitted_points(self, points: List[Tuple[float, float, float]], degree: int = 5, tolerance: float = 1e-3):
        """Legacy behavior: append an error-tolerance approximate B-spline.
        
        WARNING: May generate waviness/ringing across dense or collinear regions.
        Use `add_interpolated_points` or `add_line` instead for rigorous geometry.
        """
        if not points:
            return self
            
        target_pts = [self._current_point] + [gp_Pnt(*p) for p in points]
        n_pts = len(target_pts)
        arr = TColgp_Array1OfPnt(1, n_pts)
        for i, pt in enumerate(target_pts):
            arr.SetValue(i + 1, pt)
            
        bspline = GeomAPI_PointsToBSpline(arr, 3, degree, GeomAbs_C2, tolerance)
        if not bspline.IsDone():
            bspline = GeomAPI_PointsToBSpline(arr)
            
        self._segments.append(bspline.Curve())
        self._current_point = target_pts[-1]
        return self

    def build_occ_curve(self):
        """Compile segments into a single continuous unified Geom_BSplineCurve.
        
        Returns the native OpenCASCADE curve handle.
        """
        if not self._segments:
            raise ValueError("GuideCurve has no segments. Add shapes first.")
            
        converter = GeomConvert_CompCurveToBSplineCurve(self._segments[0])
        for seg in self._segments[1:]:
            success = converter.Add(seg, 1e-4)
            if not success:
                raise RuntimeError("Failed to connect GuideCurve segments. Ensure geometry is unbroken.")
                
        return converter.BSplineCurve()

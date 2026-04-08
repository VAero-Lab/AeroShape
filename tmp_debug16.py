"""Test periodic B-spline without explicit parameters."""
from aeroshape.geometry.airfoils import AirfoilProfile
from OCP.gp import gp_Pnt, gp_Vec
from OCP.GeomAPI import GeomAPI_Interpolate
from OCP.TColgp import TColgp_HArray1OfPnt
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
import numpy as np

prof = AirfoilProfile.from_naca4("0012", num_points=50)
px, pz = prof.x.copy(), prof.z.copy()
px *= 3.0
pz *= 3.0

# For periodic, drop the last duplicate point
n_p = len(px) - 1
arr_p = TColgp_HArray1OfPnt(1, n_p)
for i in range(n_p):
    arr_p.SetValue(i + 1, gp_Pnt(float(px[i]), 0, float(pz[i])))

# Periodic without explicit parameters
try:
    interp_p = GeomAPI_Interpolate(arr_p, True, 1e-6)  # Periodic=True, no params
    interp_p.Perform()
    if interp_p.IsDone():
        curve_p = interp_p.Curve()
        print("Periodic interpolation succeeded!")
        print(f"  Periodic: {curve_p.IsPeriodic()}")
        print(f"  Degree: {curve_p.Degree()}")
        print(f"  Poles: {curve_p.NbPoles()}")
        print(f"  First/Last param: {curve_p.FirstParameter():.6f} / {curve_p.LastParameter():.6f}")
        
        # Check tangent at seam
        d1_s = gp_Vec()
        p_s = gp_Pnt()
        curve_p.D1(curve_p.FirstParameter(), p_s, d1_s)
        print(f"  Start pt: ({p_s.X():.4f}, {p_s.Y():.4f}, {p_s.Z():.4f})")
        print(f"  Start tan: ({d1_s.X():.4f}, {d1_s.Y():.4f}, {d1_s.Z():.4f})")
        
        # Build wire and loft
        edge_p = BRepBuilderAPI_MakeEdge(curve_p, curve_p.FirstParameter(), curve_p.LastParameter()).Edge()
        mk_wire = BRepBuilderAPI_MakeWire(edge_p)
        print(f"  Wire ok: {mk_wire.IsDone()}")
        
        # Now export a lofted shape
        from aeroshape.nurbs.export import NurbsExporter
        from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
        from build123d import Shape
        import os
        os.makedirs("Exports", exist_ok=True)
        
        wires = []
        for z_pos in [0, 2, 4]:
            chord = 3.0 - z_pos * 0.2
            p_l = prof.scaled(chord) if abs(chord - prof.chord) > 1e-10 else prof
            px_l, pz_l = p_l.x.copy(), p_l.z.copy()
            
            n_l = len(px_l) - 1  # Drop closing duplicate
            arr = TColgp_HArray1OfPnt(1, n_l)
            for i in range(n_l):
                arr.SetValue(i + 1, gp_Pnt(float(px_l[i]), 5.0 - float(pz_l[i]), float(z_pos)))
            
            interp = GeomAPI_Interpolate(arr, True, 1e-6)
            interp.Perform()
            if not interp.IsDone():
                print(f"Failed to interpolate at z={z_pos}")
                break
            c = interp.Curve()
            edge = BRepBuilderAPI_MakeEdge(c, c.FirstParameter(), c.LastParameter()).Edge()
            wires.append(BRepBuilderAPI_MakeWire(edge).Wire())
        
        if len(wires) == 3:
            loft = BRepOffsetAPI_ThruSections(True, False)
            for w in wires:
                loft.AddWire(w)
            loft.Build()
            NurbsExporter.to_step(Shape(loft.Shape()), "Exports/test_te_periodic.step")
            print("Exported test_te_periodic.step (periodic wires)")
    else:
        print("Periodic interpolation did not converge.")
except Exception as e:
    print(f"Periodic failed: {e}")

# Also export non-periodic for comparison
wires_np = []
for z_pos in [0, 2, 4]:
    chord = 3.0 - z_pos * 0.2
    p_l = prof.scaled(chord) if abs(chord - prof.chord) > 1e-10 else prof
    px_l, pz_l = p_l.x.copy(), p_l.z.copy()
    px_l[-1], pz_l[-1] = px_l[0], pz_l[0]
    
    from OCP.TColStd import TColStd_HArray1OfReal
    n_l = len(px_l)
    arr = TColgp_HArray1OfPnt(1, n_l)
    params = TColStd_HArray1OfReal(1, n_l)
    for i in range(n_l):
        arr.SetValue(i + 1, gp_Pnt(float(px_l[i]), 5.0 - float(pz_l[i]), float(z_pos)))
        params.SetValue(i + 1, i / (n_l - 1))
    interp = GeomAPI_Interpolate(arr, params, False, 1e-6)
    interp.Perform()
    edge = BRepBuilderAPI_MakeEdge(interp.Curve()).Edge()
    wires_np.append(BRepBuilderAPI_MakeWire(edge).Wire())

loft_np = BRepOffsetAPI_ThruSections(True, False)
for w in wires_np:
    loft_np.AddWire(w)
loft_np.Build()
from aeroshape.nurbs.export import NurbsExporter
from build123d import Shape
NurbsExporter.to_step(Shape(loft_np.Shape()), "Exports/test_te_nonperiodic.step")
print("Exported test_te_nonperiodic.step (non-periodic wires)")

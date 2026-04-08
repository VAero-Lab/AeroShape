"""Test periodic vs non-periodic B-spline interpolation for the airfoil wire.

Periodic spline forces tangent continuity at the TE seam, which should
eliminate the fat ribbon artifact in the STEP tessellation.
"""
from aeroshape.geometry.airfoils import AirfoilProfile
from OCP.gp import gp_Pnt, gp_Vec
from OCP.GeomAPI import GeomAPI_Interpolate
from OCP.TColgp import TColgp_HArray1OfPnt
from OCP.TColStd import TColStd_HArray1OfReal
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
from OCP.BRepAdaptor import BRepAdaptor_CompCurve
import numpy as np

prof = AirfoilProfile.from_naca4("0012", num_points=50)
px, pz = prof.x.copy(), prof.z.copy()

# Force closure
px[-1], pz[-1] = px[0], pz[0]

# Scale to 3m chord
px *= 3.0
pz *= 3.0

# Standard (non-periodic)
n = len(px)
arr_np = TColgp_HArray1OfPnt(1, n)
params_np = TColStd_HArray1OfReal(1, n)
for i in range(n):
    arr_np.SetValue(i + 1, gp_Pnt(float(px[i]), 0, float(pz[i])))
    params_np.SetValue(i + 1, i / (n - 1))
interp_np = GeomAPI_Interpolate(arr_np, params_np, False, 1e-6)
interp_np.Perform()
curve_np = interp_np.Curve()

# Periodic
# For periodic, we need to drop the last duplicate point
n_p = n - 1  # Remove the closing duplicate
arr_p = TColgp_HArray1OfPnt(1, n_p)
params_p = TColStd_HArray1OfReal(1, n_p)
for i in range(n_p):
    arr_p.SetValue(i + 1, gp_Pnt(float(px[i]), 0, float(pz[i])))
    params_p.SetValue(i + 1, i / n_p)

interp_p = GeomAPI_Interpolate(arr_p, params_p, True, 1e-6)  # Periodic=True
interp_p.Perform()

if interp_p.IsDone():
    curve_p = interp_p.Curve()
    print("Periodic interpolation succeeded!")
    print(f"  Periodic: {curve_p.IsPeriodic()}")
    print(f"  Degree: {curve_p.Degree()}")
    print(f"  Poles: {curve_p.NbPoles()}")
    
    # Check start/end tangents
    d1_start = gp_Vec()
    d1_end = gp_Vec()
    p_s = gp_Pnt()
    p_e = gp_Pnt()
    curve_p.D1(curve_p.FirstParameter(), p_s, d1_start)
    curve_p.D1(curve_p.LastParameter(), p_e, d1_end)
    print(f"  Start tangent: ({d1_start.X():.6f}, {d1_start.Y():.6f}, {d1_start.Z():.6f})")
    print(f"  End tangent:   ({d1_end.X():.6f}, {d1_end.Y():.6f}, {d1_end.Z():.6f})")
    
    # Build a wire from this periodic curve
    edge_p = BRepBuilderAPI_MakeEdge(curve_p, curve_p.FirstParameter(), curve_p.LastParameter()).Edge()
    wire_p = BRepBuilderAPI_MakeWire(edge_p)
    print(f"  Wire built: {wire_p.IsDone()}")
    
    # Compare with non-periodic
    d1_np_start = gp_Vec()
    d1_np_end = gp_Vec()
    p_np_s = gp_Pnt()
    p_np_e = gp_Pnt()
    curve_np.D1(curve_np.FirstParameter(), p_np_s, d1_np_start)
    curve_np.D1(curve_np.LastParameter(), p_np_e, d1_np_end)
    print(f"\nNon-periodic tangents:")
    print(f"  Start tangent: ({d1_np_start.X():.6f}, {d1_np_start.Y():.6f}, {d1_np_start.Z():.6f})")
    print(f"  End tangent:   ({d1_np_end.X():.6f}, {d1_np_end.Y():.6f}, {d1_np_end.Z():.6f})")
    
    # Now export both to STEP
    from aeroshape.nurbs.export import NurbsExporter
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from build123d import Shape
    import os
    os.makedirs("Exports", exist_ok=True)
    
    # Build 3 section wires using periodic interpolation for vertical wing
    wires_p = []
    wires_np = []
    for z in [0, 2, 4]:
        chord = 3.0 - z * 0.2
        p_local = prof.scaled(chord) if abs(chord - prof.chord) > 1e-10 else prof
        px_l, pz_l = p_local.x.copy(), p_local.z.copy()
        px_l[-1], pz_l[-1] = px_l[0], pz_l[0]
        
        n_l = len(px_l)
        
        # Non-periodic wire
        arr = TColgp_HArray1OfPnt(1, n_l)
        params = TColStd_HArray1OfReal(1, n_l)
        for i in range(n_l):
            arr.SetValue(i + 1, gp_Pnt(float(px_l[i]), 5 - float(pz_l[i]), float(z)))
            params.SetValue(i + 1, i / (n_l - 1))
        interp = GeomAPI_Interpolate(arr, params, False, 1e-6)
        interp.Perform()
        edge = BRepBuilderAPI_MakeEdge(interp.Curve()).Edge()
        wires_np.append(BRepBuilderAPI_MakeWire(edge).Wire())
        
        # Periodic wire (drop last duplicate point)
        n_p_l = n_l - 1
        arr2 = TColgp_HArray1OfPnt(1, n_p_l)
        params2 = TColStd_HArray1OfReal(1, n_p_l)
        for i in range(n_p_l):
            arr2.SetValue(i + 1, gp_Pnt(float(px_l[i]), 5 - float(pz_l[i]), float(z)))
            params2.SetValue(i + 1, i / n_p_l)
        interp2 = GeomAPI_Interpolate(arr2, params2, True, 1e-6)
        interp2.Perform()
        if interp2.IsDone():
            c = interp2.Curve()
            edge2 = BRepBuilderAPI_MakeEdge(c, c.FirstParameter(), c.LastParameter()).Edge()
            wires_p.append(BRepBuilderAPI_MakeWire(edge2).Wire())
    
    # Loft non-periodic
    loft_np = BRepOffsetAPI_ThruSections(True, False)
    for w in wires_np:
        loft_np.AddWire(w)
    loft_np.Build()
    NurbsExporter.to_step(Shape(loft_np.Shape()), "Exports/test_te_nonperiodic.step")
    print("\nExported test_te_nonperiodic.step")
    
    # Loft periodic
    if wires_p:
        loft_p = BRepOffsetAPI_ThruSections(True, False)
        for w in wires_p:
            loft_p.AddWire(w)
        loft_p.Build()
        NurbsExporter.to_step(Shape(loft_p.Shape()), "Exports/test_te_periodic.step")
        print("Exported test_te_periodic.step")
else:
    print("Periodic interpolation FAILED!")
    print("This may mean the periodic constraint can't be satisfied.")

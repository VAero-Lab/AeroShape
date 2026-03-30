"""Box Wing Assembly Configuration.

Demonstrates assembling a robust box wing aircraft topology.
Highlights the use of deeply embedded MultiSegmentWings (lower deck,
upper deck, and spanning winglets) to construct a continuous closed-loop 
surface natively supported by OpenCASCADE boolean representations.
"""

import os
import sys
from aeroshape import AircraftModel, show_interactive
from aeroshape.geometry.fuselage import FuselageSegment, MultiSegmentFuselage
from aeroshape.geometry.cross_sections import EllipticalProfile
from aeroshape.geometry.wings import MultiSegmentWing, SegmentSpec, AirfoilProfile

import math

def ellipsoid_blend(t):
    """Produces a perfectly rounded 1/4 ellipsoid lobe from t=0 to t=1."""
    return math.sqrt(1.0 - (1.0 - t)**2)

def inverse_paraboloid_blend(t):
    """Upswept tail closing paraboloid from t=0 (full) to t=1 (point)."""
    return 1.0 - math.sqrt(1.0 - t)

def create_box_fuselage() -> MultiSegmentFuselage:
    fuse = MultiSegmentFuselage(name="Airliner Fuselage")
    
    # Exact geometric profiles
    nose_root = EllipticalProfile(width=0.01, height=0.01)    # Spherical coordinate point
    fuse_mid  = EllipticalProfile(width=10.5, height=10.0)      # Passenger cabin bounds
    apu_tip   = EllipticalProfile(width=0.75, height=0.75)      # Tail APU exhaust

    # 1. Nose Segment (Exact Ellipsoid boundary, drooping nose)
    fuse.add_segment(FuselageSegment(
        length=12.0, root_profile=nose_root, tip_profile=fuse_mid, 
        z_offset=-0.6, num_sections=20, blend_curve=ellipsoid_blend
    ))
    
    # 2. Midbody Segment (Constant Tubular Passenger Cabin)
    fuse.add_segment(FuselageSegment(
        length=30.0, root_profile=fuse_mid, num_sections=5
    ))
    
    # 3. Tail Segment (Paraboloid Ogive geometry, sweeps upward)
    fuse.add_segment(FuselageSegment(
        length=18.0, root_profile=fuse_mid, tip_profile=apu_tip, 
        z_offset=1.8, num_sections=20, blend_curve=inverse_paraboloid_blend
    ))
    return fuse

def bezier_quadratic(t, p0, p1, p2):
    """Evaluate a quadratic Bezier curve at parameter t [0, 1]."""
    u = 1.0 - t
    return (
        u**2 * p0[0] + 2*u*t * p1[0] + t**2 * p2[0],
        u**2 * p0[1] + 2*u*t * p1[1] + t**2 * p2[1],
        u**2 * p0[2] + 2*u*t * p1[2] + t**2 * p2[2]
    )

def create_box_wings() -> list:
    naca_lower = AirfoilProfile.from_naca4("2412", num_points=50)
    naca_upper = AirfoilProfile.from_naca4("0012", num_points=50)

    # Expanded 40m wingspan for much larger volumetric topology!
    span_val = 40.0

    lower = MultiSegmentWing(name="Lower Deck", symmetric=True)
    lower.add_segment(SegmentSpec(
        span=span_val, root_airfoil=naca_lower, root_chord=7.5, tip_chord=4.5,
        sweep_le_deg=10.0, dihedral_deg=3.0
    ))
    
    upper = MultiSegmentWing(name="Upper Deck", symmetric=True)
    upper.add_segment(SegmentSpec(
        span=span_val, root_airfoil=naca_upper, root_chord=6.0, tip_chord=4.5,
        sweep_le_deg=-5.0, dihedral_deg=-1.0
    ))

    # Roots explicitly stationed perfectly embedding into the exact center plane (Y=0.0)
    loc_lower = (16.0, 0.0, -5)
    loc_upper = (24.0, 0.0, 2)
    
    # Calculate exact Tip boundaries parametrically defining P0 and P5 geometries:
    le0 = (loc_lower[0] + span_val * math.tan(math.radians(10.0)), span_val, loc_lower[2] + span_val * math.tan(math.radians(3.0)))
    te0 = (le0[0] + 4.5, span_val, le0[2])

    le5 = (loc_upper[0] + span_val * math.tan(math.radians(-5.0)), span_val, loc_upper[2] + span_val * math.tan(math.radians(-1.0)))
    te5 = (le5[0] + 4.5, span_val, le5[2])

    # EXACT G1 TANGENCY MATH FOR THE WINGLET:
    d_out = 3.0
    flat_y = span_val + d_out
    
    p1_le = (le0[0] + d_out * math.tan(math.radians(10.0)), flat_y, le0[2] + d_out * math.tan(math.radians(3.0)))
    p1_te = (te0[0] + d_out * math.tan(math.radians(10.0)), flat_y, te0[2] + d_out * math.tan(math.radians(3.0)))
    
    p4_le = (le5[0] + d_out * math.tan(math.radians(-5.0)), flat_y, le5[2] + d_out * math.tan(math.radians(-1.0)))
    p4_te = (te5[0] + d_out * math.tan(math.radians(-5.0)), flat_y, te5[2] + d_out * math.tan(math.radians(-1.0)))
    
    p2_le = (p1_le[0] + 0.3*(p4_le[0]-p1_le[0]), flat_y, p1_le[2] + 0.3*(p4_le[2]-p1_le[2]))
    p3_le = (p1_le[0] + 0.7*(p4_le[0]-p1_le[0]), flat_y, p1_le[2] + 0.7*(p4_le[2]-p1_le[2]))

    p2_te = (p1_te[0] + 0.3*(p4_te[0]-p1_te[0]), flat_y, p1_te[2] + 0.3*(p4_te[2]-p1_te[2]))
    p3_te = (p1_te[0] + 0.7*(p4_te[0]-p1_te[0]), flat_y, p1_te[2] + 0.7*(p4_te[2]-p1_te[2]))

    le_pts, te_pts = [], []
    num_c, num_s = 20, 10
    
    # P0 -> P1 -> P2 smoothly curving outwards
    for i in range(num_c):
        t = i / float(num_c)
        le_pts.append(bezier_quadratic(t, le0, p1_le, p2_le))
        te_pts.append(bezier_quadratic(t, te0, p1_te, p2_te))
        
    # P2 -> P3 strictly linear flat pillar
    for i in range(num_s):
        t = i / float(num_s)
        le_pts.append((p2_le[0] + t*(p3_le[0]-p2_le[0]), flat_y, p2_le[2] + t*(p3_le[2]-p2_le[2])))
        te_pts.append((p2_te[0] + t*(p3_te[0]-p2_te[0]), flat_y, p2_te[2] + t*(p3_te[2]-p2_te[2])))
        
    # P3 -> P4 -> P5 cleanly matching upper wing tangents natively
    for i in range(num_c):
        t = i / float(num_c - 1)
        le_pts.append(bezier_quadratic(t, p3_le, p4_le, le5))
        te_pts.append(bezier_quadratic(t, p3_te, p4_te, te5))

    fin = MultiSegmentWing.from_planform_curves(
        le_points=le_pts, te_points=te_pts,
        airfoil_stations=[(0.0, naca_lower), (1.0, naca_upper)],
        num_sections=50, name="G1 Continuous Flat Fin"
    )
    fin.symmetric = True

    return [
        (lower, loc_lower),
        (upper, loc_upper),
        (fin, (0.0, 0.0, 0.0)) 
    ]

def main():
    ac = AircraftModel("Box Wing Configuration")
    ac.add_fuselage(create_box_fuselage())
    for w, p in create_box_wings():
        ac.add_wing(w, origin=p)

    props = ac.compute_properties(method='gvm', density=2000.0) 
    print(f"Volume: {props['volume']:.2f} m^3")
    print(f"Mass:   {props['mass']:.1f} kg")

    os.makedirs("Exports", exist_ok=True)
    from aeroshape.nurbs.export import NurbsExporter
    export_path = "Exports/aircraft_box_wing.step"
    NurbsExporter.to_step(ac.to_occ_shape(fuse=False), export_path)
    print(f"Exported Assembly to {export_path}")

    if "--no-show" not in sys.argv:
        show_interactive(ac.to_triangles(), props['volume'], props['mass'], props['cg'], props['inertia'], title="Box Wing Aircraft")

if __name__ == "__main__":
    main()

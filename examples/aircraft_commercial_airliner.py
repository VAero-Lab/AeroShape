"""Modern Commercial Airliner Configuration.

Demonstrates assembling a robust, highly realistic commercial
transport aircraft. Unifies explicit mathematical fuselage primitives
(Ellipsoids and Paraboloids) with comprehensive MultiSegmentWing
assemblies for the main wing, horizontal stabilizer, and vertical tail.
"""

import os
import math
import sys
from aeroshape import AircraftModel, show_interactive
from aeroshape.geometry.fuselage import FuselageSegment, MultiSegmentFuselage
from aeroshape.geometry.cross_sections import EllipticalProfile
from aeroshape.geometry.wings import MultiSegmentWing, SegmentSpec, AirfoilProfile

def ellipsoid_blend(t):
    """Produces a perfectly rounded 1/4 ellipsoid lobe from t=0 to t=1."""
    return math.sqrt(1.0 - (1.0 - t)**2)

def inverse_paraboloid_blend(t):
    """Upswept tail closing paraboloid from t=0 (full) to t=1 (point)."""
    return 1.0 - math.sqrt(1.0 - t)

def create_fuselage() -> MultiSegmentFuselage:
    fuse = MultiSegmentFuselage(name="Airliner Fuselage")
    
    # Exact geometric profiles
    nose_root = EllipticalProfile(width=0.01, height=0.01)    # Spherical coordinate point
    fuse_mid  = EllipticalProfile(width=4.0, height=4.5)      # Passenger cabin bounds
    apu_tip   = EllipticalProfile(width=0.4, height=0.4)      # Tail APU exhaust

    # 1. Nose Segment (Exact Ellipsoid boundary, drooping nose)
    fuse.add_segment(FuselageSegment(
        length=5.0, root_profile=nose_root, tip_profile=fuse_mid, 
        z_offset=-0.6, num_sections=20, blend_curve=ellipsoid_blend
    ))
    
    # 2. Midbody Segment (Constant Tubular Passenger Cabin)
    fuse.add_segment(FuselageSegment(
        length=25.0, root_profile=fuse_mid, num_sections=5
    ))
    
    # 3. Tail Segment (Paraboloid Ogive geometry, sweeps upward)
    fuse.add_segment(FuselageSegment(
        length=9.0, root_profile=fuse_mid, tip_profile=apu_tip, 
        z_offset=1.8, num_sections=20, blend_curve=inverse_paraboloid_blend
    ))
    return fuse

def create_wings() -> list:
    naca_2412 = AirfoilProfile.from_naca4("2412", num_points=50)
    naca_0010 = AirfoilProfile.from_naca4("0010", num_points=50)
    naca_0008 = AirfoilProfile.from_naca4("0008", num_points=50)

    wings = []
    
    # -- 1. Main Wing (Swept, Dihedral) --
    main_wing = MultiSegmentWing(name="Main Wing", symmetric=True)
    # Wing root buried smoothly into the fuselage side (y=1.5)
    main_wing.add_segment(SegmentSpec(
        span=10.0, root_airfoil=naca_2412, root_chord=10.0, tip_chord=4.5,
        sweep_le_deg=30.0, dihedral_deg=5.0
    ))
    # Outer wing extending to a winglet/tip
    main_wing.add_segment(SegmentSpec(
        span=10.0, root_airfoil=naca_0010, root_chord=4.5, tip_chord=1.75,
        sweep_le_deg=32.0, dihedral_deg=6.0
    ))
    wings.append((main_wing, (12.0, 1.5, -1.0)))

    # -- 2. Airliner G1 Winglet using piecewise Bézier from_planform_curves --
    w_origin = [12.0, 1.5, -1.0] 
    
    # Calculate exactly where the main wing tip ends to build the winglet structurally out from it.
    span1, span2 = 10.0, 10.0
    dx1 = span1 * math.tan(math.radians(30.0))
    dz1 = span1 * math.tan(math.radians(5.0))
    dx2 = span2 * math.tan(math.radians(32.0))
    dz2 = span2 * math.tan(math.radians(6.0))
    
    tip_y = w_origin[1] + span1 + span2
    tip_x = w_origin[0] + dx1 + dx2
    tip_z = w_origin[2] + dz1 + dz2
    tip_chord = 1.75
    
    le0 = (tip_x, tip_y, tip_z)
    te0 = (tip_x + tip_chord, tip_y, tip_z)

    # Tangency projection (matching outer_wing): sweep=32.0, dihedral=6.0
    d_out = 0.5 
    p1_le = (le0[0] + d_out * math.tan(math.radians(32.0)), tip_y + d_out, le0[2] + d_out * math.tan(math.radians(6.0)))
    p1_te = (te0[0] + d_out * math.tan(math.radians(32.0)), tip_y + d_out, te0[2] + d_out * math.tan(math.radians(6.0)))
    
    # Winglet tip extends primarily UPWARDS! (Z domain) -> Collinear to match tangent curve flawlessly
    p2_le = (p1_le[0] + 0.5, p1_le[1], p1_le[2] + 2.0)
    p2_te = (p1_te[0] + 0.5, p1_te[1], p1_te[2] + 2.0)

    def bezier_quadratic(t, p0, p1, p2):
        u = 1.0 - t
        return (
            u**2 * p0[0] + 2*u*t * p1[0] + t**2 * p2[0],
            u**2 * p0[1] + 2*u*t * p1[1] + t**2 * p2[1],
            u**2 * p0[2] + 2*u*t * p1[2] + t**2 * p2[2]
        )

    le_pts, te_pts = [], []
    num_pts = 25
    for i in range(num_pts):
        t = i / float(num_pts - 1)
        le_pts.append(bezier_quadratic(t, le0, p1_le, p2_le))
        te_pts.append(bezier_quadratic(t, te0, p1_te, p2_te))

    winglet = MultiSegmentWing.from_planform_curves(
        le_points=le_pts, te_points=te_pts,
        airfoil_stations=[(0.0, naca_0010), (1.0, naca_0010)],
        num_sections=25, name="Smooth Blended Winglet"
    )
    winglet.symmetric = True
    wings.append((winglet, (0.0, 0.0, 0.0)))

    # -- 3. V-Tail Stabilizer --
    v_tail = MultiSegmentWing(name="V-Tail Stabilizer", symmetric=True)
    v_tail.add_segment(SegmentSpec(
        span=6.0, root_airfoil=naca_0008, root_chord=4.5, tip_chord=1.2,
        sweep_le_deg=40.0, dihedral_deg=45.0 
    ))
    wings.append((v_tail, (30.0, 0, 1.)))
    
    return wings

def main():
    ac = AircraftModel("Commercial Airliner")
    
    # 1. Attach the core fuselage geometry
    ac.add_fuselage(create_fuselage())
    
    # 2. Attach symmetric main wings and stabilizers
    for wing, pos in create_wings():
        ac.add_wing(wing, origin=pos)

    # Compute high-fidelity volume metrics organically across all NURBS patches
    props = ac.compute_properties(method='gvm', density=3000.0) # Density models payload/structure
    print(f"Volume: {props['volume']:.2f} m^3")
    print(f"Mass:   {props['mass']:.1f} kg")

    # Native representation STEP export
    from aeroshape.nurbs.export import NurbsExporter
    os.makedirs("Exports", exist_ok=True)
    export_path = "Exports/aircraft_commercial_airliner.step"
    NurbsExporter.to_step(ac.to_occ_shape(fuse=False), export_path)
    print(f"Exported High-Fidelity CAD to {export_path}")

    # Visualize 3D Mesh natively generated from NURBS
    if "--no-show" not in sys.argv:
        tris = ac.to_triangles()
        show_interactive(tris, props['volume'], props['mass'], props['cg'], props['inertia'], title="Airliner Assembly")

if __name__ == "__main__":
    main()

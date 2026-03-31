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
from aeroshape.geometry.fuselage import FuselageSegment, MultiSegmentFuselage, ellipsoid_blend, inverse_paraboloid_blend
from aeroshape.geometry.cross_sections import EllipticalProfile
from aeroshape.geometry.wings import MultiSegmentWing, SegmentSpec, AirfoilProfile
import time

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
        length=25.0, root_profile=fuse_mid, num_sections=20
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
    
    # 3. Dynamic Winglet Factory
    # The G1 analytic explicit bridge natively bounds to the host `main_wing` local frames!
    winglet = MultiSegmentWing.create_blended_winglet(
        base_wing=main_wing,
        height_z=2.0,
        sweep_out_y=0.5,
        tip_chord_ratio=0.4,
        num_sections=25,
        name="Smooth Blended Winglet"
    )
    # Winglet intrinsically mirrors the outer boundary tangency completely relative to the host's root origin.
    wings.append((winglet, tuple(w_origin)))

    # -- 3. V-Tail Stabilizer --
    v_tail = MultiSegmentWing(name="V-Tail Stabilizer", symmetric=True)
    v_tail.add_segment(SegmentSpec(
        span=6.0, root_airfoil=naca_0008, root_chord=4.5, tip_chord=1.2,
        sweep_le_deg=40.0, dihedral_deg=45.0 
    ))
    wings.append((v_tail, (30.0, 0, 1.)))
    
    return wings

def main():

    start_time = time.time()
    ac = AircraftModel("Commercial Airliner")
    
    # 1. Attach the core fuselage geometry
    ac.add_fuselage(create_fuselage())
    
    # 2. Attach symmetric main wings and stabilizers
    for wing, pos in create_wings():
        ac.add_wing(wing, origin=pos)
    end_time = time.time()
    print(f"Time to create aircraft: {end_time - start_time:.2f} seconds")

    start_time2 = time.time()
    # Compute high-fidelity mass properties in parallel with non-adaptive integration
    props = ac.compute_properties(method='occ', density=3000.0, uproc=True, tolerance=0.1)
    print(f"Volume: {props['volume']:.2f} m^3")
    print(f"Mass:   {props['mass']:.1f} kg")
    end_time2 = time.time()
    print(f"Time to compute properties: {end_time2 - start_time2:.2f} seconds")

    start_time3 = time.time()
    # Native representation STEP export (optimized assembly-aware writer)
    from aeroshape.nurbs.export import NurbsExporter
    os.makedirs("Exports", exist_ok=True)
    step_path = "Exports/aircraft_commercial_airliner.step"
    NurbsExporter.to_step(ac.to_occ_shape(), step_path)
    end_time3 = time.time()
    print(f"Time to export STEP: {end_time3 - start_time3:.2f} seconds")
    print(f"Exported High-Fidelity CAD (STEP) to {step_path}")

    # Visualize 3D Mesh natively generated from NURBS for smooth rendering
    if "--no-show" not in sys.argv:
        tris = ac.to_triangles(num_points_profile=80)
        show_interactive(tris, props['volume'], props['mass'], props['cg'], props['inertia'], title="Commercial Airliner")

if __name__ == "__main__":
    main()

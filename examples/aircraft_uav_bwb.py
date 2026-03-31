"""UAV Blended Wing Body Configuration.

Demonstrates integrating a flattened stealth-like body lifting profile
with perfectly swept multi-segment wings. Utilizes exact Elliptical
cross sections seamlessly feeding into a Circular exhaust boundary.
"""

import os
import sys
import math
import time

from aeroshape import AircraftModel, show_interactive
from aeroshape.geometry.fuselage import FuselageSegment, MultiSegmentFuselage
from aeroshape.geometry.cross_sections import EllipticalProfile, CircularProfile
from aeroshape.geometry.wings import MultiSegmentWing, SegmentSpec, AirfoilProfile

def paraboloid_blend(t):
    return math.sqrt(t)

def inverse_paraboloid_blend(t):
    return 1.0 - math.sqrt(1.0 - t)

def create_uav_body() -> MultiSegmentFuselage:
    fuse = MultiSegmentFuselage(name="BWB Core")
    
    # Generate parametric profiles for mathematically flawless lofts.
    nose_point = EllipticalProfile(width=0.01, height=0.01)
    mid_blend  = EllipticalProfile(width=4.0, height=2.25)  # Flattened stealth body
    exhaust    = CircularProfile(radius=0.6)

    # 1. Nose sweep to centerbody
    fuse.add_segment(FuselageSegment(
        length=4.0, root_profile=nose_point, tip_profile=mid_blend, 
        z_offset=0.2, num_sections=20, blend_curve=paraboloid_blend
    ))
    # 2. Central thick payload bay
    fuse.add_segment(FuselageSegment(
        length=5.0, root_profile=mid_blend, tip_profile=mid_blend
    ))
    # 3. Empennage sweep to uniform exhaust
    fuse.add_segment(FuselageSegment(
        length=4.0, root_profile=mid_blend, tip_profile=exhaust, 
        z_offset=0.3, num_sections=20, blend_curve=inverse_paraboloid_blend
    ))
    return fuse

def create_uav_wings() -> list:
    naca_stealth = AirfoilProfile.from_naca4("0020", num_points=50) # Symmetrical 
    naca_tip     = AirfoilProfile.from_naca4("0012", num_points=50)

    # Note: UAV is a flying wing, so we only need the massive main wings
    main_wing = MultiSegmentWing(name="Stealth Wing", symmetric=True)
    
    # The wings start at y=1.0 (embedded deeply into the 4.0-wide fuselage body)
    main_wing.add_segment(SegmentSpec(
        span=10.0, root_airfoil=naca_stealth, root_chord=7.0, tip_chord=3.0,
        sweep_le_deg=40.0, dihedral_deg=2.0 
    ))
    # Outer cranked panels
    main_wing.add_segment(SegmentSpec(
        span=6.0, root_airfoil=naca_tip, root_chord=3.0, tip_chord=0.5,
        sweep_le_deg=30.0, dihedral_deg=4.0
    ))
    return [(main_wing, (1.65, 0, 0.2))] # Mounted effortlessly deep inside the centerbody

def main():
    start_time = time.time()
    ac = AircraftModel("UAV Demo")
    
    ac.add_fuselage(create_uav_body())
    
    for wing, pos in create_uav_wings():
        ac.add_wing(wing, origin=pos)
    end_time = time.time()
    print(f"Time to create aircraft: {end_time - start_time:.2f} seconds")

    # Fast mass property extraction directly utilizing robust parallel NURBS evaluations
    start_time1 = time.time()
    props = ac.compute_properties(method='occ', density=1500.0, uproc=True, tolerance=0.1) # UAV avionics density
    end_time1 = time.time()
    print(f"Time to compute properties: {end_time1 - start_time1:.2f} seconds")
    print(f"Volume: {props['volume']:.2f} m^3")
    print(f"Mass:   {props['mass']:.1f} kg")

    start_time2 = time.time()
    # Native representation STEP export (optimized assembly-aware writer)
    from aeroshape.nurbs.export import NurbsExporter
    os.makedirs("Exports", exist_ok=True)
    export_path = "Exports/aircraft_uav_bwb.step"
    NurbsExporter.to_step(ac.to_occ_shape(), export_path)
    print(f"Exported Assembly to {export_path}")
    end_time2 = time.time()
    print(f"Time to export STEP: {end_time2 - start_time2:.2f} seconds")
    
    if "--no-show" not in sys.argv:
        show_interactive(ac.to_triangles(), props['volume'], props['mass'], props['cg'], props['inertia'], title="UAV Assembly")

if __name__ == "__main__":
    main()

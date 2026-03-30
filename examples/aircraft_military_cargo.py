"""Military Cargo Transport Configuration.

Demonstrates combining explicit `NurbsCrossSection` geometry (to produce
a rounded-square double-bubble cargo fuselage) with high-lift,
strut-braced straight wings. Highlights integration of complex multi-part
geometries into a single analyzable model.
"""

import os
import sys
import numpy as np
import math
from aeroshape import AircraftModel, show_interactive
from aeroshape.geometry.fuselage import FuselageSegment, MultiSegmentFuselage
from aeroshape.geometry.cross_sections import NurbsCrossSection, CircularProfile, EllipticalProfile
from aeroshape.geometry.wings import MultiSegmentWing, SegmentSpec, AirfoilProfile


def ellipsoid_blend(t):
    """Produces a perfectly rounded 1/4 ellipsoid lobe from t=0 to t=1."""
    return math.sqrt(1.0 - (1.0 - t)**2)

def inverse_paraboloid_blend(t):
    """Upswept tail closing paraboloid from t=0 (full) to t=1 (point)."""
    return 1.0 - math.sqrt(1.0 - t)

def create_cargo_fuselage() -> MultiSegmentFuselage:
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

def create_cargo_wings() -> list:
    naca_high = AirfoilProfile.from_naca4("6415", num_points=50) # High camber/lift
    naca_strut = AirfoilProfile.from_naca4("0012", num_points=50)
    wings = []
    
    # -- 1. High Mounted Main Wing --
    main_wing = MultiSegmentWing(name="Main Wing", symmetric=True)
    main_wing.add_segment(SegmentSpec(
        span=30.0, root_airfoil=naca_high, root_chord=4.5, tip_chord=2.5,
        sweep_le_deg=5.0, dihedral_deg=2.0
    ))
    # Airliner profile goes from z=-2.25 to 2.25
    # Mount high main wing deeply embedded at top of fuselage (y=1.2, z=1.8)
    wings.append((main_wing, (12.0, 0, -0.2)))

    # -- 2. Strut Braces --
    strut = MultiSegmentWing(name="Support Strut", symmetric=True)
    # The strut mounts on the belly (z=-1.8) and points sharply upwards!
    strut_length = 7.0
    strut.add_segment(SegmentSpec(
        span=strut_length, root_airfoil=naca_strut, root_chord=1.75, tip_chord=1.75,
        dihedral_deg=20.0
    ))
    # Mount deeply embedded at bottom hull of fuselage
    wings.append((strut, (13, 1.2, -2.2)))
    
    # -- 3. V-Tail Empennage --
    v_tail = MultiSegmentWing(name="V-Tail", symmetric=True)
    v_tail.add_segment(SegmentSpec(span=5.0, root_airfoil=naca_strut, root_chord=5.0, tip_chord=2.5, sweep_le_deg=25.0, dihedral_deg=50.0))
    wings.append((v_tail, (32.0, 0, 1)))

    return wings

def main():
    ac = AircraftModel("Military Cargo")
    ac.add_fuselage(create_cargo_fuselage())
    for w, p in create_cargo_wings():
        ac.add_wing(w, origin=p)

    props = ac.compute_properties(method='gvm', density=1200.0) 
    print(f"Volume: {props['volume']:.2f} m^3")
    print(f"Mass:   {props['mass']:.1f} kg")

    os.makedirs("Exports", exist_ok=True)
    from aeroshape.nurbs.export import NurbsExporter
    export_path = "Exports/aircraft_military_cargo.step"
    NurbsExporter.to_step(ac.to_occ_shape(fuse=False), export_path)
    print(f"Exported Assembly to {export_path}")

    if "--no-show" not in sys.argv:
        show_interactive(ac.to_triangles(), props['volume'], props['mass'], props['cg'], props['inertia'], title="Military Cargo")

if __name__ == "__main__":
    main()

"""Box Wing Assembly Configuration.

Demonstrates assembling a robust box wing aircraft topology.
Highlights the use of deeply embedded MultiSegmentWings (lower deck,
upper deck, and spanning winglets) to construct a continuous closed-loop 
surface natively supported by OpenCASCADE boolean representations.
"""

import os
import sys
import time
from aeroshape import AircraftModel, show_interactive
from aeroshape.geometry.fuselage import FuselageSegment, MultiSegmentFuselage, ellipsoid_blend, inverse_paraboloid_blend
from aeroshape.geometry.cross_sections import EllipticalProfile
from aeroshape.geometry.wings import MultiSegmentWing, SegmentSpec, AirfoilProfile

import math

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
    
    # 3. Piecewise Composite Fin 
    # Generates a structurally flawless G1 continuous connecting flat winglet parametrically referencing origin bounds!
    fin = MultiSegmentWing.create_box_fin(
        lower_wing=lower, upper_wing=upper,
        lower_origin=loc_lower, upper_origin=loc_upper,
        d_out=3.0, num_sections=25, name="Composite Box Fin"
    )

    return [
        (lower, loc_lower),
        (upper, loc_upper),
        (fin, (0.0, 0.0, 0.0))  # Evaluated absolutely from explicit bounds
    ]

def main():
    now = time.time()
    ac = AircraftModel("Box Wing Configuration")
    ac.add_fuselage(create_box_fuselage())
    for w, p in create_box_wings():
        ac.add_wing(w, origin=p)

    end = time.time()
    print(f"Time to create aircraft: {end - now:.2f} seconds")
    
    props = ac.compute_properties(method="occ", density=2000.0, uproc=True, tolerance=0.1) 
    print(f"Volume: {props['volume']:.2f} m^3")
    print(f"Mass:   {props['mass']:.1f} kg")
    end1 = time.time()
    print(f"Time to compute properties: {end1 - end:.2f} seconds")

    os.makedirs("Exports", exist_ok=True)
    from aeroshape.nurbs.export import NurbsExporter
    export_path = "Exports/aircraft_box_wing.step"
    NurbsExporter.to_step(ac.to_occ_shape(fuse=False), export_path)
    end2 = time.time()
    print(f"Time to export: {end2 - end1:.2f} seconds")
    print(f"Exported Assembly to {export_path}")

    if "--no-show" not in sys.argv:
        ac.show()

if __name__ == "__main__":
    main()

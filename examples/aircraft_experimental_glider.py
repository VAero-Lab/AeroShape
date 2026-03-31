"""Experimental Glider Configuration.

Demonstrates combining explicit low-level `NurbsProfile` modeling
for highly customized aerodynamics with a single continuous
curved fuselage boom utilizing the mathematical `guide_curve_z` logic.
"""

import os
import sys
import math
import numpy as np

from aeroshape import AircraftModel, show_interactive
from aeroshape.geometry.fuselage import FuselageSegment, MultiSegmentFuselage
from aeroshape.geometry.cross_sections import CircularProfile
from aeroshape.geometry.wings import MultiSegmentWing, SegmentSpec
from aeroshape.geometry.airfoils import NurbsProfile


def arch_curve_z(t):
    """A natural drooping tail boom arch."""
    return 4.0 * (t - t**2)

def create_glider_fuselage() -> MultiSegmentFuselage:
    fuse = MultiSegmentFuselage(name="Curved Boom")
    
    # Simple low-drag circular fuselage
    nose = CircularProfile(radius=0.4)
    body = CircularProfile(radius=1)
    tip = CircularProfile(radius=0.4)
    
    # 1. Nose
    fuse.add_segment(FuselageSegment(length=2.0, root_profile=nose, tip_profile=body))
    
    # 2. Arching Glider Boom (20 meters long, bends up 1 meter via parabolic arch)
    fuse.add_segment(FuselageSegment(
        length=20.0, root_profile=body, tip_profile=tip, 
        num_sections=40, guide_curve_z=arch_curve_z
    ))
    return fuse

def create_glider_wings() -> list:
    # Build an explicitly controlled reflexed airfoil using NURBS
    poles = np.array([
        [1.0, 0.0],
        [0.8, -0.05], # Under-camber
        [0.4, -0.02],
        [0.0, -0.05], # Blunt bottom LE
        [0.0, 0.05],  # Blunt top LE
        [0.3, 0.15],  # High camber upper
        [0.7, 0.05],  # Reflex upper
        [1.0, 0.0]
    ])
    # 8 poles, degree 3 -> 12 knots
    knots = np.concatenate(([0.0]*4, [0.2, 0.4, 0.6, 0.8], [1.0]*4))
    multiplicities = np.ones(len(knots), dtype=int)
    unique_knots, mults = np.unique(knots, return_counts=True)
    reflex_profile = NurbsProfile(poles, unique_knots, mults, degree=3, weights=None)

    wings = []
    
    # -- Extremely high aspect ratio main wing --
    glider_wing = MultiSegmentWing(name="Reflex Glider Wing", symmetric=True)
    glider_wing.add_segment(SegmentSpec(
        span=20.0, root_airfoil=reflex_profile, root_chord=2.25, tip_chord=0.75, sweep_le_deg=5.0
    ))
    # Mounted deeply inside the arch body (Y=0.0 center) rather than tangent
    wings.append((glider_wing, (5.0, 0.0, 0.6)))
    
    return wings

def main():
    ac = AircraftModel("Experimental Glider")
    ac.add_fuselage(create_glider_fuselage())
    for w, p in create_glider_wings():
        ac.add_wing(w, origin=p)

    props = ac.compute_properties(method='gvm', density=300.0) # Extremely light foam/carbon
    print(f"Volume: {props['volume']:.2f} m^3")
    print(f"Mass:   {props['mass']:.1f} kg")

    os.makedirs("Exports", exist_ok=True)
    from aeroshape.nurbs.export import NurbsExporter
    export_path = "Exports/aircraft_experimental_glider.step"
    NurbsExporter.to_step(ac.to_occ_shape(fuse=False), export_path)
    print(f"Exported STEP model to {export_path}")

    if "--no-show" not in sys.argv:
        show_interactive(ac.to_triangles(num_points_profile=80), props['volume'], props['mass'], props['cg'], props['inertia'], title="Experimental Glider")

if __name__ == "__main__":
    main()

"""Twin-Boom Aircraft Configuration.

Demonstrates assembling a robust multi-body aircraft involving a central
fuselage pod, two outboard tail booms, connected by spanning inner and
outer multi-segment wings. Highlights spatial assembly and positioning.
"""

import os
import sys
import math
from aeroshape import AircraftModel, show_interactive
from aeroshape.geometry.fuselage import FuselageSegment, MultiSegmentFuselage
from aeroshape.geometry.cross_sections import EllipticalProfile
from aeroshape.geometry.wings import MultiSegmentWing, SegmentSpec, AirfoilProfile

def ellipsoid_blend(t):
    return math.sqrt(1.0 - (1.0 - t)**2)

def inverse_paraboloid_blend(t):
    return 1.0 - math.sqrt(1.0 - t)

def create_central_pod() -> MultiSegmentFuselage:
    pod = MultiSegmentFuselage(name="Central Pod")
    nose = EllipticalProfile(width=0.1, height=0.1)
    body = EllipticalProfile(width=2.5, height=2.0)
    tail = EllipticalProfile(width=0.2, height=0.2)

    pod.add_segment(FuselageSegment(
        length=3.0, root_profile=nose, tip_profile=body, z_offset=-0.2
    ))
    pod.add_segment(FuselageSegment(length=8.0, root_profile=body, tip_profile=body))
    pod.add_segment(FuselageSegment(
        length=4.0, root_profile=body, tip_profile=tail, 
        z_offset=0.5, num_sections=20, blend_curve=inverse_paraboloid_blend
    ))
    return pod

def create_tail_boom(name: str) -> MultiSegmentFuselage:
    boom = MultiSegmentFuselage(name=name)
    nose = EllipticalProfile(width=0.4, height=0.4)
    tip = EllipticalProfile(width=0.1, height=0.1)
    # Long slender boom extending backwards
    boom.add_segment(FuselageSegment(
        length=12.0, root_profile=nose, tip_profile=tip,
        num_sections=30, blend_curve=inverse_paraboloid_blend
    ))
    return boom

def create_wings() -> list:
    airfoil = AirfoilProfile.from_naca4("2412", num_points=50)
    tail_airfoil = AirfoilProfile.from_naca4("0010", num_points=50)

    # 1. Central Inner Wing (connects pod to booms)
    # Spans 4.2 from the pod to embed safely inside the booms stationed at y=4.0
    inner_wing = MultiSegmentWing(name="Inner Wing", symmetric=True)
    inner_wing.add_segment(SegmentSpec(
        span=4.2, root_airfoil=airfoil, root_chord=3.0, tip_chord=3.0, sweep_le_deg=0.0
    ))

    # 2. Horizontal Stabilizer (bridges the two tail booms deeply)
    # Distance between booms is 8.0 meters. A span of 8.4 starting from -4.2 embeds 0.2m into each boom.
    h_stab = MultiSegmentWing(name="Horizontal Stabilizer", symmetric=False)
    h_stab.add_segment(SegmentSpec(
        span=8.4, root_airfoil=tail_airfoil, root_chord=1.5, tip_chord=1.5
    ))

    return [
        (inner_wing, (6.0, 0.0, 0.5)),          # Mounted securely through the mid-fuselage
        (h_stab, (17.5, -4.2, 0.5))             # Attaches firmly embedded from -Y boom to +Y boom
    ]

def main():
    ac = AircraftModel("Twin-Boom Demo")

    # 1. The Central Pod
    ac.add_fuselage(create_central_pod())

    # 2. The Tail Booms (Placed exactly at the tip of the inner wings: y = +/- 4.0)
    boom_right = create_tail_boom("Right Boom")
    ac.add_fuselage(boom_right, origin=(6.0, 4.0, 0.5))
    
    boom_left = create_tail_boom("Left Boom")
    ac.add_fuselage(boom_left, origin=(6.0, -4.0, 0.5))

    # 3. All Wing Segments
    for w, p in create_wings():
        ac.add_wing(w, origin=p)

    props = ac.compute_properties(method='gvm', density=2700.0) 
    print(f"Volume: {props['volume']:.2f} m^3")
    print(f"Mass:   {props['mass']:.1f} kg")

    os.makedirs("Exports", exist_ok=True)
    from aeroshape.nurbs.export import NurbsExporter
    export_path = "Exports/aircraft_twin_boom.step"
    NurbsExporter.to_step(ac.to_occ_shape(fuse=False), export_path)
    print(f"Exported Assembly to {export_path}")

    if "--no-show" not in sys.argv:
        show_interactive(ac.to_triangles(), props['volume'], props['mass'], props['cg'], props['inertia'], title="Twin Boom Aircraft")

if __name__ == "__main__":
    main()

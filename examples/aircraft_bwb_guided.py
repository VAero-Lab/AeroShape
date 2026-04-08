"""Blended Wing Body (BWB) with leading- and trailing-edge guide curves.

Instead of defining discrete segments (which produce linear LE/TE lines
with kinks at segment boundaries), this example defines the BWB planform
via smooth B-spline guide curves for the leading and trailing edges.

MultiSegmentWing.from_planform_curves() fits B-spline curves through the
LE and TE control points, samples them at many intermediate stations, and
lofts through all sections to produce a smooth surface with no kinks.

Key benefits over the segment-based approach:
- Smooth LE and TE transitions (C2-continuous curves)
- No visible breaks between regions (center body, transition, outboard)
- Guide curves give direct control over the planform shape
"""

import os
import sys
import numpy as np
import time
from aeroshape import AircraftModel, show_interactive
from aeroshape.geometry.fuselage import FuselageSegment, MultiSegmentFuselage, ellipsoid_blend
from aeroshape.geometry.cross_sections import EllipticalProfile
from aeroshape.geometry.wings import MultiSegmentWing, SegmentSpec, AirfoilProfile
from aeroshape.analysis.mesh import MeshTopologyManager
from aeroshape.analysis.volume import VolumeCalculator
from aeroshape.analysis.mass import MassPropertiesCalculator
from aeroshape.nurbs.export import NurbsExporter

EXPORT_DIR = "Exports"


def main():
    start_time = time.time()
    # ── Airfoil profiles at key spanwise stations ────────────────
    # Center body: very thick symmetric (cabin volume)
    center = AirfoilProfile.from_naca4("0020", num_points=40)
    # Extra center profile
    flatcenter = AirfoilProfile.from_naca4("0018", num_points=40)
    # Inner transition: thick cambered
    transition = AirfoilProfile.from_naca4("4415", num_points=40)
    # Outboard root: conventional
    outboard = AirfoilProfile.from_naca4("2412", num_points=40)
    # Outboard mid: thinner
    mid_outboard = AirfoilProfile.from_naca4("2410", num_points=40)
    # Tip: thin symmetric
    tip = AirfoilProfile.from_naca4("0009", num_points=40)

    from aeroshape.geometry.curves import GuideCurve

    # ── Leading-edge guide curve ─────────────────────────────────
    le = GuideCurve(start_point=(0.0, 0.0, 0.0))
    # 1. Manta Nose: A cubic bezier that guarantees zero sweep at the very root 
    #    (by placing the first pole at X=0.0, Y=1.0) and then sweeps back 
    #    aggressively. This forms the blended forward strake.
    le.add_bezier([(0.0, 1.75, 0.0), (8.0, 3.0, 0.0), (10.0, 6.0, 0.0)])
    
    # 2. Main Wing Sweep: A quadratic bezier continuing out to the wingtip.
    #    The control pole (12.0, 9.0) sits exactly on the vector exiting the 
    #    previous segment [(10,6) - (8,3) = (2,3) -> (10+2, 6+3) = (12,9)], 
    #    which perfectly guarantees G1 mathematical continuity across the joint!
    le.add_tangent_bezier(control_point=(12.0, 9.0, 0.0), end_point=(16.0, 15.0, 0.0))

    # ── Trailing-edge guide curve ────────────────────────────────
    te = GuideCurve(start_point=(20.0, 0.0, 0.0))
    # 1. Center engine deck: A perfectly straight, zero-sweep segment covering 
    #    the fuselage centerline to hold the engine nacelles.
    te.add_line(end_point=(20.0, 3.0, 0.0))
    
    # 2. Forward Scallop: Sweeps forward into a "bat-wing" trailing edge.
    #    The control pole (20.0, 5.0) lies perfectly along the +Y axis of the 
    #    incoming straight line, ensuring a completely smooth exit tangency.
    te.add_tangent_bezier(control_point=(20.0, 5.0, 0.0), end_point=(17.0, 8.0, 0.0))
    
    # 3. Outer sweep back: The trailing edge reverses to sweep back to the tip.
    #    The control pole (14.0, 11.0) perfectly aligns with the -3.0X, +3.0Y 
    #    tangent vector exiting the previous scallop [(17,8) - (20,5) = (-3,3)], 
    #    ensuring immaculate G1 continuity without geometric wiggles.
    te.add_tangent_bezier(control_point=(14.0, 11.0, 0.0), end_point=(18.0, 15.0, 0.0))

    # ── Airfoil stations (spanwise fraction, profile) ────────────
    # Fraction 0 = root (Y=0), fraction 1 = tip (Y=15.0)
    airfoil_stations = [
        (0.00, center),         # (Y=0) Root centerbody
        (0.20, flatcenter),     # (Y=3) End of flat engine deck
        (0.40, transition),     # (Y=6) Mid forward sweep
        (0.53, outboard),       # (Y=8) Scallop max forward point
        (0.80, mid_outboard),   # (Y=12) Mid outer wing
        (1.00, tip),            # (Y=15) Wingtip
    ]

    # ── Build wing from guide curves ─────────────────────────────
    bwb = MultiSegmentWing.from_planform_curves(
        le_curve=le,
        te_curve=te,
        airfoil_stations=airfoil_stations,
        num_sections=40,        # high-res guided loft
        name="BWB (Guide Curves)"
    )

    # ── Aircraft Assembly & Analysis ─────────────────────────────
    # By using AircraftModel, symmetry is handled automatically
    model = AircraftModel(name="BWB Guided")
    model.add_wing(bwb)
    end_time = time.time()
    print(f"Time to create aircraft: {end_time - start_time:.2f} seconds")
    
    start_time1 = time.time()
    props = model.compute_properties(method='occ', density=150.0, uproc=True, tolerance=0.1)
    volume = props['volume']
    mass = props['mass']
    cg = props['cg']
    inertia = props['inertia']
    end_time1 = time.time()
    print(f"Time to compute properties: {end_time1 - start_time1:.2f} seconds")

    print(f"Configuration: {model.name}")
    print(f"  Sections: {len(bwb.get_section_frames())}")
    print(f"  Volume: {volume:.4f} m^3")
    print(f"  Mass:   {mass:.1f} kg")
    print(f"  CG:     ({cg[0]:.3f}, {cg[1]:.3f}, {cg[2]:.3f}) m")
    print(f"  Ixx={inertia[0]:.1f}, Iyy={inertia[1]:.1f}, "
          f"Izz={inertia[2]:.1f} kg*m^2")

    start_time2 = time.time()
    # ── NURBS export ─────────────────────────────────────────────
    os.makedirs(EXPORT_DIR, exist_ok=True)
    # We can use the same model.to_occ_shape() for export
    shape = model.to_occ_shape()
    step_path = os.path.join(EXPORT_DIR, "blended_wing_body_guided.step")
    NurbsExporter.to_step(shape, step_path)
    print(f"\n  STEP exported: {step_path}")
    end_time2 = time.time()
    print(f"Time to export: {end_time2 - start_time2:.2f} seconds")
    
    # ── Visualize ────────────────────────────────────────────────
    if "--no-show" not in sys.argv:
        model.show()


if __name__ == "__main__":
    main()
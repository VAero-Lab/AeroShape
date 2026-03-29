"""Compare volume computation methods: GVM, SAI, and OCC.

Demonstrates the ``compute_properties(method=...)`` API that lets users
choose between three volume computation approaches:

- **GVM** (``method='gvm'``): Divergence-Theorem on a triangulated mesh.
  The original GVM methodology from the paper.
- **SAI** (``method='sai'``): Section-Area Integration — uses the shoelace
  formula per section + trapezoidal rule along span.  ~100x faster than
  GVM, eliminates chordwise inscribed-polygon error.
- **OCC** (``method='occ'``): Exact NURBS integration via OpenCASCADE's
  BRepGProp.  Most accurate but requires the OCC kernel.

Also shows how clustering laws can be passed to the GVM/SAI methods.
"""

import time
from aeroshape import (
    AirfoilProfile,
    SegmentSpec,
    MultiSegmentWing,
    clustering,
)


def main():
    # ── Define a three-segment wing ─────────────────────────────────
    root = AirfoilProfile.from_naca4("2412", num_points=60)
    mid = AirfoilProfile.from_naca4("2410", num_points=60)
    tip = AirfoilProfile.from_naca4("0009", num_points=60)

    wing = MultiSegmentWing(name="Cranked Wing + Winglet")
    wing.add_segment(SegmentSpec(
        span=5.0, root_airfoil=root, tip_airfoil=mid,
        root_chord=3.0, tip_chord=2.0,
        sweep_le_deg=20, dihedral_deg=3, twist_deg=-1, num_sections=12,
    ))
    wing.add_segment(SegmentSpec(
        span=4.0, root_airfoil=mid, tip_airfoil=tip,
        root_chord=2.0, tip_chord=0.8,
        sweep_le_deg=30, dihedral_deg=5, twist_deg=-2, num_sections=10,
    ))
    wing.add_segment(SegmentSpec(
        span=0.8, root_airfoil=tip, tip_airfoil=tip,
        root_chord=0.8, tip_chord=0.3,
        sweep_le_deg=45, dihedral_deg=75, num_sections=6,
    ))

    density = 2700.0  # aluminum [kg/m^3]

    # ── Compare methods ─────────────────────────────────────────────
    print(f"Wing: {wing.name}")
    print(f"Density: {density} kg/m^3")
    print()

    methods = [
        ("GVM (Divergence Theorem)", "gvm", {}),
        ("SAI (Section-Area Integration)", "sai", {}),
        ("OCC (Exact NURBS)", "occ", {}),
        ("SAI + cosine clustering", "sai",
         {"spanwise_clustering": clustering.cosine}),
    ]

    print(f"{'Method':<35s} {'Volume':>10s} {'Mass':>10s} "
          f"{'Time':>9s}")
    print("-" * 70)

    for label, method, kwargs in methods:
        t0 = time.perf_counter()
        props = wing.compute_properties(
            method=method,
            density=density,
            num_points_profile=60,
            **kwargs,
        )
        elapsed = time.perf_counter() - t0

        print(f"{label:<35s} {props['volume']:10.4f} {props['mass']:10.1f} "
              f"{elapsed * 1000:8.1f}ms")

    # ── Show full property output for the SAI method ────────────────
    print()
    props = wing.compute_properties(method="sai", density=density,
                                     num_points_profile=60)
    print("Full SAI output:")
    print(f"  Volume: {props['volume']:.6f} m^3")
    print(f"  Mass:   {props['mass']:.2f} kg")
    cg = props["cg"]
    print(f"  CG:     ({cg[0]:.4f}, {cg[1]:.4f}, {cg[2]:.4f}) m")
    inertia = props["inertia"]
    print(f"  Ixx={inertia[0]:.1f}, Iyy={inertia[1]:.1f}, "
          f"Izz={inertia[2]:.1f} kg*m^2")


if __name__ == "__main__":
    main()

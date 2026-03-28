"""Multi-segment cranked wing with winglet.

Demonstrates the NURBS-based pipeline:
1. Define airfoil profiles (NACA 4-digit)
2. Build a three-segment wing (inboard, outboard, winglet)
3. Compute volume and mass properties via GVM pipeline
4. Loft NURBS surface and export to STEP
5. Visualize the result
"""

from aeroshape import (
    AirfoilProfile,
    SegmentSpec,
    MultiSegmentWing,
    VolumeCalculator,
    MassPropertiesCalculator,
    NurbsExporter,
    show_interactive,
)
from aeroshape.mesh_utils import MeshTopologyManager

# ── 1. Define airfoil profiles ────────────────────────────────────
root = AirfoilProfile.from_naca4("2412", num_points=60)
mid = AirfoilProfile.from_naca4("2410", num_points=60)
tip = AirfoilProfile.from_naca4("0009", num_points=60)

# ── 2. Build a three-segment wing ────────────────────────────────
wing = MultiSegmentWing(name="Cranked Wing + Winglet")

# Inboard segment
wing.add_segment(SegmentSpec(
    span=5.0,
    root_airfoil=root,
    tip_airfoil=mid,
    root_chord=3.0,
    tip_chord=2.0,
    sweep_le_deg=20,
    dihedral_deg=3,
    twist_deg=-1,
    num_sections=12,
))

# Outboard segment
wing.add_segment(SegmentSpec(
    span=4.0,
    root_airfoil=mid,
    tip_airfoil=tip,
    root_chord=2.0,
    tip_chord=0.8,
    sweep_le_deg=30,
    dihedral_deg=5,
    twist_deg=-2,
    num_sections=10,
))

# Winglet
wing.add_segment(SegmentSpec(
    span=0.8,
    root_airfoil=tip,
    tip_airfoil=tip,
    root_chord=0.8,
    tip_chord=0.3,
    sweep_le_deg=45,
    dihedral_deg=75,
    num_sections=6,
))

# ── 3. Compute GVM properties ────────────────────────────────────
num_pts = 60
X, Y, Z = wing.to_vertex_grids(num_points_profile=num_pts)
triangles = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=True)
volume = VolumeCalculator.compute_solid_volume(triangles)

density = 2700.0  # aluminum, kg/m^3
mass = volume * density
cg, inertia, _ = MassPropertiesCalculator.compute_all(X, Y, Z, mass)

print(f"Wing: {wing.name}")
print(f"  Segments: {len(wing.segments)}")
print(f"  Total sections: {X.shape[0]}")
print(f"  Volume: {volume:.6f} m^3")
print(f"  Mass:   {mass:.2f} kg")
print(f"  CG:     ({cg[0]:.4f}, {cg[1]:.4f}, {cg[2]:.4f}) m")
print(f"  Ixx={inertia[0]:.2f}, Iyy={inertia[1]:.2f}, Izz={inertia[2]:.2f} kg·m²")

# ── 4. NURBS loft and STEP export ────────────────────────────────
try:
    shape = wing.to_occ_shape()
    NurbsExporter.to_step(shape, "cranked_wing.step")
    print("\n  STEP exported: cranked_wing.step")
except ImportError:
    print("\n  (Skipping STEP export — OCP not installed)")

# ── 5. Visualize ─────────────────────────────────────────────────
show_interactive(triangles, volume, mass, cg, inertia, title=wing.name)

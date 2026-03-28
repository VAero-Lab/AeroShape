"""Wing with custom airfoil profiles from .dat files or NACA 5-digit.

Demonstrates:
- Loading airfoil profiles from Selig-format .dat files
- Using NACA 5-digit series profiles
- Mixing different profile sources in a multi-segment wing
"""

import os
import numpy as np

from aeroshape import (
    AirfoilProfile,
    SegmentSpec,
    MultiSegmentWing,
    VolumeCalculator,
    MassPropertiesCalculator,
    show_interactive,
)
from aeroshape.mesh_utils import MeshTopologyManager


def create_sample_dat_file(filepath):
    """Write a sample Selig-format .dat file for demonstration."""
    # NACA 2412-like profile in Selig format (upper TE -> LE -> lower TE)
    profile = AirfoilProfile.from_naca4("2412", num_points=40)
    # Selig format: upper TE -> LE -> lower TE (reverse of our convention)
    x = profile.x[::-1] / profile.chord
    z = profile.z[::-1] / profile.chord
    with open(filepath, 'w') as f:
        f.write("SAMPLE 2412\n")
        for xi, zi in zip(x, z):
            f.write(f"  {xi:.6f}  {zi:.6f}\n")


# ── Option A: NACA 5-digit profiles ──────────────────────────────
print("=== NACA 5-digit wing ===")
root_5 = AirfoilProfile.from_naca5("23012", num_points=60)
tip_5 = AirfoilProfile.from_naca5("23009", num_points=60)

wing_5 = MultiSegmentWing(name="NACA 5-Digit Wing")
wing_5.add_segment(SegmentSpec(
    span=6.0,
    root_airfoil=root_5,
    tip_airfoil=tip_5,
    root_chord=2.5,
    tip_chord=1.2,
    sweep_le_deg=15,
    dihedral_deg=4,
    twist_deg=-2,
    num_sections=15,
))

X, Y, Z = wing_5.to_vertex_grids(num_points_profile=60)
triangles = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=True)
volume = VolumeCalculator.compute_solid_volume(triangles)
mass = volume * 2700.0
cg, inertia, _ = MassPropertiesCalculator.compute_all(X, Y, Z, mass)

print(f"  Volume: {volume:.6f} m^3")
print(f"  Mass:   {mass:.2f} kg")
print(f"  CG:     ({cg[0]:.4f}, {cg[1]:.4f}, {cg[2]:.4f}) m")

# ── Option B: Profile from .dat file ─────────────────────────────
print("\n=== Wing from .dat file ===")
dat_path = "/tmp/sample_airfoil.dat"
create_sample_dat_file(dat_path)
root_dat = AirfoilProfile.from_dat_file(dat_path, chord=1.0)
print(f"  Loaded profile: {root_dat.name}, {len(root_dat.x)} points")

# Build a simple tapered wing with the loaded profile
tip_dat = AirfoilProfile.from_naca4("0008", num_points=len(root_dat.x))

wing_dat = MultiSegmentWing(name="Custom Airfoil Wing")
wing_dat.add_segment(SegmentSpec(
    span=5.0,
    root_airfoil=root_dat,
    tip_airfoil=tip_dat,
    root_chord=2.0,
    tip_chord=1.0,
    sweep_le_deg=10,
    num_sections=12,
))

X2, Y2, Z2 = wing_dat.to_vertex_grids(num_points_profile=len(root_dat.x))
triangles2 = MeshTopologyManager.get_wing_triangles(X2, Y2, Z2, closed=True)
volume2 = VolumeCalculator.compute_solid_volume(triangles2)
mass2 = volume2 * 2700.0
cg2, inertia2, _ = MassPropertiesCalculator.compute_all(X2, Y2, Z2, mass2)

print(f"  Volume: {volume2:.6f} m^3")
print(f"  Mass:   {mass2:.2f} kg")
print(f"  CG:     ({cg2[0]:.4f}, {cg2[1]:.4f}, {cg2[2]:.4f}) m")

# ── Option C: Profile from raw points ────────────────────────────
print("\n=== Wing from raw coordinates ===")
theta = np.linspace(0, 2 * np.pi, 80)
# Elliptical profile
x_raw = 0.5 * (1.0 + np.cos(theta))
z_raw = 0.06 * np.sin(theta)
elliptic = AirfoilProfile.from_points(x_raw, z_raw, name="Elliptic", chord=1.0)

wing_e = MultiSegmentWing(name="Elliptic Profile Wing")
wing_e.add_segment(SegmentSpec(
    span=4.0,
    root_airfoil=elliptic,
    tip_airfoil=elliptic,
    root_chord=1.5,
    tip_chord=0.5,
    sweep_le_deg=25,
    num_sections=10,
))

X3, Y3, Z3 = wing_e.to_vertex_grids(num_points_profile=80)
triangles3 = MeshTopologyManager.get_wing_triangles(X3, Y3, Z3, closed=True)
volume3 = VolumeCalculator.compute_solid_volume(triangles3)
print(f"  Volume: {volume3:.6f} m^3")

# Clean up
os.remove(dat_path)

# Visualize the NACA 5-digit wing
show_interactive(triangles, volume, mass, cg, inertia, title=wing_5.name)

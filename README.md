# AeroShape: NURBS-Based Geometry Engine for Lifting Surfaces

AeroShape is an open-source Python package implementing the **GVM (Geometry, Volume, and Mass)** methodology for computing volume, mass, center of gravity, and moments of inertia of 3D lifting surfaces and wing-box structures — without requiring commercial CAD software.

Based on the paper:

> Valencia, E., Alulema, V., Hidalgo, V., Rodriguez, D. (2021). *A CAD-free methodology for volume and mass properties computation of 3-D lifting surfaces and wing-box structures.* Aerospace Science and Technology, 108, 106378.

---

## Features

- **NACA Profile Generation** — 4-digit NACA airfoil profiles with configurable point-distribution laws
- **3D Wing Mesh Construction** — Multi-segment wings with sweep, taper, dihedral, and twist
- **Volume Computation** — Divergence Theorem on triangulated surfaces (Eq. 2)
- **Thin-Shell Volume** — Offset (exact) and unfolding (approximate) approaches (Fig. 4)
- **Mass Distribution** — Chord-thickness based particle model (Eqs. 4–9)
- **Center of Mass & Inertia Tensor** — Full 6-DOF inertial properties (Eqs. 10–14)
- **Aircraft Assembly** — Multi-surface configurations via `AircraftModel`
- **CAD Export** — STEP, IGES, STL, and BREP formats (via OpenCASCADE / OCP)
- **Clustering Laws** — Uniform, cosine, tanh, exponential, Vinokur point distributions
- **Visualization** — Interactive 3D viewer (vedo/VTK) and static matplotlib figures
- **Interactive GUI** — Streamlit-based dashboard with 3D visualization

---

## Installation

```bash
# Core package (numpy only)
pip install -e .

# With GUI support (streamlit, plotly, pandas)
pip install -e ".[gui]"

# With CAD export (OCP / pythonocc)
pip install -e ".[export]"

# With visualization (vedo, matplotlib)
pip install -e ".[viz]"

# Everything
pip install -e ".[all]"
```

---

## Quick Start

### From a script

```python
from aeroshape import (
    AirfoilProfile,
    SegmentSpec,
    MultiSegmentWing,
    MeshTopologyManager,
    VolumeCalculator,
    MassPropertiesCalculator,
)

# 1. Define airfoil profiles
root = AirfoilProfile.from_naca4("2412", num_points=40)
tip  = AirfoilProfile.from_naca4("2412", num_points=40)

# 2. Build a wing
wing = MultiSegmentWing(name="NACA 2412 Wing")
wing.add_segment(SegmentSpec(
    span=10.0, root_airfoil=root, tip_airfoil=tip,
    root_chord=2.0, tip_chord=1.0, sweep_le_deg=15.0,
    num_sections=15,
))

# 3. Triangulate and compute volume
X, Y, Z = wing.to_vertex_grids(num_points_profile=40)
triangles = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=True)
volume = VolumeCalculator.compute_solid_volume(triangles)

# 4. Mass properties (aluminum, ρ = 2700 kg/m³)
mass = volume * 2700.0
cg, inertia, _ = MassPropertiesCalculator.compute_all(X, Y, Z, mass)

print(f"Volume: {volume:.6f} m³")
print(f"Mass:   {mass:.2f} kg")
print(f"CG:     X={cg[0]:.4f}, Y={cg[1]:.4f}, Z={cg[2]:.4f} m")
```

### NURBS Export

```python
from aeroshape import NurbsExporter

shape = wing.to_occ_shape()
NurbsExporter.to_step(shape, "Exports/my_wing.step")
NurbsExporter.to_iges(shape, "Exports/my_wing.iges")
NurbsExporter.to_stl(shape,  "Exports/my_wing.stl", linear_deflection=0.01)
NurbsExporter.to_brep(shape, "Exports/my_wing.brep")
```

All exported files are saved to the **`Exports/`** directory.

### Interactive GUI

```bash
streamlit run app.py
# or
python examples/run_gui.py
```

---

## Package Structure

```
aeroshape/
  __init__.py           # Top-level re-exports
  core/
    mesh.py             # MeshTopologyManager — structured triangulation
    volume.py           # VolumeCalculator — solid, shell-offset, shell-unfolding
    mass.py             # MassPropertiesCalculator — CG and inertia tensor
    clustering.py       # Point-distribution laws (uniform, cosine, tanh, …)
  geometry/
    airfoils.py         # AirfoilProfile, NACAProfileGenerator
    wings.py            # SegmentSpec, MultiSegmentWing
    aircraft.py         # AircraftModel — multi-surface assembly
  cad/
    surfaces.py         # NurbsSurfaceBuilder — NURBS lofting
    export.py           # NurbsExporter — STEP / IGES / STL / BREP
    utils.py            # OCC helpers (tessellate, sample, mass properties)
  vis/
    rendering.py        # show_interactive / show_static viewers
app.py                  # Streamlit GUI dashboard
examples/               # Runnable example scripts
Exports/                # Output directory for all exported CAD files
```

---

## Examples

Run any example from the project root:

```bash
python examples/<script>.py
```

| Script | Description |
|---|---|
| `basic_wing.py` | Solid wing: volume, mass, CG, inertia — exports `basic_wing.step` |
| `thin_shell_wing.py` | Thin-shell analysis: offset vs unfolding methods |
| `multi_segment_wing.py` | Cranked wing + winglet — exports `cranked_wing.step` |
| `blended_wing_body.py` | BWB configuration (4 segments) — exports `blended_wing_body.step` |
| `blended_wing_body_guided.py` | Guided BWB design — exports `blended_wing_body_guided.step` |
| `box_wing.py` | Prandtl-plane box-wing via `AircraftModel` — exports `box_wing.step` |
| `strut_braced_wing.py` | Strut-braced / truss-braced wing — exports `strut_braced_wing.step` |
| `clustering_laws.py` | Visualise point-distribution laws (matplotlib only, no OCC needed) |
| `custom_airfoil.py` | Wing built from custom airfoil coordinates |
| `compute_properties.py` | Standalone GVM property computation |
| `analytical_validation.py` | Unit-cube validation and convergence study (Sections 3.1.1–3.1.2) |
| `export_stl.py` | Export to STEP, IGES, STL, and BREP formats |
| `visualize_wing.py` | Interactive 3D viewer and static matplotlib figure |
| `run_gui.py` | Launch the Streamlit interactive dashboard |

All scripts that export CAD geometry write their output files to the `Exports/` directory.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

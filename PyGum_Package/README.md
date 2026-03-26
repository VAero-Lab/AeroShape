# AeroShape: CAD-Free Volume and Mass Properties for Lifting Surfaces

AeroShape is an open-source Python package implementing the **GVM (Geometry, Volume, and Mass)** methodology for computing volume, mass, center of gravity, and moments of inertia of 3D lifting surfaces and wing-box structures — without requiring commercial CAD software.



Based on the paper:

> Valencia, E., Alulema, V., Hidalgo, V., Rodriguez, D. (2021). *A CAD-free methodology for volume and mass properties computation of 3-D lifting surfaces and wing-box structures.* Aerospace Science and Technology, 108, 106378.

## Features

- **NACA Profile Generation** — 4-digit NACA airfoil profiles with cosine spacing
- **3D Wing Mesh Construction** — Structured vertex grids with sweep, taper, and twist
- **Volume Computation** — Divergence Theorem on triangulated surfaces (Eq. 2)
- **Thin-Shell Volume** — Offset (exact) and unfolding (approximate) approaches (Fig. 4)
- **Mass Distribution** — Chord-thickness based particle model (Eqs. 4-9)
- **Center of Mass & Inertia Tensor** — Full 6-DOF inertial properties (Eqs. 10-14)
- **CAD Export** — STL, IGES, and STEP formats
- **Visualization** — Interactive 3D viewer (vedo/VTK) and static matplotlib figures
- **Interactive GUI** — Streamlit-based dashboard with 3D visualization

## Installation

```bash
# Core package (numpy only)
pip install -e .

# With GUI support (streamlit, plotly, pandas)
pip install -e ".[gui]"

# With CAD export (gmsh)
pip install -e ".[export]"

# With visualization (vedo, matplotlib)
pip install -e ".[viz]"

# Everything
pip install -e ".[all]"
```

## Quick Start

### From a script

```python
from aeroshape import (
    WingMeshFactory,
    MeshTopologyManager,
    VolumeCalculator,
    MassPropertiesCalculator,
)

# Generate a wing mesh
X, Y, Z = WingMeshFactory.create(
    naca_root="2412", naca_tip="2412",
    semi_span=10.0, chord_root=2.0, chord_tip=1.0,
    sweep_angle_deg=15.0, num_points_profile=40, num_sections=15
)

# Triangulate and compute volume
triangles = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=True)
volume = VolumeCalculator.compute_solid_volume(triangles)

# Mass properties (aluminum, density=2700 kg/m3)
mass = volume * 2700.0
cg, inertia, mass_dist = MassPropertiesCalculator.compute_all(X, Y, Z, mass)

print(f"Volume: {volume:.6f} m3")
print(f"Mass: {mass:.2f} kg")
print(f"Center of mass: X={cg[0]:.4f}, Y={cg[1]:.4f}, Z={cg[2]:.4f}")
```

### Interactive GUI

```bash
streamlit run app.py
# or
python examples/run_gui.py
```

## Package Structure

```
aeroshape/
  __init__.py       # Package exports
  geometry.py       # NACA profiles and wing mesh generation
  mesh_utils.py     # Structured triangulation (Section 2.2.1)
  volume.py         # Volume computation: solid, shell offset, shell unfolding
  mass.py           # Mass distribution, center of mass, inertia tensor
  exporter.py       # STL/IGES/STEP export
  visualization.py  # Interactive 3D viewer (vedo) and static figures (matplotlib)
app.py              # Streamlit GUI dashboard
examples/           # Usage examples
```

## Examples

See the [examples/](examples/) directory for:

- `basic_wing.py` — Solid wing volume and mass computation
- `thin_shell_wing.py` — Thin-shell analysis comparing offset and unfolding approaches
- `analytical_validation.py` — Unit cube validation and convergence study (Sections 3.1.1-3.1.2)
- `export_stl.py` — Export wing to STL, IGES, and STEP formats
- `visualize_wing.py` — Interactive 3D viewer and static matplotlib figure
- `run_gui.py` — Launch the interactive dashboard

## License

MIT License. See [LICENSE](LICENSE) for details.

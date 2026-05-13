# AeroShape: Robust 3D Aircraft Geometry Modeling Framework

AeroShape is an open-source Python package designed as a general-purpose 3D aircraft geometry modeling framework. It provides robust parametric NURBS surface generation for lifting surfaces and implements the **GVM (Geometry, Volume, and Mass)** methodology for computing volume, mass, center of gravity, and moments of inertia, based on the paper:

> Valencia, E., Alulema, V., Hidalgo, V., Rodriguez, D. (2021). _A CAD-free methodology for volume and mass properties computation of 3-D lifting surfaces and wing-box structures._ Aerospace Science and Technology, 108, 106378.

AeroShape also supports volume and mass properties computation using the OpenCascade Library (OCC). While traditional GVM analysis is extremely fast for simplified meshes, our new **AeroShape Parallel OCC Engine** provides high-fidelity analysis that is now near-instantaneous (~100x speedup compared to legacy adaptive integration) and significantly more robust for complex, multi-body aircraft assemblies.

As the library continues to expand, AeroShape aims to become a fully comprehensive 3D geometry engine for all aircraft components (fuselages, nacelles, etc.), capable of generating geometry representations for various aerodynamic structural analyses, CAD export, and design optimization.

---

## Features

- **High-Performance Mass Properties** — Parallelized OpenCascade integration with up to 100x speedup.
- **Airfoil Generation** — 4 digit NACA airfoil with configurable point-distribution laws and airfoil from file.
- **3D Wing & Fuselage Construction** — Multi-segment wings and complex fuselages with automatic blending.
- **Loft Subdivision** — Automatic subdivision of large surfaces for robust parallel analysis and export.
- **Native Symmetry Handling** — Automatic mirroring of wings and stabilizers across the XZ plane.
- **Complex Assembly Aggregation** — Multi-body property integration (Volume, Mass, CG, Inertia) via `AircraftModel`.
- **Advanced Volume Computation** — Hybrid GVM/OCC methodology choosing between speed and fidelity.
- **Optimized CAD/Mesh Export** — Assembly-aware STEP (via XDE), IGES, STL, BREP, and compliant CGNS unstructured formats.
- **Interactive 3D Visualization** — VTK-based interactive CAD viewer with on-screen property annotations.

---

## Installation

AeroShape relies on OpenCascade (via `build123d` and `cadquery-ocp`) to perform complex NURBS lofting and boolean operations. Follow these steps to ensure a robust environment setup.

### 1. Prerequisites (OpenCascade)

The easiest way to install the required OCC backend is using `conda` or `mamba`.

```bash
# Create a new environment
conda create -n aeroshape-env python=3.10
conda activate aeroshape-env

# Install the OpenCascade backend (cadquery-ocp)
conda install -c conda-forge cadquery-ocp
```

### 2. Install AeroShape

Clone the repository and install the Python package.

```bash
git clone https://github.com/VAero-Lab/AeroShape.git
cd AeroShape

# Install the core package
pip install -e .

# (Optional) Install visualization tools (matplotlib)
pip install -e ".[viz]"

# (Optional) Install mesh export tools (pyCGNS)
pip install -e ".[mesh]"

# Install everything
pip install -e ".[all]"
```

---

## Quick Start

AeroShape makes it easy to build lifting surfaces, construct an aircraft assembly, and extract physical properties.

```python
from aeroshape import (
    AirfoilProfile,
    SegmentSpec,
    MultiSegmentWing,
    AircraftModel,
    show_interactive
)

# 1. Define airfoil profiles
root = AirfoilProfile.from_naca4("2412", num_points=40)
tip  = AirfoilProfile.from_naca4("2412", num_points=40)

# 2. Build a wing (starboard side)
wing = MultiSegmentWing(name="NACA 2412 Wing", symmetric=True)
wing.add_segment(SegmentSpec(
    span=10.0, root_airfoil=root, tip_airfoil=tip,
    root_chord=2.0, tip_chord=1.0, sweep_le_deg=15.0,
    num_sections=15,
))

# 3. Assemble and Analyze
ac = AircraftModel("Symmetric UAV")
ac.add_wing(wing, origin=(1.0, 0.0, 0.0))

# High-fidelity parallel OCC analysis (near-instantaneous)
props = ac.compute_properties(method='occ', density=2700.0, uproc=True, tolerance=0.1)

print(f"Volume: {props['volume']:.6f} m³")
print(f"Mass:   {props['mass']:.2f} kg")

# 4. Interactive 3D Visualization
if __name__ == "__main__":
    show_interactive(ac.to_triangles(), props['volume'], props['mass'],
                     props['cg'], props['inertia'], title=ac.name)
```

### CAD and Mesh Export

AeroShape natively exports to high-fidelity STEP files preserving assembly hierarchy, as well as computational meshes suitable for aerodynamics.

```python
from aeroshape.nurbs.export import NurbsExporter

# 1. Full symmetric assembly export to STEP
shape = ac.to_occ_shape()
NurbsExporter.to_step(shape, "Exports/full_model.step")

# 2. Export structured aerodynamic mesh to SIDS-compliant CGNS
ac.export_mesh_cgns("Exports/model_mesh.cgns", closed=True)
```

---

## Package Structure

```
aeroshape/
  analysis/
    mesh.py           # MeshTopologyManager — structured triangulation
    volume.py         # VolumeCalculator — GVM integration
    mass.py           # MassPropertiesCalculator — CG and inertia
  geometry/
    airfoils.py       # AirfoilProfile generation
    wings.py          # MultiSegmentWing & SegmentSpec
    fuselage.py       # Fuselage definitions & blending
    aircraft.py       # AircraftModel — Multi-body assembly (Master)
  nurbs/
    surfaces.py       # NurbsSurfaceBuilder — B-spline lofting
    export.py         # NurbsExporter — STEP/IGES/STL
    mesh_export.py    # MeshExporter — pyCGNS and binary STL formats
  visualization/
    rendering.py      # show_interactive / show_static
```

---

## Examples

Run any example from the project root:

```bash
python examples/<script>.py
```

### Prominent Examples

| Script                            | Description                                                                                                                                                     |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `aircraft_commercial_airliner.py` | Constructs a large twin-jet airliner with complex winglets, vertical stabilizers, and a tapered fuselage. Demonstrates managing complex multi-body hierarchies. |
| `aircraft_bwb_guided.py`          | Creates a Blended-Wing-Body using mathematical spline guide curves instead of discrete segments. Demonstrates perfectly smooth G1/C2 continuous swept lofts.    |
| `aircraft_box_wing.py`            | A Prandtl-plane box-wing aircraft featuring smooth G1-continuous vertical connecting fins between the upper and lower wings.                                    |
| `aircraft_military_cargo.py`      | High-wing cargo aircraft with a T-tail configuration and a heavy-lift blended fuselage.                                                                         |
| `export_mesh_demo.py`             | Demonstrates extracting structured triangulation grids and exporting them directly to unstructured **CGNS** and **STL** formats for CFD or FEM workflows.       |
| `analytical_validation.py`        | A convergence study validating the GVM methodology against known analytical solutions.                                                                          |

All scripts that export geometry will automatically write their output files to the `Exports/` directory in the root of the project.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

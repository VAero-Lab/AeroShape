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
- **Optimized CAD Export** — Assembly-aware STEP (via XDE), IGES, STL, and BREP formats.
- **Visualization** — Interactive high-fidelity 3D viewer (Plotly/build123d) and static figures.

---

## Installation

```bash
# Core package (Includes: build123d, plotly, numpy, scipy)
pip install -e .

# With GUI support (streamlit, pandas)
pip install -e ".[gui]"

# With extra visualization support (matplotlib)
pip install -e ".[viz]"

# All-in-one
pip install -e ".[all]"
```

---

## Quick Start

### Using AircraftModel

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

# High-fidelity analysis (100x faster than legacy OCC)
props = ac.compute_properties(method='occ', density=2700.0, uproc=True, tolerance=0.1)

print(f"Volume: {props['volume']:.6f} m³")
print(f"Mass:   {props['mass']:.2f} kg")

# 4. Interactive 3D Visualization
if __name__ == "__main__":
    show_interactive(ac.to_triangles(), props['volume'], props['mass'],
                     props['cg'], props['inertia'], title=ac.name)
```

### NURBS Export

AeroShape uses an assembly-aware XDE (Extended Data Exchange) writer for high-performance STEP export, preserving the logical structure of the aircraft.

```python
from aeroshape import NurbsExporter

# Full symmetric assembly export to STEP
shape = ac.to_occ_shape() 
NurbsExporter.to_step(shape, "Exports/full_model.step")
```

All exported files are saved to the **`Exports/`** directory.

### Interactive GUI

Launch the Streamlit dashboard for real-time parametric modeling and 3D visualization:

```bash
streamlit run app.py
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
  visualization/
    rendering.py      # show_interactive / show_static
```

---

## Examples

Run any example from the project root:

```bash
python examples/<script>.py
```

| Script                            | Description                                                            |
| --------------------------------- | ---------------------------------------------------------------------- |
| `aircraft_commercial_airliner.py` | Large twin-jet airliner with complex winglets and tapered fuselage     |
| `aircraft_bwb_guided.py`          | Blended-Wing-Body using spline guide curves and native symmetry        |
| `aircraft_box_wing.py`            | Prandtl-plane box-wing with smooth G1-continuous vertical fins         |
| `aircraft_military_cargo.py`      | High-wing cargo aircraft with T-tail and heavy-lift fuselage           |
| `aircraft_twin_boom.py`           | Twin-boom configuration demonstrating multi-body coordinate mapping    |
| `aircraft_uav_bwb.py`             | Flying-wing UAV with flattened stealth body and high-dihedral tips     |
| `aircraft_experimental_glider.py` | High-aspect ratio glider with extreme spanwise clustering requirements |
| `clustering_laws.py`              | Visualise point-distribution laws (uniform, cosine, tanh, Vinokur)     |
| `analytical_validation.py`        | Convergence study comparing GVM vs analytical solutions                |
| `run_gui.py`                      | Launch the Streamlit interactive dashboard                             |

All scripts that export CAD geometry write their output files to the `Exports/` directory.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

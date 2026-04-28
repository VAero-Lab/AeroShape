import os
import pytest
from aeroshape.geometry.airfoils import AirfoilProfile
from aeroshape.geometry.wings import MultiSegmentWing, SegmentSpec
from aeroshape.geometry.aircraft import AircraftModel
from aeroshape.nurbs.export import NurbsExporter

@pytest.fixture
def simple_aircraft():
    root = AirfoilProfile.from_naca4("0012", num_points=20)
    wing = MultiSegmentWing(name="Sym Wing", symmetric=True)
    wing.add_segment(SegmentSpec(
        span=2.0, root_airfoil=root, tip_airfoil=root,
        root_chord=1.0, tip_chord=1.0, num_sections=3
    ))
    
    model = AircraftModel("Test Assembly")
    model.add_wing(wing)
    return model

def test_step_export(simple_aircraft, tmp_path):
    """Test STEP export capability."""
    export_path = tmp_path / "test_model.step"
    
    shape = simple_aircraft.to_occ_shape()
    NurbsExporter.to_step(shape, str(export_path))
    
    assert export_path.exists()
    assert export_path.stat().st_size > 1000  # Ensure file is not empty

def test_cgns_export(simple_aircraft, tmp_path):
    """Test CGNS mesh export."""
    export_path = tmp_path / "test_mesh.cgns"
    
    simple_aircraft.export_mesh_cgns(str(export_path), closed=True)
    
    assert export_path.exists()
    
    # We can also verify it using pyCGNS
    import CGNS.MAP as MAP
    import CGNS.PAT.cgnslib as CGL
    
    # Check that we can open and read the Base
    try:
        tree = MAP.load(str(export_path))
        assert tree is not None
    except ImportError:
        # If pyCGNS is not installed, we skip the deep validation
        pass

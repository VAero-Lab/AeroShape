import pytest
import numpy as np
from aeroshape.geometry.airfoils import AirfoilProfile
from aeroshape.geometry.wings import MultiSegmentWing, SegmentSpec

@pytest.fixture
def base_airfoils():
    root = AirfoilProfile.from_naca4("0012", num_points=20)
    tip = AirfoilProfile.from_naca4("0012", num_points=20)
    return root, tip

def test_multisegment_wing_creation(base_airfoils):
    """Test basic wing creation and segment addition."""
    root, tip = base_airfoils
    wing = MultiSegmentWing(name="Test Wing", symmetric=False)
    
    seg = SegmentSpec(
        span=5.0,
        root_airfoil=root,
        tip_airfoil=tip,
        root_chord=2.0,
        tip_chord=1.0,
        sweep_le_deg=10,
        num_sections=5
    )
    wing.add_segment(seg)
    
    assert len(wing.segments) == 1
    assert sum(s.span for s in wing.segments) == 5.0
    
    # Check frames
    frames = wing.get_section_frames()
    assert len(frames) == 5
    
    # Root frame check
    assert frames[0]["y"] == 0.0
    assert frames[0]["chord"] == 2.0
    
    # Tip frame check
    assert frames[-1]["y"] == 5.0
    assert frames[-1]["chord"] == 1.0

def test_wing_symmetry(base_airfoils):
    """Test symmetry flag for wings."""
    root, tip = base_airfoils
    wing = MultiSegmentWing(name="Sym Wing", symmetric=True)
    
    seg = SegmentSpec(
        span=5.0, root_airfoil=root, tip_airfoil=tip,
        root_chord=2.0, tip_chord=1.0, num_sections=3
    )
    wing.add_segment(seg)
    
    # Even though it's symmetric, get_total_span() returns the half span for the definition
    # The AircraftModel handles the mirroring for properties.
    # However, let's just assert symmetric is True
    assert wing.symmetric is True
    assert sum(s.span for s in wing.segments) == 5.0

def test_occ_shape_generation(base_airfoils):
    """Test that the OCC BRep generator doesn't crash on a basic wing."""
    root, tip = base_airfoils
    wing = MultiSegmentWing(name="OCC Wing")
    wing.add_segment(SegmentSpec(
        span=2.0, root_airfoil=root, tip_airfoil=tip,
        root_chord=1.0, tip_chord=0.5, num_sections=3
    ))
    
    shape = wing.to_occ_shape()
    assert shape is not None
    # We can check it's a solid (which NurbsSurfaceBuilder defaults to)
    from build123d import Solid
    assert isinstance(shape, Solid)

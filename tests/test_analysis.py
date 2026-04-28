import pytest
import numpy as np
from aeroshape.geometry.airfoils import AirfoilProfile
from aeroshape.geometry.wings import MultiSegmentWing, SegmentSpec
from aeroshape.geometry.aircraft import AircraftModel

@pytest.fixture
def box_wing():
    """Create a simple untapered, unswept wing (like a box) for validation."""
    root = AirfoilProfile.from_naca4("0012", num_points=30) # symmetrical
    wing = MultiSegmentWing(name="Test Box")
    wing.add_segment(SegmentSpec(
        span=10.0,
        root_airfoil=root,
        tip_airfoil=root,
        root_chord=1.0,
        tip_chord=1.0,
        num_sections=5
    ))
    return wing

def test_gvm_volume(box_wing):
    """Test GVM volume calculation."""
    # Theoretical volume of an extruded NACA 0012 (area ~ 0.0822 * t * c^2)
    # Actually, let's just assert that GVM and OCC give similar results
    
    props_gvm = box_wing.compute_properties(method="gvm", density=1.0)
    props_occ = box_wing.compute_properties(method="occ", density=1.0, uproc=False)
    
    vol_gvm = props_gvm["volume"]
    vol_occ = props_occ["volume"]
    
    # GVM with 30 points per surface (so 60 points per section) might have a ~1-2% inscribed polygon error.
    # OCC is exact NURBS. So they should be within 3% of each other.
    assert np.isclose(vol_gvm, vol_occ, rtol=0.03)

def test_occ_parallel_vs_sequential(box_wing):
    """Test that AircraftModel parallel OCC matches sequential OCC."""
    model = AircraftModel("Test Assembly")
    model.add_wing(box_wing)
    
    props_seq = model.compute_properties(method="occ", density=1000.0, uproc=False, tolerance=0.1)
    props_par = model.compute_properties(method="occ", density=1000.0, uproc=True, tolerance=0.1)
    
    # Volumes should match perfectly
    assert np.isclose(props_seq["volume"], props_par["volume"], rtol=1e-5)
    
    # Masses should match
    assert np.isclose(props_seq["mass"], props_par["mass"], rtol=1e-5)
    
    # CG should match
    assert np.allclose(props_seq["cg"], props_par["cg"], atol=1e-3)

def test_symmetric_mass_properties():
    """Test that a symmetric wing doubles mass and sets CG Y to 0."""
    root = AirfoilProfile.from_naca4("0012", num_points=30)
    wing = MultiSegmentWing(name="Sym Wing", symmetric=True)
    wing.add_segment(SegmentSpec(
        span=5.0, root_airfoil=root, tip_airfoil=root,
        root_chord=1.0, tip_chord=1.0, num_sections=3
    ))
    
    model = AircraftModel("Symmetric Assembly")
    model.add_wing(wing)
    
    props = model.compute_properties(method="sai", density=1.0)
    
    # Since it's symmetric, the Y center of mass must be exactly 0
    cg_y = props["cg"][1]
    assert np.isclose(cg_y, 0.0, atol=1e-7)
    
    # Mass of symmetric should be exactly double the mass of the half-wing
    half_props = wing.compute_properties(method="sai", density=1.0)
    assert np.isclose(props["mass"], half_props["mass"] * 2.0, rtol=1e-5)

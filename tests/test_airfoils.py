import pytest
import numpy as np
from aeroshape.geometry.airfoils import AirfoilProfile, NACAProfileGenerator, NurbsProfile

def test_naca4_generation():
    """Test standard 4-digit NACA profile generation."""
    x, z = NACAProfileGenerator.generate("2412", num_points=50, chord=1.0)
    
    # 50 points per surface, but TE is combined, so 2 * 50 - 1 = 99 points
    assert len(x) == 99
    assert len(z) == 99
    
    # Check chord length scaling
    assert np.isclose(np.max(x) - np.min(x), 1.0)
    
    # Check TE closure (last point should roughly match first point in X)
    assert np.isclose(x[0], x[-1], atol=1e-5)

def test_airfoil_profile_factory():
    """Test the AirfoilProfile dataclass and its factory methods."""
    profile = AirfoilProfile.from_naca4("0012", num_points=30, chord=2.0)
    
    assert profile.name == "NACA 0012"
    assert profile.chord == 2.0
    
    # Check scaling method
    scaled_profile = profile.scaled(4.0)
    assert scaled_profile.chord == 4.0
    assert np.isclose(np.max(scaled_profile.x) - np.min(scaled_profile.x), 4.0)

def test_nurbs_profile():
    """Test NurbsProfile generation and scaling."""
    poles = np.array([
        [1.0, 0.0],
        [0.5, 0.1],
        [0.0, 0.0],
        [0.5, -0.1],
        [1.0, 0.0]
    ])
    knots = [0.0, 0.3333, 0.6666, 1.0]
    mults = [3, 1, 1, 3]
    
    profile = NurbsProfile(
        poles=poles,
        knots=knots,
        multiplicities=mults,
        degree=2,
        name="Test NURBS",
        chord=1.0,
        num_eval_points=50
    )
    
    assert profile.name == "Test NURBS"
    assert len(profile.x) == 50
    assert len(profile.z) == 50
    
    scaled = profile.scaled(2.0)
    assert scaled.chord == 2.0
    # Check that poles scaled
    assert np.isclose(scaled.poles[0, 0], 2.0)

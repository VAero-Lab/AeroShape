"""Geometric component definitions: airfoils, wings, aircraft assembly."""

from aeroshape.geometry.airfoils import AirfoilProfile, NACAProfileGenerator
from aeroshape.geometry.wings import SegmentSpec, MultiSegmentWing
from aeroshape.geometry.aircraft import AircraftModel

__all__ = [
    "AirfoilProfile",
    "NACAProfileGenerator",
    "SegmentSpec",
    "MultiSegmentWing",
    "AircraftModel",
]

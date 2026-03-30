"""Aircraft assembly model.

Provides AircraftModel, which holds multiple lifting surfaces (wings,
tails, canards, struts, etc.) and combines them for analysis and CAD
export.  In the future, fuselages, nacelles, and propellers will also
be supported as component types.

Coordinate convention:
    X = chordwise, Y = spanwise, Z = thickness.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class AircraftModel:
    """Multi-body aircraft assembly.

    Holds multiple lifting surfaces (wings, tails, canards, struts)
    and combines them into a single OCC compound shape for export
    and analysis.

    Each surface is a MultiSegmentWing with a 3D placement offset.

    Attributes
    ----------
    name : str
        Aircraft/configuration name.
    surfaces : list of dict
        Each dict: {"wing": MultiSegmentWing, "origin": (x, y, z)}.
        origin translates the entire wing to its mounting position.
    """
    name: str = "aircraft"
    surfaces: List[dict] = field(default_factory=list)

    def add_surface(self, wing, origin=(0.0, 0.0, 0.0)):
        """Add a lifting surface to the model.

        Parameters
        ----------
        wing : MultiSegmentWing
            The lifting surface definition.
        origin : tuple of float
            (x, y, z) mounting position offset applied to the entire surface.

        Returns
        -------
        self
            For chaining.
        """
        self.surfaces.append({"wing": wing, "origin": origin})
        return self

    def to_occ_shape(self, fuse=False):
        """Build an OCC shape from all surfaces.

        Parameters
        ----------
        fuse : bool
            If True, boolean-fuse all bodies into one solid.
            If False (default), create a compound with separate bodies.

        Returns
        -------
        TopoDS_Shape
            Compound or fused shape.
        """
        from OCP.gp import gp_Vec, gp_Trsf
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        from aeroshape.nurbs.surfaces import NurbsSurfaceBuilder

        shapes = []
        for entry in self.surfaces:
            wing = entry["wing"]
            ox, oy, oz = entry["origin"]

            shape = wing.to_occ_shape()

            # Apply origin translation if non-zero
            if abs(ox) > 1e-10 or abs(oy) > 1e-10 or abs(oz) > 1e-10:
                trsf = gp_Trsf()
                trsf.SetTranslation(gp_Vec(ox, oy, oz))
                shape = BRepBuilderAPI_Transform(shape, trsf, True).Shape()

            shapes.append(shape)

        if fuse:
            return NurbsSurfaceBuilder.fuse_shapes(shapes)
        else:
            return NurbsSurfaceBuilder.make_compound(shapes)

    def to_vertex_grids_list(self, num_points_profile=50,
                             spanwise_clustering=None,
                             chordwise_clustering=None):
        """Get vertex grids for each surface separately.

        Parameters
        ----------
        num_points_profile : int
            Chordwise resolution.
        spanwise_clustering : callable or None
            Spanwise distribution law.
        chordwise_clustering : callable or None
            Chordwise distribution law.

        Returns
        -------
        list of tuple
            Each element is (X, Y, Z, name) for one surface.
        """
        result = []
        for entry in self.surfaces:
            wing = entry["wing"]
            ox, oy, oz = entry["origin"]
            X, Y, Z = wing.to_vertex_grids(num_points_profile,
                                             spanwise_clustering,
                                             chordwise_clustering)
            X = X + ox
            Y = Y + oy
            Z = Z + oz
            result.append((X, Y, Z, wing.name))
        return result

    def to_triangles(self, num_points_profile=50, closed=True,
                     spanwise_clustering=None, chordwise_clustering=None):
        """Get combined triangle list from all surfaces.

        Parameters
        ----------
        num_points_profile : int
            Chordwise resolution.
        closed : bool
            If True, add end-cap triangles for watertight meshes.
        spanwise_clustering : callable or None
            Spanwise distribution law.
        chordwise_clustering : callable or None
            Chordwise distribution law.

        Returns
        -------
        triangles : list of tuple
            Combined triangle list from all surfaces.
        """
        from aeroshape.analysis.mesh import MeshTopologyManager

        all_triangles = []
        for entry in self.surfaces:
            wing = entry["wing"]
            ox, oy, oz = entry["origin"]
            X, Y, Z = wing.to_vertex_grids(num_points_profile,
                                             spanwise_clustering,
                                             chordwise_clustering)
            X = X + ox
            Y = Y + oy
            Z = Z + oz
            tris = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=closed)
            all_triangles.extend(tris)
        return all_triangles

    def compute_properties(self, method="gvm", density=1.0,
                           num_points_profile=50,
                           spanwise_clustering=None,
                           chordwise_clustering=None):
        """Compute volume, mass, CG, and inertia for the full aircraft.

        Parameters
        ----------
        method : str
            ``'gvm'`` — GVM Divergence-Theorem on triangulated mesh.
            ``'sai'`` — Section-Area Integration (shoelace + trapezoidal).
            ``'occ'`` — OCC BRepGProp on the exact NURBS surface.
        density : float
            Material density in kg/m^3.
        num_points_profile : int
            Chordwise resolution (GVM/SAI only).
        spanwise_clustering : callable or None
            Spanwise distribution law (GVM/SAI only).
        chordwise_clustering : callable or None
            Chordwise distribution law (GVM/SAI only).

        Returns
        -------
        dict
            Keys: ``volume``, ``mass``, ``cg`` (3-array), ``inertia`` (6-tuple).
        """
        method = method.lower()
        if method == "occ":
            from aeroshape.nurbs.utils import occ_mass_properties
            shape = self.to_occ_shape(fuse=False)
            props = occ_mass_properties(shape, density)
            com = props["center_of_mass"]
            imat = props["inertia_matrix"]
            return {
                "volume": props["volume"],
                "mass": props["mass"],
                "cg": np.array(com),
                "inertia": (imat[0, 0], imat[1, 1], imat[2, 2],
                            imat[0, 1], imat[0, 2], imat[1, 2]),
            }

        from aeroshape.analysis.volume import VolumeCalculator
        from aeroshape.analysis.mass import MassPropertiesCalculator

        grids = self.to_vertex_grids_list(num_points_profile,
                                           spanwise_clustering,
                                           chordwise_clustering)

        if method == "sai":
            volume = sum(
                VolumeCalculator.compute_solid_volume_sai(X, Y, Z)
                for X, Y, Z, _ in grids
            )
        else:
            # GVM Divergence-Theorem on NURBS-sampled geometry
            from aeroshape.analysis.mesh import MeshTopologyManager

            all_triangles = []
            for X, Y, Z, _ in grids:
                tris = MeshTopologyManager.get_wing_triangles(
                    X, Y, Z, closed=True)
                all_triangles.extend(tris)

            volume = VolumeCalculator.compute_solid_volume(all_triangles)

        all_X = np.vstack([g[0] for g in grids])
        all_Y = np.vstack([g[1] for g in grids])
        all_Z = np.vstack([g[2] for g in grids])

        mass = volume * density
        cg, inertia, _ = MassPropertiesCalculator.compute_all(
            all_X, all_Y, all_Z, mass)
        return {
            "volume": volume,
            "mass": mass,
            "cg": cg,
            "inertia": inertia,
        }

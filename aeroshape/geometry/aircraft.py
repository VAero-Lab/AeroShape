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

    Holds multiple lifting surfaces (wings, tails) and fuselages,
    combining them into a single OCC compound shape for export
    and analysis.

    Attributes
    ----------
    name : str
        Aircraft/configuration name.
    surfaces : list of dict
        Each dict: {"wing": MultiSegmentWing, "origin": (x, y, z)}.
        origin translates the entire wing to its mounting position.
    fuselages : list of dict
        Each dict: {"fuselage": MultiSegmentFuselage, "origin": (x, y, z)}.
    """
    name: str = "aircraft"
    surfaces: List[dict] = field(default_factory=list)
    fuselages: List[dict] = field(default_factory=list)

    def add_surface(self, wing, origin=(0.0, 0.0, 0.0)):
        """Add a lifting surface (backwards compatibility)."""
        return self.add_wing(wing, origin)

    def add_wing(self, wing, origin=(0.0, 0.0, 0.0)):
        """Add a lifting surface to the model.

        Parameters
        ----------
        wing : MultiSegmentWing
            The lifting surface definition.
        origin : tuple of float
            (x, y, z) mounting position offset applied to the surface.

        Returns
        -------
        self
        """
        self.surfaces.append({"wing": wing, "origin": origin})
        return self

    def add_fuselage(self, fuselage, origin=(0.0, 0.0, 0.0)):
        """Add a fuselage to the model.

        Parameters
        ----------
        fuselage : MultiSegmentFuselage
            The fuselage definition.
        origin : tuple of float
            (x, y, z) mounting position offset applied to the fuselage.

        Returns
        -------
        self
        """
        self.fuselages.append({"fuselage": fuselage, "origin": origin})
        return self

    def to_occ_shape(self, fuse=False):
        """Build an OCC shape from all aircraft components.

        Parameters
        ----------
        fuse : bool
            If True, boolean-fuse all bodies into one solid.
            If False (default), create a compound with separate bodies.

        Returns
        -------
        TopoDS_Shape
        """
        from OCP.gp import gp_Vec, gp_Trsf, gp_Ax2, gp_Pnt, gp_Dir
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        from aeroshape.nurbs.surfaces import NurbsSurfaceBuilder

        shapes = []
        
        # Process Wings
        for entry in self.surfaces:
            wing = entry["wing"]
            ox, oy, oz = entry["origin"]
            shape = wing.to_occ_shape()

            if abs(ox) > 1e-10 or abs(oy) > 1e-10 or abs(oz) > 1e-10:
                trsf = gp_Trsf()
                trsf.SetTranslation(gp_Vec(ox, oy, oz))
                shape_trans = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
                shapes.append(shape_trans)
            else:
                shapes.append(shape)
                
            if wing.symmetric:
                # Mirror across the local XZ plane (Y=0), then apply offset
                mirror_trsf = gp_Trsf()
                mirror_trsf.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)))
                mirrored = BRepBuilderAPI_Transform(shape, mirror_trsf, True).Shape()
                
                if abs(ox) > 1e-10 or abs(oy) > 1e-10 or abs(oz) > 1e-10:
                    trsf2 = gp_Trsf()
                    # Apply translation mirroring the Y offset naturally
                    trsf2.SetTranslation(gp_Vec(ox, -oy, oz))
                    mirrored = BRepBuilderAPI_Transform(mirrored, trsf2, True).Shape()
                shapes.append(mirrored)

        # Process Fuselages
        for entry in self.fuselages:
            fuselage = entry["fuselage"]
            ox, oy, oz = entry["origin"]
            shape = fuselage.to_occ_shape()

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
        """Get vertex grids for each surface and fuselage separately.

        Returns
        -------
        list of tuple
            Each element is (X, Y, Z, name, is_mirrored) for one component.
        """
        result = []
        
        # Process Wings
        for entry in self.surfaces:
            wing = entry["wing"]
            ox, oy, oz = entry["origin"]
            X, Y, Z = wing.to_vertex_grids(num_points_profile,
                                             spanwise_clustering,
                                             chordwise_clustering)
            result.append((X + ox, Y + oy, Z + oz, wing.name, False))
            
            if wing.symmetric:
                result.append((X + ox, -Y - oy, Z + oz, wing.name + "_mirrored", True))

        # Process Fuselages
        for entry in self.fuselages:
            fuselage = entry["fuselage"]
            ox, oy, oz = entry["origin"]
            # For fuselages, spanwise roughly translates to lengthwise
            X, Y, Z = fuselage.to_vertex_grids(num_points_profile,
                                             spanwise_clustering,
                                             chordwise_clustering)
            result.append((X + ox, Y + oy, Z + oz, fuselage.name, False))
            
        return result

    def to_triangles(self, num_points_profile=80, closed=True,
                     spanwise_clustering=None, chordwise_clustering=None):
        """Get combined triangle list from all components natively inheriting symmetry."""
        from aeroshape.analysis.mesh import MeshTopologyManager

        all_triangles = []
        
        # We explicitly inherit all physically modeled components (with mirrored boundaries) over numpy matrices identically and convert to closed meshes
        grids = self.to_vertex_grids_list(
            num_points_profile=num_points_profile,
            spanwise_clustering=spanwise_clustering,
            chordwise_clustering=chordwise_clustering
        )
        
        for (X, Y, Z, name, is_mirrored) in grids:
            tris = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=closed)
            if is_mirrored:
                # Reverse winding to keep outward normals on the mirrored solid
                tris = [(A, C, B) for (A, B, C) in tris]
            all_triangles.extend(tris)
            
        return all_triangles

    def compute_properties(self, method="gvm", density=1.0,
                           num_points_profile=50,
                           spanwise_clustering=None,
                           chordwise_clustering=None):
        """Compute volume, mass, CG, and inertia for the full aircraft."""
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
        from aeroshape.analysis.mesh import MeshTopologyManager

        grids = self.to_vertex_grids_list(
            num_points_profile=num_points_profile,
            spanwise_clustering=spanwise_clustering,
            chordwise_clustering=chordwise_clustering
        )

        total_vol = 0.0
        sum_m_r = np.zeros(3)
        sum_I = np.zeros(6) # Ixx, Iyy, Izz, Ixy, Ixz, Iyz

        for X, Y, Z, name, is_mirrored in grids:
            # 1. Compute triangles for this specific grid
            tris = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=True)
            if is_mirrored:
                tris = [(A, C, B) for (A, B, C) in tris]
            
            # 2. Volume via GVM
            v_comp = VolumeCalculator.compute_solid_volume(tris)
            m_comp = v_comp * density
            
            # 3. Mass properties for this component (at global origin)
            # compute_all returns (CG, Inertia_at_origin, distribution)
            cg_comp, i_comp, _ = MassPropertiesCalculator.compute_all(X, Y, Z, m_comp)
            
            total_vol += v_comp
            sum_m_r += np.array(cg_comp) * m_comp
            sum_I += np.array(i_comp)

        total_mass = total_vol * density
        final_cg = sum_m_r / total_mass if total_mass > 1e-9 else np.zeros(3)
        final_i = tuple(sum_I)

        return {
            "volume": total_vol,
            "mass": total_mass,
            "cg": final_cg,
            "inertia": final_i,
        }

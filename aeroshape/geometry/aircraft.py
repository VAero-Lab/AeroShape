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
import math


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
            # build123d objects need to be unwrapped for OCP calls
            occ_shape = shape.wrapped if hasattr(shape, "wrapped") else shape

            if abs(ox) > 1e-10 or abs(oy) > 1e-10 or abs(oz) > 1e-10:
                trsf = gp_Trsf()
                trsf.SetTranslation(gp_Vec(ox, oy, oz))
                # copy=True to ensure independent geometry for export robustness
                shape_trans = BRepBuilderAPI_Transform(occ_shape, trsf, True).Shape()
                shapes.append(shape_trans)
            else:
                shapes.append(occ_shape)
                
            if wing.symmetric:
                # Mirror across the local XZ plane (Y=0), then apply offset
                mirror_trsf = gp_Trsf()
                mirror_trsf.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)))
                mirrored = BRepBuilderAPI_Transform(occ_shape, mirror_trsf, True).Shape()
                
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
            occ_shape = shape.wrapped if hasattr(shape, "wrapped") else shape

            if abs(ox) > 1e-10 or abs(oy) > 1e-10 or abs(oz) > 1e-10:
                trsf = gp_Trsf()
                trsf.SetTranslation(gp_Vec(ox, oy, oz))
                occ_shape = BRepBuilderAPI_Transform(occ_shape, trsf, True).Shape()
            shapes.append(occ_shape)

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
                           chordwise_clustering=None,
                           uproc=True,
                           tolerance=1e-4):
        """Compute volume, mass, CG, and inertia for the full aircraft.

        Parameters
        ----------
        method : str
            "gvm" (fast mesh-based) or "occ" (exact NURBS integration).
        density : float
            Material density in kg/m^3.
        num_points_profile : int
            Grid resolution (for GVM only).
        uproc : bool
            If True and method="occ", use multiprocessing for acceleration.
        tolerance : float
            Relative error tolerance for OCC integration.
        """
        method = method.lower()
        if method == "occ":
            if uproc:
                return self._compute_properties_occ_parallel(density, tolerance)
            else:
                from aeroshape.nurbs.utils import occ_mass_properties
                shape = self.to_occ_shape(fuse=False)
                props = occ_mass_properties(shape, density, tolerance=tolerance)
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

    def _compute_properties_occ_parallel(self, density, tolerance):
        """Compute OCC properties by distributing individual segments to a process pool."""
        import multiprocessing as mp
        import numpy as np

        tasks = []
        # Each task: (component_type, definition, segment_index, origin, is_mirrored, symmetric_wing)
        
        # Wings
        for entry in self.surfaces:
            wing = entry['wing']
            origin = entry['origin']
            n_total = len(wing.get_section_frames())
            # Match chunking logic in to_occ_segments (max_sections=15)
            if n_total < 2:
                num_tasks = 0
            else:
                num_tasks = math.ceil((n_total - 1) / (15 - 1))
                
            # Add starboard segments
            for i in range(num_tasks):
                tasks.append(('wing_seg', wing, i, origin, False))
            # Add port segments if symmetric
            if wing.symmetric:
                for i in range(num_tasks):
                    tasks.append(('wing_seg', wing, i, origin, True))

        # Fuselages
        for entry in self.fuselages:
            fuse = entry['fuselage']
            origin = entry['origin']
            n_total = len(fuse.get_section_frames())
            if n_total < 2:
                num_tasks = 0
            else:
                num_tasks = math.ceil((n_total - 1) / (15 - 1))
                
            for i in range(num_tasks):
                tasks.append(('fuse_seg', fuse, i, origin, False))

        if not tasks:
            return {"volume": 0.0, "mass": 0.0, "cg": np.zeros(3), "inertia": (0,0,0,0,0,0)}

        # Use Pool to compute segment properties
        with mp.Pool(processes=min(len(tasks), mp.cpu_count())) as pool:
            results = pool.starmap(_worker_compute_segment_props, [(t, density, tolerance) for t in tasks])

        # Aggregate results
        total_vol = 0.0
        total_mass = 0.0
        sum_m_r = np.zeros(3)
        sum_I = np.zeros((3, 3))

        for res in results:
            total_vol += res['volume']
            total_mass += res['mass']
            sum_m_r += np.array(res['center_of_mass']) * res['mass']
            sum_I += res['inertia_matrix']

        final_cg = sum_m_r / total_mass if total_mass > 1e-9 else np.zeros(3)
        
        return {
            "volume": total_vol,
            "mass": total_mass,
            "cg": final_cg,
            "inertia": (sum_I[0, 0], sum_I[1, 1], sum_I[2, 2],
                        sum_I[0, 1], sum_I[0, 2], sum_I[1, 2]),
        }
    def show(self, method='occ', uproc=True, tolerance=0.1, props=None, **kwargs):
        """Launch the high-fidelity interactive CAD viewer.
        
        This is the recommended way to inspect the aircraft. It uses 
        OpenCASCADE B-Rep data directly to produce a professional 
        CAD-like visualization with a navigation cube.
        
        Parameters
        ----------
        method : str
            Analysis method used to compute properties for the annotation box.
        uproc : bool
            Enable parallel analysis.
        tolerance : float
            Integration tolerance for OCC analysis.
        props : dict or None
            Manual properties dictionary (volume, mass, cg, inertia).
        **kwargs : dict
            Additional arguments passed to `show_interactive` (color, opacity, etc.)
        """
        from aeroshape.visualization.rendering import show_interactive
        
        # 1. Compute properties for the on-screen display if not provided
        if props is None:
            props = self.compute_properties(method=method, uproc=uproc, tolerance=tolerance)
        
        # 2. Get NURBS sampling grids (for the original "lattice" look)
        grids = self.to_vertex_grids_list(num_points_profile=80)
        
        # 3. Launch viewer
        show_interactive(
            grids, 
            props['volume'], 
            props['mass'], 
            props['cg'], 
            props['inertia'],
            title=kwargs.pop('title', self.name),
            **kwargs
        )

def _worker_compute_segment_props(task, density, tolerance):
    """Worker function to build a single segment and compute its properties."""
    import numpy as np
    from OCP.gp import gp_Trsf, gp_Vec, gp_Ax2, gp_Pnt, gp_Dir
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    
    comp_type, definition, seg_idx, origin, is_mirrored = task
    ox, oy, oz = origin
    
    # 1. Build segment shape
    # We use to_occ_segments which we just implemented
    segs = definition.to_occ_segments()
    if seg_idx >= len(segs):
         return {"volume": 0.0, "mass": 0.0, "center_of_mass": (0,0,0), "inertia_matrix": np.zeros((3,3))}
         
    shape = segs[seg_idx]
    occ_shape = shape.wrapped if hasattr(shape, "wrapped") else shape

    # 2. Transform to assembly position
    trsf = gp_Trsf()
    
    if comp_type == 'wing_seg' and is_mirrored:
        # Mirror across Y=0 (local)
        mirror_trsf = gp_Trsf()
        mirror_trsf.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)))
        occ_shape = BRepBuilderAPI_Transform(occ_shape, mirror_trsf, False).Shape()
        # Apply mirrored translation
        trsf.SetTranslation(gp_Vec(ox, -oy, oz))
    else:
        # Apply standard translation
        trsf.SetTranslation(gp_Vec(ox, oy, oz))
        
    occ_final = BRepBuilderAPI_Transform(occ_shape, trsf, False).Shape()

    # 3. Compute properties
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(occ_final, props, tolerance, False)
    
    volume = float(props.Mass())
    cg = props.CentreOfMass()
    mat = props.MatrixOfInertia()
    
    inertia = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            inertia[i, j] = mat.Value(i+1, j+1)

    return {
        "volume": abs(volume),
        "mass": abs(volume) * density,
        "center_of_mass": (cg.X(), cg.Y(), cg.Z()),
        "inertia_matrix": inertia * density,
    }

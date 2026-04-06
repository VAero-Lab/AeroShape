"""Visualization for lifting surface geometry and computed properties.

Provides two rendering backends:

- **Matplotlib** (static): Publication-quality multi-view figures for papers
  and reports.  Four orthographic projections (front, top, side, isometric)
  with a properties annotation box.

- **Vedo / VTK** (interactive): Professional CAD-like 3D viewer with navigation cube, 
  rotate, zoom, and pan. Renders the surface with realistic lighting, smooth 
  NURBS-derived edges, and an on-screen properties panel.

Both backends share a common data interface: they accept the triangle list
produced by MeshTopologyManager together with the scalar results from
VolumeCalculator and MassPropertiesCalculator.
"""

import numpy as np


# ── Shared helpers ────────────────────────────────────────────────────

def _triangles_to_arrays(triangles):
    """Convert list-of-tuple triangles to vertex and face arrays.

    Returns
    -------
    vertices : np.ndarray, shape (N, 3)
    faces : np.ndarray, shape (M, 3)
    """
    verts = []
    faces = []
    for A, B, C in triangles:
        idx = len(verts)
        verts.extend([A, B, C])
        faces.append([idx, idx + 1, idx + 2])
    return np.array(verts, dtype=float), np.array(faces, dtype=int)


def _build_props_text(volume, mass, center_of_mass, inertia):
    """Build a multi-line string summarising the computed properties."""
    cx, cy, cz = center_of_mass
    lines = [
        f"Volume   {volume:.6f} m\u00b3",
        f"Mass     {mass:.4f} kg",
        "",
        f"CG_x     {cx:+.5f} m",
        f"CG_y     {cy:+.5f} m",
        f"CG_z     {cz:+.5f} m",
    ]
    if inertia is not None:
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = inertia
        lines += [
            "",
            f"Ixx      {Ixx:+.4f} kg\u00b7m\u00b2",
            f"Iyy      {Iyy:+.4f} kg\u00b7m\u00b2",
            f"Izz      {Izz:+.4f} kg\u00b7m\u00b2",
            f"Ixy      {Ixy:+.4f} kg\u00b7m\u00b2",
            f"Ixz      {Ixz:+.4f} kg\u00b7m\u00b2",
            f"Iyz      {Iyz:+.4f} kg\u00b7m\u00b2",
        ]
    return "\n".join(lines)


def _configure_view(ax, all_pts, elev, azim):
    """Set tight limits with proportional box aspect.

    The box shape matches the data extents so the geometry fills the
    subplot instead of floating inside an oversized cube.  Very thin
    dimensions are clamped to a minimum fraction to stay visible.
    """
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    ranges = maxs - mins
    pad = ranges.max() * 0.02

    ax.set_xlim(mins[0] - pad, maxs[0] + pad)
    ax.set_ylim(mins[1] - pad, maxs[1] + pad)
    ax.set_zlim(mins[2] - pad, maxs[2] + pad)

    # Box proportional to data, thin dims boosted to >= 8 % of max
    r = ranges + 2 * pad
    floor = r.max() * 0.08
    ax.set_box_aspect([max(r[0], floor), max(r[1], floor), max(r[2], floor)])
    ax.view_init(elev=elev, azim=azim)


# =====================================================================
#  Interactive 3-D viewer  (Plotly)
# =====================================================================

def show_interactive(
    triangles,
    volume,
    mass,
    center_of_mass,
    inertia=None,
    title="AeroShape CAD Viewer",
    surface_color="#ACB5BD",
    surface_opacity=1,
    cg_color="red",
    window_size=(1200, 900),
    background="white",
    save_screenshot=None,
    backend="vedo"
):
    """Launch a high-fidelity interactive CAD viewer.

    Standard backend is 'vedo' (VTK-based), which provides a professional
    CAD look with navigation cube, perspective views, and smooth rendering.

    Parameters
    ----------
    triangles : list of tuple or build123d.Shape
        Tessellated exterior boundaries or a build123d/OCP Shape.
    volume, mass : float
        Computed volume and mass.
    center_of_mass : tuple of float
        (CG_x, CG_y, CG_z).
    inertia : tuple of float or None
        (Ixx, Iyy, Izz, Ixy, Ixz, Iyz).
    title : str
        Viewer title.
    surface_color : str
        Surface color (hex or name).
    surface_opacity : float
        Surface opacity (0-1).
    cg_color : str
        Colour for the centre-of-mass marker.
    window_size : tuple of int
        Size of the window.
    background : str
        Background colour.
    save_screenshot : str or None
        Path to save a PNG screenshot.
    backend : str
        'vedo' (default) or 'ocp_vscode' (if available).
    """
    if backend == "ocp_vscode":
        try:
            from ocp_vscode import show
            # Try to pass the shape directly if it's there
            if hasattr(triangles, "wrapped") or str(type(triangles)).find("TopoDS_Shape") != -1:
                show(triangles, names=[title], collapse="all")
                return
        except ImportError:
            pass

    import vedo

    objs = []
    
    # 1. Prepare Mesh (Grid-based or Triangle-based)
    if isinstance(triangles, list) and len(triangles) > 0 and isinstance(triangles[0], tuple) and len(triangles[0]) == 5:
        # It's a list of (X, Y, Z, name, is_mirrored) grids - original NURBS sampling look!
        for X, Y, Z, name, is_mirrored in triangles:
            from aeroshape.analysis.mesh import MeshTopologyManager
            tris = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=True)
            if is_mirrored:
                tris = [(A, C, B) for (A, B, C) in tris]
            
            verts, faces = _triangles_to_arrays(tris)
            surf = vedo.Mesh([verts, faces])
            surf.c(surface_color).alpha(surface_opacity)
            surf.compute_normals().lighting("plastic")
            objs.append(surf)

            # # Create lattice (wireframe)
            # lattice = surf.clone().wireframe().c("black").alpha(0)
            # objs.append(lattice)

    elif hasattr(triangles, "wrapped") or str(type(triangles)).find("TopoDS_Shape") != -1:
        # It's a build123d/OCP object
        from aeroshape.nurbs.utils import tessellate_shape
        occ_shape = triangles.wrapped if hasattr(triangles, "wrapped") else triangles
        tris = tessellate_shape(occ_shape, linear_deflection=0.005, angular_deflection=0.1)
        verts, faces = _triangles_to_arrays(tris)
        
        mesh = vedo.Mesh([verts, faces])
        mesh.c(surface_color).alpha(surface_opacity)
        mesh.compute_normals().lighting("plastic")
        
        # B-rep edges
        from build123d import Shape
        b123_shape = Shape(occ_shape)
        lines = []
        for edge in b123_shape.edges():
            pts = edge.tessellate(1e-3)
            line_pts = [[p.X, p.Y, p.Z] for p in pts]
            lines.append(vedo.Line(line_pts, c="black", lw=1))
        
        objs = [mesh] + lines
    else:
        # raw triangles
        verts, faces = _triangles_to_arrays(triangles)
        mesh = vedo.Mesh([verts, faces])
        mesh.c(surface_color).alpha(surface_opacity)
        mesh.compute_normals().lighting("plastic")
        objs = [mesh]

    # 2. Add CG Marker
    cx, cy, cz = center_of_mass
    cg_sphere = vedo.Sphere(pos=[cx, cy, cz], r=0.25, c=cg_color)
    diag = 1.0
    meshes = [o for o in objs if isinstance(o, (vedo.Mesh, vedo.Line, vedo.Points))]
    if meshes:
        # Use first mesh for diagonal if available
        diag = max(m.diagonal_size() if hasattr(m, 'diagonal_size') else 1.0 for m in meshes)
    # cg_sphere.scale(diag * 0.1)
    # cg_sphere.name = "Center of Mass"
    # objs.append(cg_sphere)

    # 3. Create Plotter
    plt = vedo.Plotter(
        title=title, 
        size=window_size, 
        bg=background, 
        axes=14,          # Orientation widget (triad/cube)
        interactive=True
    )


    # 5. Properties Text Box
    props_text = _build_props_text(volume, mass, center_of_mass, inertia)
    props_actor = vedo.Text2D(props_text, pos="bottom-left", s=0.7, c="black", bg="white", alpha=0.8)
    objs.append(props_actor)

    # 6. Show everything
    plt.show(
        objs,
        camera={
            'pos': [cx - diag*3, cy + diag*0.5, cz + diag*0.5],
            'focal_point': [cx, cy, cz],
            'viewup': [0, 0, 1],
        },
        interactive=True
    ).close()


# =====================================================================
#  Static matplotlib figure  (multi-view)
# =====================================================================

def show_static(
    triangles,
    volume,
    mass,
    center_of_mass,
    inertia=None,
    title="AeroShape",
    surface_color="#DAA520",
    edge_color="#8B7520",
    surface_opacity=0.85,
    cg_color="red",
    figsize=(14, 10),
    dpi=150,
    save_path=None,
):
    """Render a publication-quality multi-view static figure.

    Produces a 2x2 grid of orthographic projections (front, top, side,
    isometric) with a properties annotation box.  The surface is rendered
    as filled polygons with visible structural edges.

    Parameters
    ----------
    triangles : list of tuple
        Tessellated exterior boundaries originating natively from CAD.
    volume, mass : float
        Computed volume and mass.
    center_of_mass : tuple of float
        (CG_x, CG_y, CG_z).
    inertia : tuple of float or None
        (Ixx, Iyy, Izz, Ixy, Ixz, Iyz).
    title : str
        Figure super-title.
    surface_color : str
        Face colour for the surface polygons.
    edge_color : str
        Edge colour for the visible mesh edges.
    surface_opacity : float
        Surface alpha (0-1).
    cg_color : str
        Colour for the CG marker.
    figsize : tuple of float
        Figure size in inches (width, height).
    dpi : int
        Resolution for raster output.
    save_path : str or None
        If given, save the figure to this path (PNG, PDF, SVG, ...).
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        raise ImportError(
            "Static visualization requires 'matplotlib'.  "
            "Install with:  pip install matplotlib"
        )

    polys = [[A, B, C] for A, B, C in triangles]
    all_pts = np.concatenate([np.array(p) for p in polys], axis=0)

    cx, cy, cz = center_of_mass

    # Views chosen to show the 3-D shape from all relevant angles.
    # Edge-on views (pure front/side) are avoided because lifting
    # surfaces are inherently thin and collapse to a line.
    views = [
        ("Top",           90,  -90),
        ("Front quarter", 20,  -15),
        ("Rear quarter",  20, -160),
        ("Isometric",     25,  -55),
    ]

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")

    for idx, (label, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d", facecolor="white")

        poly_col = Poly3DCollection(
            polys,
            facecolors=surface_color,
            edgecolors=surface_color,
            linewidths=0,
            alpha=surface_opacity,
        )
        ax.add_collection3d(poly_col)

        _configure_view(ax, all_pts, elev, azim)

        # CG marker
        ax.scatter(
            [cx], [cy], [cz],
            color=cg_color, s=40, zorder=10,
            depthshade=False, edgecolors="darkred", linewidths=0.5,
        )

        ax.set_axis_off()
        ax.set_title(label, fontsize=10, fontweight="bold", color="#444444",
                     pad=-5)

    # ── Properties text box (bottom-right of figure) ──────────────
    props = _build_props_text(volume, mass, center_of_mass, inertia)
    fig.text(
        0.98, 0.02, props,
        transform=fig.transFigure,
        fontsize=8, fontfamily="monospace",
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="#999999",
            alpha=0.92,
        ),
    )

    fig.suptitle(title, fontsize=13, fontweight="bold", color="#333333", y=0.98)
    fig.subplots_adjust(wspace=-0.05, hspace=0.02, left=0.0, right=1.0,
                        top=0.94, bottom=0.0)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Figure saved to: {save_path}")

    plt.show()

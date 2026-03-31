"""Visualization for lifting surface geometry and computed properties.

Provides two rendering backends:

- **Matplotlib** (static): Publication-quality multi-view figures for papers
  and reports.  Four orthographic projections (front, top, side, isometric)
  with a properties annotation box.

- **Vedo / VTK** (interactive): CAD-like 3D viewer with rotate, zoom, and pan.
  Renders the surface with realistic lighting, a centre-of-mass sphere, and
  an on-screen properties panel.

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
    title="AeroShape",
    surface_color="gold",
    surface_opacity=0.8,
    cg_color="red",
    cg_radius=None,
    window_size=(1000, 800),
    background="white",
    save_screenshot=None,
):
    """Launch an interactive 3-D viewer using Plotly.

    Replaces the 'vedo' dependency with 'plotly'. Provides zoom, rotate,
    pan, and high-fidelity B-rep edges if a build123d shape is provided.

    Parameters
    ----------
    triangles : list of tuple or build123d.Shape
        Tessellated exterior boundaries or a build123d Shape.
    volume, mass : float
        Computed volume and mass.
    center_of_mass : tuple of float
        (CG_x, CG_y, CG_z).
    inertia : tuple of float or None
        (Ixx, Iyy, Izz, Ixy, Ixz, Iyz).
    title : str
        Viewer title.
    surface_color : str
        Surface color.
    surface_opacity : float
        Surface opacity (0-1).
    cg_color : str
        Colour for the centre-of-mass marker.
    cg_radius : float or None
        Marker size.
    window_size : tuple of int
        (width, height) of the viewer (in some contexts).
    background : str
        Background colour.
    save_screenshot : str or None
        If given, save as an HTML file (Plotly doesn't directly save PNG easily without extra deps).
    """
    import plotly.graph_objects as go
    from build123d import Shape, Edge

    # 1. Prepare Mesh Data
    if hasattr(triangles, "wrapped") or str(type(triangles)).find("TopoDS_Shape") != -1:
        # It's a build123d/OCP object
        from aeroshape.nurbs.utils import tessellate_shape
        occ_shape = triangles.wrapped if hasattr(triangles, "wrapped") else triangles
        b123_shape = Shape(occ_shape)
        
        tris = tessellate_shape(occ_shape)
        verts, faces = _triangles_to_arrays(tris)
        
        # Extract B-rep edges for high-fidelity look
        edge_traces = []
        for edge in b123_shape.edges():
            pts = edge.tessellate(1e-3)
            ex = [p.X for p in pts]
            ey = [p.Y for p in pts]
            ez = [p.Z for p in pts]
            edge_traces.append(go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode='lines',
                line=dict(color='black', width=3),
                showlegend=False,
                hoverinfo='none'
            ))
    else:
        # It's a list of triangles
        verts, faces = _triangles_to_arrays(triangles)
        edge_traces = []
        # In this case, we don't have B-rep info, so we just show the mesh edges if desired
        # but the user wanted 'Outer Mold' edges only. Without B-rep, that's hard.
        # We'll just show the mesh.

    # 2. Create the Mesh3d trace
    mesh_trace = go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=surface_color,
        opacity=surface_opacity,
        lighting=dict(
            ambient=0.4,
            diffuse=0.6,
            fresnel=0.2,
            specular=0.3,
            roughness=0.4
        ),
        lightposition=dict(x=100, y=200, z=150),
        name="Lifting Surface",
        showlegend=True
    )

    # 3. CG Marker
    cx, cy, cz = center_of_mass
    cg_trace = go.Scatter3d(
        x=[cx], y=[cy], z=[cz],
        mode='markers',
        marker=dict(size=8, color=cg_color, symbol='diamond'),
        name="Center of Mass"
    )

    # 4. Properties Annotation
    props_text = _build_props_text(volume, mass, center_of_mass, inertia)
    # Convert to HTML-like for Plotly
    props_html = props_text.replace("\n", "<br>")
    
    layout = go.Layout(
        title=dict(text=title, x=0.5, y=0.95),
        scene=dict(
            xaxis_title="X (Chordwise)",
            yaxis_title="Y (Spanwise)",
            zaxis_title="Z (Thickness)",
            aspectmode='data',
            bgcolor=background
        ),
        paper_bgcolor=background,
        margin=dict(l=0, r=0, b=0, t=40),
        annotations=[
            dict(
                text=f"<b>Geometric & Mass Properties</b><br>{props_html}",
                align='left',
                showarrow=False,
                xref='paper', yref='paper',
                x=0.02, y=0.05,
                bgcolor='white',
                bordercolor='black',
                borderwidth=1,
                font=dict(family="Courier New", size=12)
            )
        ]
    )

    fig = go.Figure(data=[mesh_trace, cg_trace] + edge_traces, layout=layout)

    if save_screenshot:
        if save_screenshot.endswith(".html"):
            fig.write_html(save_screenshot)
        else:
            # Requires kaleido, which might not be there. Fallback to html.
            fig.write_html(save_screenshot + ".html")
        print(f"Interactive view saved to {save_screenshot}")
    else:
        fig.show()


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

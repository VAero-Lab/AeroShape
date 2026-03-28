"""Streamlit-based interactive GUI for the AeroShape GVM Wing Laboratory.

This application provides a web-based dashboard for designing lifting
surfaces, computing their volume and mass properties using the GVM
methodology, and exporting the results to standard CAD formats.

Launch with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math
import pandas as pd

from aeroshape.geometry import NACAProfileGenerator, WingMeshFactory
from aeroshape.mesh_utils import MeshTopologyManager
from aeroshape.volume import VolumeCalculator
from aeroshape.mass import MassPropertiesCalculator
from aeroshape import AirfoilProfile, SegmentSpec, MultiSegmentWing, NurbsExporter


def _prepare_plot_data(triangles):
    """Convert triangles to flat arrays for Plotly Mesh3d."""
    x_pts, y_pts, z_pts = [], [], []
    i_idx, j_idx, k_idx = [], [], []
    for t_id, (A, B, C) in enumerate(triangles):
        x_pts.extend([A[0], B[0], C[0]])
        y_pts.extend([A[1], B[1], C[1]])
        z_pts.extend([A[2], B[2], C[2]])
        i_idx.append(t_id * 3)
        j_idx.append(t_id * 3 + 1)
        k_idx.append(t_id * 3 + 2)
    return x_pts, y_pts, z_pts, i_idx, j_idx, k_idx


def _save_local_file(filename, data):
    """Save export data to Exports/ directory."""
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exports_dir = os.path.join(base_dir, "Exports")
    os.makedirs(exports_dir, exist_ok=True)
    name, ext = os.path.splitext(filename)
    file_path = os.path.join(exports_dir, filename)
    counter = 1
    while os.path.exists(file_path):
        file_path = os.path.join(exports_dir, f"{name}_{counter}{ext}")
        counter += 1
    with open(file_path, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode('utf-8'))
    return file_path

st.set_page_config(
    page_title="Virtual Lab: Lifting Surfaces", layout="wide"
)

# Suppress Streamlit transition animations for smoother interaction
st.markdown("""
    <style>
    .element-container, .stMarkdown, .stPlotlyChart, div, span, label {
        transition: none !important;
        animation: none !important;
    }
    .stApp > pre, .stApp > div {
        transition: none !important;
    }
    [data-testid="stAppViewBlockContainer"] {
        transition: none !important;
        animation: none !important;
    }
    [data-testid="stStatusWidget"] {
        visibility: hidden !important;
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Virtual Laboratory: CAD-Free Wings")
st.markdown(
    "Based on the article *A CAD-free methodology for volume and mass "
    "properties computation of 3-D lifting surfaces and wing-box structures*"
)

# --- Sidebar: Design Mode ---
st.sidebar.header("Design Mode")
design_mode = st.sidebar.radio(
    "Choose how to create the wing:",
    ["Standard NACA Profile", "Manual Coordinates (Editor)",
     "Free Triangles (A,B,C)"]
)

st.sidebar.markdown("---")
st.sidebar.header("Structural Topology")
structure_type = st.sidebar.radio(
    "Select the wing model (GVM):", ["Solid Body", "Thin-Shell (Hollow)"]
)
t_shell = 0.002
if structure_type == "Thin-Shell (Hollow)":
    t_shell = st.sidebar.number_input(
        "Shell skin thickness (m)", value=0.002, step=0.001, format="%.4f"
    )
st.sidebar.markdown("---")

if design_mode == "Standard NACA Profile":
    st.sidebar.header("Wing Parameters")
    naca_base = st.sidebar.text_input("Root NACA Profile (4 digits)", "2412")
    naca_tip = st.sidebar.text_input("Tip NACA Profile (4 digits)", "2412")

    st.sidebar.subheader("Geometry (meters / degrees)")
    c_root = st.sidebar.number_input("Root Chord", value=2.0, step=0.1)
    c_tip = st.sidebar.number_input("Tip Chord", value=1.0, step=0.1)
    b = st.sidebar.number_input("Semi-Span", value=10.0, step=0.5)
    sweep = st.sidebar.number_input("Sweep Angle (deg)", value=15.0, step=1.0)

    st.sidebar.subheader("Mesh Resolution")
    num_points = st.sidebar.number_input(
        "Points per Profile", value=40, step=5, min_value=3
    )
    num_sections = st.sidebar.number_input(
        "Sections along span", value=15, step=1, min_value=2
    )

elif design_mode == "Manual Coordinates (Editor)":
    st.sidebar.header("Custom Mode")
    st.sidebar.info("Edit section coordinates in the main tab.")
    b = st.sidebar.number_input("Semi-Span", value=10.0, step=0.5)
    num_sections = st.sidebar.number_input(
        "Sections (resolution along Y)", value=15, step=1, min_value=2
    )
    c_root = 1.0; c_tip = 1.0; sweep = 0.0; num_points = 0
    naca_base = "0000"; naca_tip = "0000"

else:
    st.sidebar.header("Raw Triangles Mode")
    st.sidebar.info(
        "Specify the exact 3D location of each triangle for pure "
        "GVM calculation."
    )
    b = 10.0; num_sections = 0; c_root = 1.0; c_tip = 1.0
    sweep = 0.0; num_points = 0
    naca_base = "0000"; naca_tip = "0000"


try:
    X, Y, Z = None, None, None
    triangles_for_volume = []
    triangles_for_display = []
    is_solid = (structure_type == "Solid Body")

    # --- Mesh construction logic ---
    if design_mode == "Standard NACA Profile":
        X, Y, Z = WingMeshFactory.create(
            naca_base, naca_tip, b, c_root, c_tip, sweep,
            num_points, num_sections
        )
        triangles_for_volume = MeshTopologyManager.get_wing_triangles(
            X, Y, Z, closed=True
        )
        triangles_for_display = triangles_for_volume

    elif design_mode == "Manual Coordinates (Editor)":
        st.markdown("### Manual Vertex Editor [METERS]")
        st.info(
            "All X, Y, Z coordinates must be in **Meters (m)** as required "
            "by the GVM computation standards."
        )

        col_sec, col_pts = st.columns(2)
        num_sections_manual = col_sec.number_input(
            "Number of Wing Sections/Planes (Y)",
            min_value=2, max_value=20, value=2
        )
        num_points_manual = col_pts.number_input(
            "Vertices per plane", min_value=3, max_value=100, value=5
        )

        csv_file = st.file_uploader(
            "Optional: Upload CSV with columns "
            "[Section_ID, Point_ID, X, Y, Z]",
            type=["csv"]
        )

        if csv_file is not None:
            df_init = pd.read_csv(csv_file)
            num_sections_manual = len(df_init["Section_ID"].unique())
            num_points_manual = len(df_init["Point_ID"].unique())
        else:
            # Default: simple diamond shape for editing
            initial_data = []
            for j in range(int(num_sections_manual)):
                y_val = float(j * 5.0)
                for i in range(int(num_points_manual)):
                    ang = (2 * math.pi * i) / num_points_manual
                    x_val = math.cos(ang)
                    z_val = math.sin(ang) * 0.2
                    initial_data.append({
                        "Section_ID": j, "Point_ID": i,
                        "X [m]": float(round(x_val, 4)),
                        "Y [m]": float(round(y_val, 4)),
                        "Z [m]": float(round(z_val, 4))
                    })
            df_init = pd.DataFrame(initial_data)

        st.markdown("**Vertex Coordinates Table (by section, in METERS):**")
        df_edited = st.data_editor(
            df_init, use_container_width=True, hide_index=True
        )

        # Reconstruct mesh from the edited vertex table
        X = np.zeros((num_sections_manual, num_points_manual))
        Y = np.zeros((num_sections_manual, num_points_manual))
        Z = np.zeros((num_sections_manual, num_points_manual))

        for j in range(num_sections_manual):
            for i in range(num_points_manual):
                mask = (
                    (df_edited["Section_ID"] == j)
                    & (df_edited["Point_ID"] == i)
                )
                if mask.any():
                    row = df_edited[mask].iloc[0]
                    X[j, i] = row["X [m]"]
                    Y[j, i] = row["Y [m]"]
                    Z[j, i] = row["Z [m]"]

        triangles_for_volume = MeshTopologyManager.get_wing_triangles(
            X, Y, Z, closed=True
        )
        triangles_for_display = triangles_for_volume

    else:
        # Free Triangles mode: pure raw mesh input
        st.markdown("### Raw Mesh Editor (METERS)")
        st.info(
            "Enter the 3 vertices of each triangle (A, B, C) with "
            "coordinates in METERS [m]."
        )

        csv_mesh_file = st.file_uploader(
            "Upload Triangles CSV "
            "[Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz]",
            type=["csv"]
        )
        if csv_mesh_file is not None:
            df_tri = pd.read_csv(csv_mesh_file)
        else:
            # Demo: sealed tetrahedron (4 faces, perfect for GVM validation)
            demo_tetrahedron = [
                {"Ax [m]": 0.0, "Ay [m]": 0.0, "Az [m]": 0.0,
                 "Bx [m]": 2.0, "By [m]": 0.0, "Bz [m]": 0.0,
                 "Cx [m]": 0.0, "Cy [m]": 2.0, "Cz [m]": 0.0},
                {"Ax [m]": 0.0, "Ay [m]": 0.0, "Az [m]": 0.0,
                 "Bx [m]": 0.0, "By [m]": 2.0, "Bz [m]": 0.0,
                 "Cx [m]": 0.0, "Cy [m]": 0.0, "Cz [m]": 2.0},
                {"Ax [m]": 0.0, "Ay [m]": 0.0, "Az [m]": 0.0,
                 "Bx [m]": 0.0, "By [m]": 0.0, "Bz [m]": 2.0,
                 "Cx [m]": 2.0, "Cy [m]": 0.0, "Cz [m]": 0.0},
                {"Ax [m]": 2.0, "Ay [m]": 0.0, "Az [m]": 0.0,
                 "Bx [m]": 0.0, "By [m]": 0.0, "Bz [m]": 2.0,
                 "Cx [m]": 0.0, "Cy [m]": 2.0, "Cz [m]": 0.0},
            ]
            df_tri = pd.DataFrame(demo_tetrahedron)

        st.markdown(
            "**Triangle Vertex Table (A, B, C coordinates in METERS):**"
        )
        df_tri_edited = st.data_editor(
            df_tri, num_rows="dynamic",
            use_container_width=True, hide_index=True
        )

        for _, r in df_tri_edited.iterrows():
            try:
                A = np.array([
                    float(r["Ax [m]"]), float(r["Ay [m]"]),
                    float(r["Az [m]"])
                ])
                B = np.array([
                    float(r["Bx [m]"]), float(r["By [m]"]),
                    float(r["Bz [m]"])
                ])
                C = np.array([
                    float(r["Cx [m]"]), float(r["Cy [m]"]),
                    float(r["Cz [m]"])
                ])
                triangles_for_volume.append((A, B, C))
                triangles_for_display.append((A, B, C))
            except Exception:
                pass

    # --- Volume computation via Divergence Theorem (Eq. 2) ---
    solid_volume = VolumeCalculator.compute_solid_volume(triangles_for_volume)

    # Total surface area
    total_area = VolumeCalculator.compute_surface_area(triangles_for_display)

    # Material selection for mass estimation
    material_options = {
        "EPS Foam (50 kg/m3)": 50.0,
        "Balsa Wood (150 kg/m3)": 150.0,
        "Carbon Fiber (1600 kg/m3)": 1600.0,
        "Aluminum 6061 (2700 kg/m3)": 2700.0,
        "Steel (7850 kg/m3)": 7850.0,
    }

    # --- Main Results ---
    if not is_solid:
        st.markdown(
            "### Comparison: Offset (V_outer - V_inner) vs. Unfolding (Area x t)"
        )
        col1, col2, col3, col4, col5 = st.columns(5)

        if X is not None and Y is not None and Z is not None:
            # Offset method: V_outer - V_inner
            vol_shell, vol_outer, vol_inner = (
                VolumeCalculator.compute_shell_volume_offset(
                    X, Y, Z, t_shell
                )
            )
            # Unfolding method: area * t_shell
            vol_approx = VolumeCalculator.compute_shell_volume_unfolding(
                triangles_for_volume, t_shell
            )
            area_base = total_area
        else:
            area_base = total_area
            vol_shell = solid_volume
            vol_outer = solid_volume
            vol_inner = 0.0
            vol_approx = area_base * t_shell

        effective_volume = vol_shell
        error_diff = vol_approx - vol_shell

        col1.metric(
            "1. Offset (V_out - V_in)",
            f"{vol_shell:.5f} m3",
            f"V_outer={vol_outer:.5f}, V_inner={vol_inner:.5f}"
        )
        col2.metric(
            "2. Unfolding (Area x t)",
            f"{vol_approx:.5f} m3",
            f"Difference: {error_diff:+.6f} m3",
            delta_color="off"
        )

        material_sel = col3.selectbox(
            "Wing Material:", list(material_options.keys())
        )
        density = material_options[material_sel]
        mass = effective_volume * density

        col4.metric("Approx Mass", f"{mass:.2f} kg", f"{structure_type}")
        col5.metric(
            "Outer Mesh",
            f"{len(triangles_for_display)} faces",
            f"Skin Area: {area_base:.2f} m2"
        )
    else:
        effective_volume = solid_volume
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Effective Vol.",
            f"{effective_volume:.5f} m3",
            "Divergence Theorem"
        )

        material_sel = col2.selectbox(
            "Wing Material:", list(material_options.keys())
        )
        density = material_options[material_sel]
        mass = effective_volume * density

        col3.metric("Final Mass", f"{mass:.2f} kg", f"{structure_type}")
        col4.metric(
            "Mesh Performance",
            f"{len(triangles_for_display)} faces",
            f"GVM Area: {total_area:.2f} m2"
        )

    # --- GVM: Mass Distribution and Inertia (Section 2.3) ---
    if X is not None and Y is not None and Z is not None and mass > 0:
        st.markdown("---")
        st.markdown(
            "### Inertia and Center of Mass (GVM)"
        )
        st.info(
            "Particle-system implementation for computing moments of "
            "inertia of aeronautical surfaces (Eqs. 4-14 of the paper)."
        )

        CG, inertias, M_3D = MassPropertiesCalculator.compute_all(
            X, Y, Z, mass
        )
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = inertias

        col_cg, col_iner = st.columns(2)
        with col_cg:
            st.success("**Center of Mass (Spatial Location)**")
            st.write(f"- **X Axis (Chord):** `{CG[0]:.5f}` m")
            st.write(f"- **Y Axis (Span):** `{CG[1]:.5f}` m")
            st.write(f"- **Z Axis (Height):** `{CG[2]:.5f}` m")

        with col_iner:
            st.success("**Moments of Inertia (Tensor)**")
            st.write(
                f"- **Ixx:** `{Ixx:.5f}` kg m2 &nbsp; | &nbsp; "
                f"**Ixy:** `{Ixy:.5f}` kg m2"
            )
            st.write(
                f"- **Iyy:** `{Iyy:.5f}` kg m2 &nbsp; | &nbsp; "
                f"**Ixz:** `{Ixz:.5f}` kg m2"
            )
            st.write(
                f"- **Izz:** `{Izz:.5f}` kg m2 &nbsp; | &nbsp; "
                f"**Iyz:** `{Iyz:.5f}` kg m2"
            )

    # --- Export ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Model")
    st.sidebar.info(
        "When importing the mesh into CAD software:\n"
        "1. Ensure the import **unit is METERS**.\n"
        "2. Use Mesh -> Convert to transform the faceted body."
    )

    volume_str = f"{effective_volume:.6f}"
    mass_str = f"{mass:.3f}"

    export_format = st.sidebar.selectbox(
        "Select 3D format:", ["STL", "IGES", "STEP"]
    )

    if st.sidebar.button(f"Save model as {export_format} to Exports/"):
        try:
            # Build NURBS shape for export
            root_prof = AirfoilProfile.from_naca4(
                naca_root, num_points=num_points_profile)
            tip_prof = AirfoilProfile.from_naca4(
                naca_tip, num_points=num_points_profile)
            _wing = MultiSegmentWing(name="GUI Wing")
            _wing.add_segment(SegmentSpec(
                span=semi_span, root_airfoil=root_prof,
                tip_airfoil=tip_prof, root_chord=chord_root,
                tip_chord=chord_tip, sweep_le_deg=sweep_angle,
                num_sections=num_sections,
            ))
            _shape = _wing.to_occ_shape()

            ext = export_format.lower()
            fname = f"Wing_GVM.{ext}"
            import tempfile, shutil
            with tempfile.NamedTemporaryFile(
                suffix=f".{ext}", delete=False
            ) as tmp:
                tmp_path = tmp.name

            if export_format == "STL":
                NurbsExporter.to_stl(_shape, tmp_path)
            elif export_format == "IGES":
                NurbsExporter.to_iges(_shape, tmp_path)
            else:
                NurbsExporter.to_step(_shape, tmp_path)

            with open(tmp_path, 'rb') as f:
                data = f.read()
            import os
            os.remove(tmp_path)
            saved_path = _save_local_file(fname, data)
            st.sidebar.success(f"Saved successfully to: {saved_path}")
        except Exception as e:
            st.sidebar.error(f"Export failed: {e}")

    # --- 3D Visualization ---
    st.markdown("---")
    st.subheader("3D Visualization")

    x_pts, y_pts, z_pts, i_idx, j_idx, k_idx = (
        _prepare_plot_data(triangles_for_display)
    )

    fig = go.Figure(data=[go.Mesh3d(
        x=x_pts, y=y_pts, z=z_pts,
        i=i_idx, j=j_idx, k=k_idx,
        colorscale='Viridis',
        intensity=z_pts,
        showscale=False,
        flatshading=True,
        lighting=dict(
            ambient=0.6, diffuse=1.0, roughness=0.3,
            specular=0.5, fresnel=0.5
        )
    )])

    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title='X (Chord)',
            yaxis_title='Y (Span)',
            zaxis_title='Z (Height)'
        ),
        uirevision='constant',
        margin=dict(l=0, r=0, b=0, t=30),
        height=600,
        transition_duration=0
    )

    st.plotly_chart(
        fig, use_container_width=True, theme=None, key="wing_plot_3d"
    )

except Exception as e:
    st.error(
        f"Error processing geometry. Check your input values. Details: {e}"
    )

st.markdown("---")
st.text("AeroShape - GVM Methodology for Geometry, Volume and Mass Computation")

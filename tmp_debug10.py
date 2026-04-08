"""Check the main wing root segment for TE ribbon.

The user's image showed fat ribbon on the entire wing TE, not just winglet tips.
Let's examine the main wing segment solid carefully.
"""
from examples.aircraft_commercial_airliner import create_wings
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
import numpy as np

wings = create_wings()
wing = wings[0][0]

# Get all segments (this creates the actual STEP solids)
segments = wing.to_occ_segments()
print(f"Total OCC segments: {len(segments)}")

for seg_idx, seg in enumerate(segments):
    shape = seg.wrapped if hasattr(seg, 'wrapped') else seg
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    idx = 0
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        gp = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, gp)
        adaptor = BRepAdaptor_Surface(face)
        u0, u1 = adaptor.FirstUParameter(), adaptor.LastUParameter()
        v0, v1 = adaptor.FirstVParameter(), adaptor.LastVParameter()
        
        # Sample multiple TE points (u_min to see if there's real variation)
        te_pts_start = []
        te_pts_end = []
        for v_frac in np.linspace(0, 1, 5):
            v = v0 + v_frac * (v1 - v0)
            ps = adaptor.Value(u0, v)
            pe = adaptor.Value(u1, v)
            te_pts_start.append((ps.X(), ps.Y(), ps.Z()))
            te_pts_end.append((pe.X(), pe.Y(), pe.Z()))
        
        print(f"\n  Segment {seg_idx}, Face {idx}:")
        print(f"    U=[{u0:.3f},{u1:.3f}], V=[{v0:.3f},{v1:.3f}], area={gp.Mass():.4f}")
        for i, (s, e) in enumerate(zip(te_pts_start, te_pts_end)):
            gap = ((s[0]-e[0])**2 + (s[1]-e[1])**2 + (s[2]-e[2])**2)**0.5
            print(f"    v_frac={i/4:.2f}: u_start=({s[0]:.3f},{s[1]:.3f},{s[2]:.3f}), "
                  f"u_end=({e[0]:.3f},{e[1]:.3f},{e[2]:.3f}), gap={gap:.6f}")
        idx += 1
        explorer.Next()

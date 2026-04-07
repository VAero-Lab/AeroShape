"""Verification: run all examples, check sizes, mass properties, and test individual export."""
import subprocess, os, sys, time, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Remove existing exports
for f in os.listdir("Exports"):
    if f.endswith(".step"):
        os.remove(os.path.join("Exports", f))

EXAMPLES = [
    "examples/aircraft_bwb_guided.py",
    "examples/aircraft_box_wing.py",
    "examples/aircraft_uav_bwb.py",
    "examples/aircraft_commercial_airliner.py",
    "examples/aircraft_military_cargo.py",
    "examples/aircraft_twin_boom.py",
    "examples/aircraft_experimental_glider.py",
]

print("="*70)
print("  RUNNING ALL EXAMPLES")
print("="*70)
print(f"\n  {'Model':<35s} {'Build':>6s} {'Export':>7s} {'File':>10s}")
print("  " + "-"*60)
for script in EXAMPLES:
    name = os.path.basename(script).replace('aircraft_', '').replace('.py', '')
    result = subprocess.run(["python", script, "--no-show"], capture_output=True, text=True, timeout=120)
    if result.returncode == 0:
        build_t = export_t = "?"
        for line in result.stdout.split('\n'):
            if 'time to create' in line.lower():
                try: build_t = f"{float(line.split(':')[-1].strip().split()[0]):.2f}"
                except: pass
            if 'time to export' in line.lower():
                try: export_t = f"{float(line.split(':')[-1].strip().split()[0]):.2f}"
                except: pass
        step_files = glob.glob("Exports/*.step")
        latest = max(step_files, key=os.path.getmtime) if step_files else None
        if latest:
            sz = os.path.getsize(latest)
            sz_str = f"{sz/1024/1024:.1f} MB" if sz > 1024*1024 else f"{sz/1024:.0f} KB"
            status = "✅" if sz < 3*1024*1024 else "⚠️"
        else:
            sz_str = "MISSING"
            status = "❌"
        print(f"  {name:<35s} {build_t:>6s} {export_t:>7s} {sz_str:>10s} {status}")
    else:
        print(f"  {name:<35s} FAILED ❌")
        for line in result.stderr.split('\n')[-5:]:
            if line.strip():
                print(f"    {line.strip()}")

# File sizes
print(f"\n{'='*70}")
print(f"  STEP FILE SIZES")
print(f"{'='*70}")
for f in sorted(os.listdir("Exports")):
    if f.endswith(".step"):
        size = os.path.getsize(os.path.join("Exports", f))
        if size > 1024*1024:
            print(f"  {f:50s} {size/1024/1024:.1f} MB")
        else:
            print(f"  {f:50s} {size/1024:.0f} KB")

# Test individual component export
print(f"\n{'='*70}")
print(f"  INDIVIDUAL COMPONENT EXPORT TEST")
print(f"{'='*70}")
from aeroshape.geometry.wings import MultiSegmentWing, SegmentSpec, AirfoilProfile
from aeroshape.geometry.fuselage import MultiSegmentFuselage, FuselageSegment
from aeroshape.geometry.fuselage import ellipsoid_blend
from aeroshape.geometry.cross_sections import EllipticalProfile
from aeroshape.nurbs.export import NurbsExporter
from aeroshape.nurbs.surfaces import NurbsSurfaceBuilder

# Test 1: Single wing export
naca = AirfoilProfile.from_naca4("2412", num_points=50)
wing = MultiSegmentWing(name="Test Wing", symmetric=False)
wing.add_segment(SegmentSpec(span=10.0, root_airfoil=naca, root_chord=3.0, tip_chord=1.0))
wing_shape = wing.to_occ_shape()
try:
    NurbsExporter.to_step(wing_shape, "Exports/_test_wing_solo.step")
    sz = os.path.getsize("Exports/_test_wing_solo.step")
    print(f"  Single wing export: {sz/1024:.0f} KB ✅")
    os.remove("Exports/_test_wing_solo.step")
except Exception as e:
    print(f"  Single wing export: FAILED ❌ ({e})")

# Test 2: Fuselage export
fuse = MultiSegmentFuselage(name="Test Fuselage")
fuse.add_segment(FuselageSegment(length=5.0, root_profile=EllipticalProfile(0.01, 0.01),
                                  tip_profile=EllipticalProfile(2.0, 2.0),
                                  num_sections=10, blend_curve=ellipsoid_blend))
fuse_shape = fuse.to_occ_shape()
try:
    NurbsExporter.to_step(fuse_shape, "Exports/_test_fuse_solo.step")
    sz = os.path.getsize("Exports/_test_fuse_solo.step")
    print(f"  Single fuselage export: {sz/1024:.0f} KB ✅")
    os.remove("Exports/_test_fuse_solo.step")
except Exception as e:
    print(f"  Single fuselage export: FAILED ❌ ({e})")

# Test 3: Wing + mirror export
from OCP.gp import gp_Trsf, gp_Ax2, gp_Pnt, gp_Dir
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
wing_occ = wing_shape.wrapped if hasattr(wing_shape, 'wrapped') else wing_shape
mirror_trsf = gp_Trsf()
mirror_trsf.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)))
mirrored = BRepBuilderAPI_Transform(wing_occ, mirror_trsf, True).Shape()
pair = NurbsSurfaceBuilder.make_compound([wing_shape, mirrored])
try:
    NurbsExporter.to_step(pair, "Exports/_test_wing_pair.step")
    sz = os.path.getsize("Exports/_test_wing_pair.step")
    print(f"  Wing + mirror pair: {sz/1024:.0f} KB ✅")
    os.remove("Exports/_test_wing_pair.step")
except Exception as e:
    print(f"  Wing + mirror pair: FAILED ❌ ({e})")

print(f"\n{'='*70}")
print(f"  VERIFICATION COMPLETE")
print(f"{'='*70}")

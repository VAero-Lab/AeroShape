"""Run all examples with --no-show and time them, then report file sizes."""
import subprocess, os, time

EXAMPLES = [
    "examples/aircraft_bwb_guided.py",
    "examples/aircraft_box_wing.py",
    "examples/aircraft_uav_bwb.py",
    "examples/aircraft_commercial_airliner.py",
    "examples/aircraft_military_cargo.py",
    "examples/aircraft_twin_boom.py",
    "examples/aircraft_experimental_glider.py",
]

# Remove existing exports first
for f in os.listdir("Exports"):
    if f.endswith(".step"):
        os.remove(os.path.join("Exports", f))

for script in EXAMPLES:
    name = os.path.basename(script).replace('.py', '')
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        ["python", script, "--no-show"],
        capture_output=True, text=True, timeout=300
    )
    t1 = time.time()
    print(f"  Wall time: {t1-t0:.2f}s")
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
    else:
        # Print relevant output lines
        for line in result.stdout.split('\n'):
            if any(k in line.lower() for k in ['time', 'export', 'volume', 'mass', 'step']):
                print(f"  {line.strip()}")

print(f"\n\n{'='*60}")
print(f"  FILE SIZES")
print(f"{'='*60}")
for f in sorted(os.listdir("Exports")):
    if f.endswith(".step"):
        size = os.path.getsize(os.path.join("Exports", f))
        if size > 1024*1024:
            print(f"  {f:45s} {size/1024/1024:.1f} MB")
        else:
            print(f"  {f:45s} {size/1024:.0f} KB")

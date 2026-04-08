import time
import math
from aeroshape.geometry.wings import MultiSegmentWing, AirfoilProfile
from examples.aircraft_box_wing import create_box_wings

print("Evaluating create_box_wings planform chord drift...")

wings = create_box_wings()
fin = wings[2][0]

frames = fin.get_section_frames()
for i, fr in enumerate(frames):
    print(f"Sec {i}: Z={fr['z_offset']:.2f}, Chord={fr['chord']:.3f}, thick_dir={fr.get('v_thickness_dir')}")


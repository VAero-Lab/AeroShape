from examples.aircraft_box_wing import create_box_wings
from aeroshape.geometry.wings import MultiSegmentWing

print("Evaluating box fin vectors...")
wings = create_box_wings()
fin = wings[2][0]

frames = fin.get_section_frames()
for i, fr in enumerate(frames):
    print(f"Sec {i}: v_chord={fr.get('v_chord_dir')}, v_thick={fr.get('v_thickness_dir')}")

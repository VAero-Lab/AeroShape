"""Check if to_vertex_grids works properly."""
from examples.aircraft_box_wing import create_box_wings

wings = create_box_wings()
fin = wings[2][0]

print(f"Testing {fin.name}...")
X, Y, Z = fin.to_vertex_grids()

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"Z shape: {Z.shape}")
print("Grids successfully computed!")

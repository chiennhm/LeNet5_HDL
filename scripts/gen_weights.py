"""
Generate example weight .hex files for LeNet-5 Verilog design.
All weights are initialized to small random values in Q8.8 format.
Replace these with trained weights for real inference.
"""
import random
import os

random.seed(42)

def q88_hex(val):
    """Convert a float to Q8.8 hex string (16-bit signed)."""
    ival = int(round(val * 256))
    ival = max(-32768, min(32767, ival))
    if ival < 0:
        ival += 65536
    return f"{ival:04X}"

def write_hex(filepath, values):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for v in values:
            f.write(q88_hex(v) + "\n")
    print(f"  {filepath}: {len(values)} values")

def rand_weights(n, scale=0.1):
    return [random.uniform(-scale, scale) for _ in range(n)]

def zero_biases(n):
    return [0.0] * n

mem = "d:/LeNetHDL/mem"

print("Generating weight files...")

# C1: 6 filters × 1 ch × 5×5 = 150 weights, 6 biases
write_hex(f"{mem}/conv1_weights.hex", rand_weights(6 * 1 * 5 * 5))
write_hex(f"{mem}/conv1_bias.hex",    zero_biases(6))

# C3: 16 filters × 6 ch × 5×5 = 2400 weights, 16 biases
write_hex(f"{mem}/conv3_weights.hex", rand_weights(16 * 6 * 5 * 5))
write_hex(f"{mem}/conv3_bias.hex",    zero_biases(16))

# FC1: 16→120 = 1920 weights, 120 biases
write_hex(f"{mem}/fc1_weights.hex", rand_weights(16 * 120))
write_hex(f"{mem}/fc1_bias.hex",    zero_biases(120))

# FC2: 120→84 = 10080 weights, 84 biases
write_hex(f"{mem}/fc2_weights.hex", rand_weights(120 * 84))
write_hex(f"{mem}/fc2_bias.hex",    zero_biases(84))

# FC3: 84→10 = 840 weights, 10 biases
write_hex(f"{mem}/fc3_weights.hex", rand_weights(84 * 10))
write_hex(f"{mem}/fc3_bias.hex",    zero_biases(10))

# Generate a test image (14×14 = 196 pixels) — simple digit "1" pattern
print("\nGenerating test image...")
img = [0] * 196
# Draw a vertical bar in the middle (column 7) for digit "1"
for r in range(2, 12):
    img[r * 14 + 7] = 200
    img[r * 14 + 6] = 150
# Small top stroke
img[2 * 14 + 5] = 100
with open(f"{mem}/test_image.hex", "w") as f:
    for p in img:
        f.write(f"{p:02X}\n")
print(f"  {mem}/test_image.hex: 196 pixels")

print("\nDone! All .hex files generated in mem/")

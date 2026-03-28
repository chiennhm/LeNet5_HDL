"""
Generate example weight .hex files for LeNet-5 Verilog design.
All weights are initialized to small random values in Q8.8 format.
Replace these with trained weights for real inference.

Architecture (matches model.py):
  Conv2d(1→6, k=5)    + ReLU  → AvgPool(2)
  Conv2d(6→16, k=5)   + ReLU  → AvgPool(2)
  Conv2d(16→120, k=4) + ReLU  → flatten
  Linear(120→84)       + ReLU
  Linear(84→10)
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

mem = "d:/LeNetHDL/LeNet_verilog"

print("Generating weight files...")
print("Architecture: 28×28 input, matching model.py\n")

# C1: Conv2d(1→6, k=5) → 6 filters × 1 ch × 5×5 = 150 weights, 6 biases
write_hex(f"{mem}/conv1_weights.hex", rand_weights(6 * 1 * 5 * 5))
write_hex(f"{mem}/conv1_bias.hex",    zero_biases(6))

# C3: Conv2d(6→16, k=5) → 16 filters × 6 ch × 5×5 = 2400 weights, 16 biases
write_hex(f"{mem}/conv3_weights.hex", rand_weights(16 * 6 * 5 * 5))
write_hex(f"{mem}/conv3_bias.hex",    zero_biases(16))

# C5: Conv2d(16→120, k=4) → 120 filters × 16 ch × 4×4 = 30720 weights, 120 biases
write_hex(f"{mem}/c5_weights.hex", rand_weights(120 * 16 * 4 * 4))
write_hex(f"{mem}/c5_bias.hex",    zero_biases(120))

# FC1: Linear(120→84) → 84×120 = 10080 weights, 84 biases
write_hex(f"{mem}/fc1_weights.hex", rand_weights(84 * 120))
write_hex(f"{mem}/fc1_bias.hex",    zero_biases(84))

# FC2: Linear(84→10) → 10×84 = 840 weights, 10 biases
write_hex(f"{mem}/fc2_weights.hex", rand_weights(10 * 84))
write_hex(f"{mem}/fc2_bias.hex",    zero_biases(10))

# Generate a test image (28×28 = 784 pixels) — simple digit "1" pattern
print("\nGenerating test image (28×28)...")
img = [0] * 784
# Draw a vertical bar in the middle (column 14) for digit "1"
for r in range(4, 24):
    img[r * 28 + 14] = 200
    img[r * 28 + 13] = 150
# Small top stroke
img[4 * 28 + 12] = 100
img[4 * 28 + 11] = 50
with open(f"{mem}/test_image.hex", "w") as f:
    for p in img:
        f.write(f"{p:02X}\n")
print(f"  {mem}/test_image.hex: 784 pixels")

print("\nDone! All .hex files generated in mem/")

# Print summary
print("\n--- Parameter Summary ---")
print(f"  C1  weights: {6*1*5*5:>6}  biases: {6:>4}")
print(f"  C3  weights: {16*6*5*5:>6}  biases: {16:>4}")
print(f"  C5  weights: {120*16*4*4:>6}  biases: {120:>4}")
print(f"  FC1 weights: {84*120:>6}  biases: {84:>4}")
print(f"  FC2 weights: {10*84:>6}  biases: {10:>4}")
total_w = 6*1*5*5 + 16*6*5*5 + 120*16*4*4 + 84*120 + 10*84
total_b = 6 + 16 + 120 + 84 + 10
print(f"  Total: {total_w} weights + {total_b} biases = {total_w+total_b} params")

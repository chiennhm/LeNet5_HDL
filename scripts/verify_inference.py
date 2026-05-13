"""
Pure-Python fixed-point inference matching the RTL's Q8.8 × INT8(Q0.7) arithmetic.
Verifies whether the weight/bias files produce correct predictions.
"""
import numpy as np
from pathlib import Path
import sys

MEM = Path(__file__).resolve().parents[1] / "mem"

def load_hex_int8(path):
    vals = []
    for line in open(path):
        t = line.strip()
        if t and not t.startswith("//"):
            v = int(t, 16)
            if v >= 128: v -= 256  # signed
            vals.append(v)
    return np.array(vals, dtype=np.int32)

def load_hex_uint8(path):
    vals = []
    for line in open(path):
        t = line.strip()
        if t and not t.startswith("//"):
            vals.append(int(t, 16))
    return np.array(vals, dtype=np.int32)

def q88_relu(acc):
    """acc is Q16.15 (32-bit), extract Q8.8, saturate, ReLU"""
    acc_q88 = (acc >> 7) & 0xFFFF
    if acc_q88 >= 0x8000: acc_q88 -= 0x10000  # sign extend
    # Check overflow
    top_bits = (acc >> 23) & 0x1FF
    if top_bits != 0 and top_bits != 0x1FF:
        if acc < 0:
            acc_q88 = -32768
        else:
            acc_q88 = 32767
    # ReLU
    if acc_q88 < 0:
        acc_q88 = 0
    return acc_q88

def q88_no_relu(acc):
    acc_q88 = (acc >> 7) & 0xFFFF
    if acc_q88 >= 0x8000: acc_q88 -= 0x10000
    top_bits = (acc >> 23) & 0x1FF
    if top_bits != 0 and top_bits != 0x1FF:
        if acc < 0:
            acc_q88 = -32768
        else:
            acc_q88 = 32767
    return acc_q88

def conv2d(input_map, weights, biases, in_size, in_ch, out_ch, kernel, relu=True):
    out_size = in_size - kernel + 1
    output = np.zeros((out_ch, out_size, out_size), dtype=np.int32)
    w_idx = 0
    for oc in range(out_ch):
        for oy in range(out_size):
            for ox in range(out_size):
                # Bias: INT8 → Q16.15 = {sign_extend, bias, 8'b0}
                b = int(biases[oc])
                acc = (b << 8)  # sign-extended to 32-bit by Python
                # Make it proper 32-bit signed
                if acc >= (1 << 31): acc -= (1 << 32)
                
                for ic in range(in_ch):
                    for ky in range(kernel):
                        for kx in range(kernel):
                            pixel = int(input_map[ic, oy+ky, ox+kx])  # Q8.8
                            w = int(weights[oc * in_ch * kernel * kernel + ic * kernel * kernel + ky * kernel + kx])
                            # Q8.8 × INT8 = Q8.15 (24-bit) → sign extend to 32
                            prod = pixel * w
                            acc += prod
                
                if relu:
                    output[oc, oy, ox] = q88_relu(acc)
                else:
                    output[oc, oy, ox] = q88_no_relu(acc)
    return output

def avgpool2d(input_map, ch, in_size):
    out_size = in_size // 2
    output = np.zeros((ch, out_size, out_size), dtype=np.int32)
    for c in range(ch):
        for oy in range(out_size):
            for ox in range(out_size):
                s = (int(input_map[c, oy*2, ox*2]) + int(input_map[c, oy*2, ox*2+1]) +
                     int(input_map[c, oy*2+1, ox*2]) + int(input_map[c, oy*2+1, ox*2+1]))
                output[c, oy, ox] = s >> 2  # divide by 4
    return output

def fc(input_vec, weights, biases, in_size, out_size, relu=True):
    output = np.zeros(out_size, dtype=np.int32)
    for j in range(out_size):
        b = int(biases[j])
        acc = (b << 8)
        for i in range(in_size):
            pixel = int(input_vec[i])
            w = int(weights[j * in_size + i])
            acc += pixel * w
        if relu:
            output[j] = q88_relu(acc)
        else:
            output[j] = q88_no_relu(acc)
    return output

def run_inference(image_path):
    # Load weights and biases (INT8)
    c1_w = load_hex_int8(MEM / "conv1_weights.hex")
    c1_b = load_hex_int8(MEM / "conv1_bias.hex")
    c3_w = load_hex_int8(MEM / "conv3_weights.hex")
    c3_b = load_hex_int8(MEM / "conv3_bias.hex")
    c5_w = load_hex_int8(MEM / "c5_weights.hex")
    c5_b = load_hex_int8(MEM / "c5_bias.hex")
    f1_w = load_hex_int8(MEM / "fc1_weights.hex")
    f1_b = load_hex_int8(MEM / "fc1_bias.hex")
    f2_w = load_hex_int8(MEM / "fc2_weights.hex")
    f2_b = load_hex_int8(MEM / "fc2_bias.hex")

    print(f"Weights: c1={len(c1_w)}, c3={len(c3_w)}, c5={len(c5_w)}, f1={len(f1_w)}, f2={len(f2_w)}")

    # Load image (UINT8 → Q8.8 = {0, pixel})
    img_raw = load_hex_uint8(image_path)
    print(f"Image: {len(img_raw)} pixels, nonzero={np.count_nonzero(img_raw)}, max={img_raw.max()}")
    
    # Image as Q8.8: upper 8 bits = 0, lower 8 bits = pixel value
    img = img_raw.reshape(1, 28, 28).astype(np.int32)  # already Q8.8 format

    # C1: Conv 28x28x1 → 24x24x6
    print("Running C1 (conv 28→24, 6 filters)...")
    c1_out = conv2d(img, c1_w, c1_b, 28, 1, 6, 5)
    print(f"  C1 output: shape={c1_out.shape}, range=[{c1_out.min()}, {c1_out.max()}]")

    # S2: AvgPool 24x24x6 → 12x12x6
    print("Running S2 (avgpool 24→12)...")
    s2_out = avgpool2d(c1_out, 6, 24)

    # C3: Conv 12x12x6 → 8x8x16
    print("Running C3 (conv 12→8, 16 filters)...")
    c3_out = conv2d(s2_out, c3_w, c3_b, 12, 6, 16, 5)

    # S4: AvgPool 8x8x16 → 4x4x16
    print("Running S4 (avgpool 8→4)...")
    s4_out = avgpool2d(c3_out, 16, 8)

    # C5: Conv 4x4x16 → 1x1x120
    print("Running C5 (conv 4→1, 120 filters)...")
    c5_out = conv2d(s4_out, c5_w, c5_b, 4, 16, 120, 4)

    # Flatten: 120x1x1 → 120
    flat = c5_out.flatten()

    # FC1: 120 → 84 + ReLU
    print("Running FC1 (120→84)...")
    fc1_out = fc(flat, f1_w, f1_b, 120, 84, relu=True)

    # FC2: 84 → 10 (no ReLU)
    print("Running FC2 (84→10)...")
    fc2_out = fc(fc1_out, f2_w, f2_b, 84, 10, relu=False)

    print(f"\nFC2 logits (Q8.8): {fc2_out}")
    print(f"FC2 logits (float): {fc2_out / 256.0}")
    
    predicted = np.argmax(fc2_out)
    print(f"\n*** Predicted digit: {predicted} ***")
    return predicted

if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else str(MEM / "test_image.hex")
    run_inference(img)

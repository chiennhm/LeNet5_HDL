"""Export LeNet-5 weights per layer in Q8.8 / INT8 / INT4 hex for Verilog."""

import os, struct, argparse
import numpy as np
import torch
from model import LeNet5

# PyTorch name → Verilog name
LAYER_MAP = {
    "feature_extractor.0.weight": ("conv1", "weights"),
    "feature_extractor.0.bias":   ("conv1", "bias"),
    "feature_extractor.3.weight": ("conv3", "weights"),
    "feature_extractor.3.bias":   ("conv3", "bias"),
    "feature_extractor.6.weight": ("c5",    "weights"),
    "feature_extractor.6.bias":   ("c5",    "bias"),
    "classifier.0.weight":        ("fc1",   "weights"),
    "classifier.0.bias":          ("fc1",   "bias"),
    "classifier.2.weight":        ("fc2",   "weights"),
    "classifier.2.bias":          ("fc2",   "bias"),
}

# ── Quantization ─────────────────────────────────────────────────────────────

def _quantize_at(tensor, threshold, qmin, qmax):
    scale = threshold / qmax if threshold > 0 else 1.0
    q = np.clip(np.round(tensor / scale), qmin, qmax).astype(np.int8)
    return q, float(scale)

def _mse_at(tensor, threshold, qmin, qmax):
    q, s = _quantize_at(tensor, threshold, qmin, qmax)
    return float(np.mean((tensor - q.astype(np.float32) * s) ** 2))

def calibrate_minmax(tensor, n_bits):
    qmax = (1 << (n_bits - 1)) - 1
    return _quantize_at(tensor, float(np.max(np.abs(tensor))), -qmax, qmax)

def calibrate_mse(tensor, n_bits, steps=200):
    qmax = (1 << (n_bits - 1)) - 1
    qmin = -qmax
    amax = float(np.max(np.abs(tensor)))
    if amax == 0:
        return np.zeros_like(tensor, dtype=np.int8), 1.0
    best_t, best_mse = amax, float('inf')
    for i in range(steps):
        t = amax * (0.5 + 0.5 * i / (steps - 1))
        m = _mse_at(tensor, t, qmin, qmax)
        if m < best_mse:
            best_mse, best_t = m, t
    return _quantize_at(tensor, best_t, qmin, qmax)

def calibrate_kl(tensor, n_bits, n_bins=2048):
    qmax = (1 << (n_bits - 1)) - 1
    qmin, n_levels = -qmax, qmax + 1
    amax = float(np.max(np.abs(tensor)))
    if amax == 0:
        return np.zeros_like(tensor, dtype=np.int8), 1.0

    hist, edges = np.histogram(np.abs(tensor).flatten(), bins=n_bins, range=(0, amax))
    hist = np.where(hist == 0, 1e-10, hist.astype(np.float64))

    best_kl, best_idx = float('inf'), n_bins
    for ci in range(max(n_levels, n_bins // 4), n_bins + 1):
        p = hist[:ci].copy()
        if ci < n_bins:
            p[-1] += np.sum(hist[ci:])
        bw = ci / n_levels
        qh = np.zeros(n_levels, dtype=np.float64)
        for lv in range(n_levels):
            s, e = int(round(lv * bw)), min(int(round((lv + 1) * bw)), ci)
            if s < e:
                qh[lv] = np.sum(p[s:e])
        qe = np.zeros(ci, dtype=np.float64)
        for lv in range(n_levels):
            s, e = int(round(lv * bw)), min(int(round((lv + 1) * bw)), ci)
            if e > s and qh[lv] > 0:
                mask = p[s:e] > 1e-10
                nz = np.sum(mask)
                if nz > 0:
                    qe[s:e] = np.where(mask, qh[lv] / nz, 0)
        qe = np.where(qe == 0, 1e-10, qe)
        pn, qn = p / p.sum(), qe / qe.sum()
        kl = float(np.sum(pn * np.log(pn / qn)))
        if kl < best_kl:
            best_kl, best_idx = kl, ci

    return _quantize_at(tensor, float(edges[best_idx]), qmin, qmax)

METHODS = {"minmax": calibrate_minmax, "mse": calibrate_mse, "kl": calibrate_kl}

def quantize(tensor, n_bits, method="mse"):
    return METHODS[method](tensor, n_bits)

# ── Hex export ───────────────────────────────────────────────────────────────

def write_q88(data, path):
    with open(path, "w") as f:
        for v in data.flatten():
            iv = max(-32768, min(32767, int(round(float(v) * 256))))
            f.write(f"{iv & 0xFFFF:04X}\n")

def write_int8(q, path):
    with open(path, "w") as f:
        for v in q.flatten():
            f.write(f"{np.uint8(v):02X}\n")

def write_int4(q, path):
    with open(path, "w") as f:
        for v in q.flatten():
            f.write(f"{int(v) & 0xF:01X}\n")

def write_int4_packed(q, path):
    flat = q.flatten()
    with open(path, "w") as f:
        for i in range(0, len(flat), 2):
            hi = int(flat[i]) & 0xF
            lo = int(flat[i + 1]) & 0xF if i + 1 < len(flat) else 0
            f.write(f"{(hi << 4) | lo:02X}\n")

def write_scale(scale, path):
    with open(path, "w") as f:
        f.write(f"{scale:.10e}\n")

# ── Per-layer export ─────────────────────────────────────────────────────────

def export_layer(name, param, out_dir, method, report):
    if name not in LAYER_MAP:
        return
    layer, ptype = LAYER_MAP[name]
    vname = f"{layer}_{ptype}"
    data = param.detach().cpu().numpy()
    flat = data.flatten()

    shape = "x".join(str(s) for s in data.shape)
    print(f"  {name} -> {vname}  [{shape}, {flat.size} vals]")

    # Q8.8
    d = os.path.join(out_dir, "q8_8"); os.makedirs(d, exist_ok=True)
    write_q88(data, os.path.join(d, f"{vname}.hex"))

    # INT8
    d = os.path.join(out_dir, "int8"); os.makedirs(d, exist_ok=True)
    q8, s8 = quantize(flat, 8, method)
    write_int8(q8, os.path.join(d, f"{vname}.hex"))
    write_scale(s8, os.path.join(d, f"{vname}_scale.txt"))
    mse8 = float(np.mean((flat - q8.astype(np.float32) * s8) ** 2))
    print(f"    INT8  scale={s8:.8f}  MSE={mse8:.2e}")

    # INT4
    d = os.path.join(out_dir, "int4"); os.makedirs(d, exist_ok=True)
    q4, s4 = quantize(flat, 4, method)
    write_int4(q4, os.path.join(d, f"{vname}.hex"))
    write_int4_packed(q4, os.path.join(d, f"{vname}_packed.hex"))
    write_scale(s4, os.path.join(d, f"{vname}_scale.txt"))
    mse4 = float(np.mean((flat - q4.astype(np.float32) * s4) ** 2))
    print(f"    INT4  scale={s4:.8f}  MSE={mse4:.2e}")

    report[vname] = {"shape": shape, "n": flat.size,
                     "int8_scale": s8, "int8_mse": mse8,
                     "int4_scale": s4, "int4_mse": mse4}

def copy_to_mem(out_dir, mem_dir):
    src = os.path.join(out_dir, "q8_8")
    if not os.path.isdir(src):
        return
    os.makedirs(mem_dir, exist_ok=True)
    n = 0
    for f in os.listdir(src):
        if f.endswith(".hex"):
            open(os.path.join(mem_dir, f), "w").write(
                open(os.path.join(src, f)).read())
            n += 1
    print(f"  Copied {n} Q8.8 hex files -> {mem_dir}/")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="lenet_mnist.pth")
    ap.add_argument("--output", default="exports")
    ap.add_argument("--method", default="mse", choices=["minmax", "mse", "kl"])
    ap.add_argument("--copy-to-mem", action="store_true")
    args = ap.parse_args()

    model = LeNet5()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"LeNet-5  |  {total:,} params  |  calibration: {args.method}")
    print("-" * 60)

    report = {}
    for name, param in model.named_parameters():
        export_layer(name, param, args.output, args.method, report)

    if args.copy_to_mem:
        mem = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mem")
        copy_to_mem(args.output, mem)

    print("\n" + "-" * 60)
    print(f"{'Layer':<18} {'N':>7} {'INT8 MSE':>11} {'INT4 MSE':>11}")
    print("-" * 60)
    for k, v in report.items():
        print(f"{k:<18} {v['n']:>7} {v['int8_mse']:>11.2e} {v['int4_mse']:>11.2e}")
    print("-" * 60)
    print(f"Output -> {args.output}/  (q8_8/ int8/ int4/)")

if __name__ == "__main__":
    main()

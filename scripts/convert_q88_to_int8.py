"""
Convert LeNet-5 weight/bias hex files from Q8.8 (16-bit) to INT8 (8-bit Q0.7).

Quantisation:
   float_value  = int16_q88 / 256.0
   int8_q07     = clip( round(float_value * 128), -128, 127 )

Usage:
   python convert_q88_to_int8.py          # converts all files in-place
   python convert_q88_to_int8.py --backup # saves originals to mem_bak/
"""

import os, sys, shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MEM_DIR = os.path.join(PROJECT_DIR, "mem")
BAK_DIR = os.path.join(PROJECT_DIR, "mem_bak")

# Files to convert (same name for both in & out)
FILES = [
    "conv1_weights.hex",
    "conv1_bias.hex",
    "conv3_weights.hex",
    "conv3_bias.hex",
    "c5_weights.hex",
    "c5_bias.hex",
    "fc1_weights.hex",
    "fc1_bias.hex",
    "fc2_weights.hex",
    "fc2_bias.hex",
]


def convert_q88_to_int8(src_path, dst_path):
    """Read Q8.8 hex, quantise to INT8 (Q0.7), write 2-char hex."""
    with open(src_path, "r") as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        tok = line.strip()
        if not tok or tok.startswith("//"):
            continue
        # Parse unsigned 16-bit hex
        val_u16 = int(tok, 16)
        # Interpret as signed 16-bit (two's complement)
        val_i16 = val_u16 - 0x10000 if val_u16 >= 0x8000 else val_u16
        # Q8.8 -> float
        val_f = val_i16 / 256.0
        # Float -> INT8 (Q0.7): multiply by 128, round, clamp
        val_i8 = int(round(val_f * 128.0))
        val_i8 = max(-128, min(127, val_i8))
        # Unsigned representation for hex output
        val_u8 = val_i8 & 0xFF
        out_lines.append(f"{val_u8:02X}\n")

    with open(dst_path, "w") as f:
        f.writelines(out_lines)

    print(f"  {os.path.basename(src_path):25s}  {len(out_lines):>6d} values  Q8.8->INT8")


def main():
    do_backup = "--backup" in sys.argv

    if do_backup:
        os.makedirs(BAK_DIR, exist_ok=True)
        print(f"Backing up originals to {BAK_DIR}/")

    print(f"Converting Q8.8 -> INT8 (Q0.7) in {MEM_DIR}/\n")

    for fname in FILES:
        src = os.path.join(MEM_DIR, fname)
        if not os.path.exists(src):
            print(f"  WARNING: {fname} not found — skipped")
            continue
        if do_backup:
            shutil.copy2(src, os.path.join(BAK_DIR, fname))
        convert_q88_to_int8(src, src)   # overwrite in place

    print("\nDone.  Re-run Quartus synthesis / ModelSim to use INT8 weights.")


if __name__ == "__main__":
    main()

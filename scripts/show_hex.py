import sys

with open("mem/test_image.hex", "r") as f:
    lines = [l.strip() for l in f.readlines()]

print("test_image.hex shape:")
for r in range(28):
    row_chars = []
    for c in range(28):
        val = int(lines[r*28 + c], 16)
        if val > 128:
            row_chars.append("##")
        elif val > 64:
            row_chars.append("::")
        elif val > 0:
            row_chars.append("..")
        else:
            row_chars.append("  ")
    print("".join(row_chars))

"""Export trained LeNet-5 weights as IEEE 754 hex files and plain text."""

import os
import struct
import torch

from model import LeNet5


def export_float32_hex(model, output_dir="weights_hex"):
    """Export each parameter as IEEE 754 float32 hex values."""
    os.makedirs(output_dir, exist_ok=True)

    for name, param in model.named_parameters():
        filepath = os.path.join(output_dir, f"{name}.txt")
        data = param.detach().cpu().view(-1)

        with open(filepath, "w") as f:
            for val in data:
                hex_bytes = struct.pack('!f', float(val))
                f.write(hex_bytes.hex() + "\n")

        print(f"  {filepath}: {data.numel()} values")


def export_plain_text(model, filepath="weights.txt"):
    """Export all parameters as human-readable plain text."""
    with open(filepath, "w") as f:
        for name, param in model.named_parameters():
            f.write(f"{name}\n{param.data}\n\n")
    print(f"  Saved to {filepath}")


def main():
    model = LeNet5()
    model.load_state_dict(torch.load("lenet_mnist.pth", map_location="cpu"))

    print("Exporting IEEE 754 hex weights...")
    export_float32_hex(model)

    print("\nExporting plain text weights...")
    export_plain_text(model)


if __name__ == "__main__":
    main()

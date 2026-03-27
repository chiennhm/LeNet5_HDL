import os
import torch
import numpy as np

from model import LeNet5
from train import get_data_loaders, evaluate


def quantize_per_tensor(tensor):
    """Quantize a float tensor to INT8 using symmetric per-tensor quantization.

    Returns:
        q (np.ndarray): INT8 quantized values.
        scale (float): Scale factor for dequantization (q * scale ≈ original).
    """
    x_np = tensor.detach().cpu().numpy()
    max_val = np.max(np.abs(x_np))
    scale = max_val / 127 if max_val != 0 else 1.0
    q = np.round(x_np / scale).astype(np.int8)
    return q, scale


def save_int8_hex(q_data, filepath):
    """Save INT8 quantized data as two's complement hex file."""
    with open(filepath, "w") as f:
        for val in q_data.flatten():
            f.write(format(np.uint8(val), '02x') + "\n")


def quantize_model(model, output_dir="quantized"):
    """Quantize all model parameters and save hex + scale files.

    Returns:
        quant_info (dict): Mapping of param name -> (q_data, scale).
    """
    os.makedirs(output_dir, exist_ok=True)
    quant_info = {}

    for name, param in model.named_parameters():
        q, scale = quantize_per_tensor(param)

        save_int8_hex(q, os.path.join(output_dir, f"{name}.txt"))

        scale_path = os.path.join(output_dir, f"{name}_scale.txt")
        with open(scale_path, "w") as f:
            f.write(str(scale))

        quant_info[name] = (q, scale)
        print(f"  {name}: scale={scale:.6f}")

    return quant_info


def load_quantized_to_model(model, quant_info):
    """Load raw INT8 weights into model"""
    for name, param in model.named_parameters():
        q, _ = quant_info[name]
        param.data = torch.tensor(q.astype(np.float32)).view_as(param)
    return model


def main():
    _, test_loader, _ = get_data_loaders()

    # Evaluate FP32 model
    model_fp32 = LeNet5()
    model_fp32.load_state_dict(torch.load("lenet_mnist.pth", map_location="cpu"))

    acc_fp32 = evaluate(model_fp32, test_loader, "cpu")
    print(f"FP32 accuracy: {acc_fp32:.4f}")

    # Quantize
    print("\nQuantizing model...")
    quant_info = quantize_model(model_fp32)

    # Evaluate INT8 (dequantized)
    model_int8 = LeNet5()
    model_int8 = load_quantized_to_model(model_int8, quant_info)

    acc_int8 = evaluate(model_int8, test_loader, "cpu")
    print(f"\nINT8 accuracy: {acc_int8:.4f}")
    print(f"Accuracy drop: {acc_fp32 - acc_int8:.4f}")


if __name__ == "__main__":
    main()

import torch
import numpy as np
from model import LeNet5
from train import get_data_loaders, evaluate

def evaluate_quantization():
    _, _, test_loader = get_data_loaders()
    
    # FP32
    model = LeNet5()
    model.load_state_dict(torch.load("lenet_mnist.pth", map_location="cpu"))
    acc_fp32 = evaluate(model, test_loader, "cpu")
    print(f"FP32: {acc_fp32:.4f}")
    
    # Q8.8 (16-bit fixed point: 8 integer bits + 8 fractional bits)
    model_q88 = LeNet5()
    model_q88.load_state_dict(torch.load("lenet_mnist.pth", map_location="cpu"))
    for name, param in model_q88.named_parameters():
        x_np = param.detach().cpu().numpy()
        q = np.round(x_np * 256.0)
        q = np.clip(q, -32768, 32767)
        param.data = torch.tensor(q / 256.0, dtype=torch.float32).view_as(param)
    acc_q88 = evaluate(model_q88, test_loader, "cpu")
    print(f"Q8.8: {acc_q88:.4f}")
    
    # INT8
    model_int8 = LeNet5()
    model_int8.load_state_dict(torch.load("lenet_mnist.pth", map_location="cpu"))
    for name, param in model_int8.named_parameters():
        x_np = param.detach().cpu().numpy()
        max_val = np.max(np.abs(x_np))
        scale = max_val / 127.0 if max_val != 0 else 1.0
        q = np.round(x_np / scale)
        q = np.clip(q, -128, 127)
        param.data = torch.tensor(q * scale, dtype=torch.float32).view_as(param)
    acc_int8 = evaluate(model_int8, test_loader, "cpu")
    print(f"INT8: {acc_int8:.4f}")
    
    # INT4
    model_int4 = LeNet5()
    model_int4.load_state_dict(torch.load("lenet_mnist.pth", map_location="cpu"))
    for name, param in model_int4.named_parameters():
        x_np = param.detach().cpu().numpy()
        max_val = np.max(np.abs(x_np))
        scale = max_val / 7.0 if max_val != 0 else 1.0
        q = np.round(x_np / scale)
        q = np.clip(q, -8, 7)
        param.data = torch.tensor(q * scale, dtype=torch.float32).view_as(param)
    acc_int4 = evaluate(model_int4, test_loader, "cpu")
    print(f"INT4: {acc_int4:.4f}")

if __name__ == "__main__":
    evaluate_quantization()

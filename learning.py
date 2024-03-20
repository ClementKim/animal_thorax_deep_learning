import torch

def device_check():
    device = ""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}")

    return device

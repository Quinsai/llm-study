import torch

def device_detact():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
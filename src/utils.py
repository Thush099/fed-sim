import torch

def generate_synthetic_data(num_clients: int):
    # Generate synthetic data for each client
    data = [torch.randn(100, 784) for _ in range(num_clients)]
    return data

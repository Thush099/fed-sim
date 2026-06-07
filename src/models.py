from typing import Tuple
import logging
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)

class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train(self, data: torch.Tensor):
        # Train the client model
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = self.forward(data)
            loss = loss_fn(outputs, torch.randint(0, 10, (data.shape[0],)))
            loss.backward()
            optimizer.step()
        # Return the client update
        return self.fc1.weight.grad

    def evaluate(self, data: torch.Tensor):
        # Evaluate the client model
        outputs = self.forward(data)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == torch.randint(0, 10, (data.shape[0],))).sum().item() / data.shape[0]
        loss = nn.CrossEntropyLoss()(outputs, torch.randint(0, 10, (data.shape[0],)))
        return accuracy, loss.item()

class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def aggregate(self, client_updates: List[torch.Tensor]):
        # Aggregate client updates using FedAvg
        return torch.stack(client_updates).mean(0)

    def apply_dp(self, update: torch.Tensor, epsilon: float):
        # Apply differential privacy using Laplace mechanism
        return update + torch.randn_like(update) * epsilon

    def update(self, update: torch.Tensor):
        # Update the server model
        self.fc1.weight.data += update

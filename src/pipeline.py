from typing import List
import logging
from src.models import ClientModel, ServerModel
from src.utils import generate_synthetic_data

logging.basicConfig(level=logging.INFO)

class FederatedLearningPipeline:
    def __init__(self, config: 'Config'):
        self.config = config
        self.clients = [ClientModel() for _ in range(config.num_clients)]
        self.server = ServerModel()

    def run(self):
        for round in range(self.config.num_rounds):
            logging.info(f'Round {round+1}')
            # Generate synthetic data for each client
            data = generate_synthetic_data(self.config.num_clients)
            # Train each client model
            client_updates = [client.train(data[i]) for i, client in enumerate(self.clients)]
            # Aggregate client updates using FedAvg
            server_update = self.server.aggregate(client_updates)
            # Apply differential privacy using Laplace mechanism
            server_update = self.server.apply_dp(server_update, self.config.epsilon)
            # Update server model
            self.server.update(server_update)
            # Evaluate client models
            for i, client in enumerate(self.clients):
                accuracy, loss = client.evaluate(data[i])
                logging.info(f'Client {i+1}: accuracy={accuracy:.4f}, loss={loss:.4f}')

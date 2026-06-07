import argparse
import logging
from src.pipeline import FederatedLearningPipeline
from config import Config

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Simulator')
    parser.add_argument('--num_clients', type=int, help='Number of clients')
    parser.add_argument('--num_rounds', type=int, help='Number of rounds')
    parser.add_argument('--epsilon', type=float, help='Epsilon value for differential privacy')
    args = parser.parse_args()

    config = Config(num_clients=args.num_clients, num_rounds=args.num_rounds, epsilon=args.epsilon)
    pipeline = FederatedLearningPipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()

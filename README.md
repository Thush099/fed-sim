# Federated Learning Simulator
## Problem Statement
Federated learning is a machine learning approach that enables multiple actors to collaborate on model training while maintaining the data private. This project aims to simulate a federated learning environment with differential privacy and per-client drift tracking.
## Architecture
```
                  +---------------+
                  |  Client 1    |
                  +---------------+
                             |
                             |
                             v
                  +---------------+
                  |  Client 2    |
                  +---------------+
                             |
                             |
                             v
                  +---------------+
                  |  Server      |
                  +---------------+
                             |
                             |
                             v
                  +---------------+
                  |  Model       |
                  +---------------+
```
## Installation
To install the project, run the following command:
```bash
pip install -r requirements.txt
```
## Usage
To run the simulator, use the following command:
```bash
python main.py --num_clients 5 --num_rounds 10 --epsilon 1.0
```
This will simulate a federated learning environment with 5 clients, 10 rounds of training, and an epsilon value of 1.0 for differential privacy.
## Sample Output
```
Round 1:
Client 1: accuracy=0.8, loss=0.2
Client 2: accuracy=0.9, loss=0.1
...
Round 10:
Client 1: accuracy=0.95, loss=0.05
Client 2: accuracy=0.95, loss=0.05
...
```
## Design Decisions
The project uses a client-server architecture, where each client trains a local model and sends the updates to the server. The server aggregates the updates using the FedAvg algorithm and applies differential privacy using the Laplace mechanism. The project also tracks the drift of each client's data distribution over time.

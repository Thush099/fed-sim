# fed-sim: Federated Learning Simulator with Differential Privacy

A clean, modular PyTorch simulator for **Federated Averaging (FedAvg)** under **Dirichlet non-IID client splits**, **per-client L2 update clipping**, **calibrated Laplace / Gaussian differential privacy**, and **per-client parameter drift tracking**.

## Architecture

```
                +-------------------------------------------+
                |              FederatedServer              |
                |  - Global TabularClassifier (PyTorch)     |
                |  - Sample-weighted FedAvg aggregation     |
                |  - L2 clipping + Laplace / Gaussian DP    |
                |  - Held-out test evaluation & drift logs  |
                +-------------------------------------------+
                       |                             ^
       1. Broadcast    |                             | 3. Upload clipped/local
       global weights  |                             |    delta_w_k = w_k - w_g
                       v                             |
        +------------------------------+------------------------------+
        |       FederatedClient 0      | ... |     FederatedClient K-1 |
        |  Persistent Dirichlet shard  |     |  Persistent Dirichlet   |
        |  (X_k, y_k) -> local SGD     |     |  shard (X_k, y_k)       |
        +------------------------------+------------------------------+
```

### Key Mathematical Components

1. **Dirichlet Non-IID Partitioning** ([src/utils.py](src/utils.py)):
   Generates a reproducible multi-class tabular classification dataset (`make_classification` with a stratified held-out test set and train-fitted `StandardScaler`). For each class $c$, sample proportions are drawn from $\mathbf{p}_c \sim \text{Dirichlet}(\alpha \cdot \mathbf{1}_K)$:
   - Low $\alpha$ (e.g., `0.2`) yields strongly skewed, non-IID local shards.
   - High $\alpha$ (e.g., `100.0`) approximates balanced IID partitions.

2. **Local Training & Parameter Deltas** ([src/models.py](src/models.py)):
   Each `FederatedClient` receives the broadcast global `state_dict`, trains locally via SGD for `local_epochs` on its persistent $(X_k, y_k)$ shard, and computes the exact parameter update:
   $$\Delta w_k = w_k^{\text{local}} - w^{\text{global}}$$

3. **Per-Client L2 Clipping & Calibrated DP** ([src/privacy.py](src/privacy.py)):
   - **Clipping**: Each client update is clipped to threshold $C$ (`clip_norm`):
     $$\tilde{\Delta}_k = \Delta w_k \cdot \min\left(1,\; \frac{C}{\|\Delta w_k\|_2}\right)$$
   - **FedAvg Aggregation**: Sample-weighted average $\bar{\Delta} = \sum_{k=1}^K \frac{n_k}{N} \tilde{\Delta}_k$ with client-level sensitivity $S = C \cdot \max_k \frac{n_k}{N}$.
   - **Noise Calibration**:
     - **Laplace**: $b = \frac{S}{\epsilon}$, adding i.i.d. $\text{Laplace}(0, b)$ noise per parameter.
     - **Gaussian**: $\sigma = \frac{S \sqrt{2 \ln(1.25 / \delta)}}{\epsilon}$, adding i.i.d. $\mathcal{N}(0, \sigma^2)$ noise per parameter.
   - **Privacy Accounting**: Tracks per-round and cumulative $(\epsilon, \delta)$ using basic and advanced composition bounds.

4. **Client Drift & Dispersion Tracking**:
   Logs per-client parameter drift against the broadcast global model ($\|\Delta w_k\|_2 = \|w_k - w_{\text{global}}\|_2$) and dispersion around the FedAvg center ($\|w_k - w_{\text{fedavg}}\|_2$) on every round.

## Repository Layout

- `config.py` — Validated `SimulationConfig` dataclass with CLI defaults.
- `main.py` — CLI entry point for running simulations.
- `src/utils.py` — Deterministic seeding, dataset generation, and Dirichlet partitioning.
- `src/privacy.py` — L2 norm clipping, Laplace/Gaussian DP calibration, and `PrivacyAccountant`.
- `src/models.py` — `TabularClassifier`, `FederatedClient`, and `FederatedServer`.
- `src/pipeline.py` — `FederatedLearningPipeline` orchestrating broadcast, local training, aggregation, and evaluation.
- `tests/test_fed_sim.py` — Automated pytest suite covering convergence, clipping, DP scaling, drift, and CLI execution.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run with default configuration (`--num-clients 5 --num-rounds 10 --alpha 0.5 --dp-mechanism laplace --epsilon 2.0 --clip-norm 1.0 --seed 42`):

```bash
python main.py
```

Run a custom simulation (e.g., 4 clients, 6 rounds, Laplace DP with $\epsilon = 2.0$):

```bash
python main.py --num-clients 4 --num-rounds 6 --epsilon 2.0
```

### Observed CLI Output

```text
12:28:46 | INFO | Fed-Sim initialized | clients=4 alpha=0.50 skew_tv=0.409 dp=laplace eps=2.00 clip=1.00 sizes=[432, 400, 502, 266] | init_acc=0.1700 init_loss=1.4575
12:28:48 | INFO | Round 01/06 | test_acc=0.2925 test_loss=1.2661 | mean_drift=2.5019 max_drift=2.6979 clipped=4/4 | noise_scale=0.1569 noise_l2=3.794 cum_eps=2.00 | drift=[c0:2.410, c1:2.698, c2:2.571, c3:2.329]
12:28:48 | INFO | Round 02/06 | test_acc=0.6300 test_loss=0.8997 | mean_drift=2.1126 max_drift=2.2882 clipped=4/4 | noise_scale=0.1569 noise_l2=3.536 cum_eps=4.00 | drift=[c0:2.148, c1:2.058, c2:2.288, c3:1.956]
12:28:48 | INFO | Round 03/06 | test_acc=0.7775 test_loss=0.6248 | mean_drift=1.8493 max_drift=2.0132 clipped=4/4 | noise_scale=0.1569 noise_l2=3.797 cum_eps=6.00 | drift=[c0:2.013, c1:1.868, c2:1.799, c3:1.717]
12:28:48 | INFO | Round 04/06 | test_acc=0.6450 test_loss=0.9430 | mean_drift=1.4534 max_drift=1.7604 clipped=4/4 | noise_scale=0.1569 noise_l2=3.872 cum_eps=8.00 | drift=[c0:1.760, c1:1.342, c2:1.566, c3:1.145]
12:28:49 | INFO | Round 05/06 | test_acc=0.8300 test_loss=0.4718 | mean_drift=1.4863 max_drift=1.7736 clipped=4/4 | noise_scale=0.1569 noise_l2=3.497 cum_eps=10.00 | drift=[c0:1.774, c1:1.360, c2:1.758, c3:1.054]
12:28:49 | INFO | Round 06/06 | test_acc=0.8850 test_loss=0.3509 | mean_drift=1.0944 max_drift=1.3293 clipped=3/4 | noise_scale=0.1569 noise_l2=3.315 cum_eps=12.00 | drift=[c0:1.328, c1:1.067, c2:1.329, c3:0.654]

Simulation Complete: round_1_acc=0.2925 -> round_6_acc=0.8850 (loss=0.3509, mean_drift=1.0944, cum_eps=12.00)
```

## Running Tests

```bash
pytest -v
```

## License

MIT License — see [LICENSE](LICENSE).

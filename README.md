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
   - **FedAvg Aggregation**: Sample-weighted average $\bar{\Delta} = \sum_{k=1}^K \frac{n_k}{N} \tilde{\Delta}_k$. For fixed client weights and roster, replacing one clipped client update changes the aggregate by at most $2C \cdot \max_k \frac{n_k}{N}$ in L2.
   - **Noise Calibration**:
     - **Laplace**: Independent coordinate-wise Laplace noise uses the L1 bound $S_1 = \sqrt{d} S_2$, where $d$ is the number of model parameters and $S_2$ is the L2 replacement bound; $b = \frac{S_1}{\epsilon}$.
     - **Gaussian**: The Gaussian mechanism uses the L2 replacement bound $S_2$; $\sigma = \frac{S_2 \sqrt{2 \ln(1.25 / \delta)}}{\epsilon}$.
   - **Privacy Accounting**: Tracks per-round and cumulative $(\epsilon, \delta)$ using basic and advanced composition bounds.
   - **Parameter Validation**: `clip_norm`, `sensitivity`, and `epsilon` must be finite real numbers (and positive where required), the Dirichlet `alpha` must be a finite positive real, and `PrivacyAccountant` rejects unknown mechanisms, non-finite budgets, and invalid `delta_per_round`/`slack_delta`; `NaN`/infinite values are rejected at each boundary instead of silently disabling clipping, noise, or privacy accounting.

4. **Client Drift & Dispersion Tracking**:
   Logs per-client parameter drift against the broadcast global model ($\|\Delta w_k\|_2 = \|w_k - w_{\text{global}}\|_2$) and dispersion around the FedAvg center ($\|w_k - w_{\text{fedavg}}\|_2$) on every round.

## Repository Layout

- `config.py` — Validated `SimulationConfig` dataclass with CLI defaults; rejects non-finite learning/DP settings and invalid dataset-shape fields (`num_samples`, `num_features`, `num_informative`, `num_classes`).
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

Run a utility-oriented demo configuration (e.g., 4 clients, 6 rounds, Laplace DP with $\epsilon = 50.0$):

```bash
python main.py --num-clients 4 --num-rounds 6 --epsilon 50.0
```

### Simulation flags

| Flag | Default | Meaning |
|---|---:|---|
| `--num-clients` | `5` | Number of federated clients |
| `--num-rounds` | `10` | Number of communication rounds |
| `--local-epochs` | `5` | Local SGD epochs per round |
| `--lr` | `0.15` | Client SGD learning rate |
| `--alpha` | `0.5` | Dirichlet concentration (low = non-IID, high = IID) |
| `--dp-mechanism` | `laplace` | `laplace`, `gaussian`, or `none` |
| `--epsilon` | `2.0` | Per-round DP budget epsilon |
| `--delta` | `1e-5` | Privacy failure probability delta (Gaussian mechanism) |
| `--clip-norm` | `1.0` | Per-client L2 update clipping threshold |
| `--seed` | `42` | Random seed |

All numeric flags must be finite: `nan`/`inf` values are rejected at startup instead
of producing a run full of `nan` losses. The server also rejects client updates with
missing/extra parameter keys, mismatched tensor shapes, or non-finite values before
mutating the global model, round counter, or privacy accountant.

Add `--min-accuracy FLOAT` to gate a CI job: the run exits non-zero when the final
round's test accuracy falls below the threshold in `[0, 1]`, and an out-of-range
threshold is rejected rather than silently disabling the gate:

```bash
python main.py --num-clients 4 --num-rounds 6 --epsilon 50.0 --min-accuracy 0.6
```

### Observed CLI Output

```text
10:14:50 | INFO | Fed-Sim initialized | clients=4 alpha=0.50 skew_tv=0.409 dp=laplace eps=50.00 clip=1.00 sizes=[432, 400, 502, 266] | init_acc=0.1700 init_loss=1.4575
10:14:52 | INFO | Round 01/06 | test_acc=0.2850 test_loss=1.4286 | mean_drift=2.5019 max_drift=2.6979 clipped=4/4 | noise_scale=0.2085 noise_l2=5.042 cum_eps=50.00 | drift=[c0:2.410, c1:2.698, c2:2.571, c3:2.329]
10:14:52 | INFO | Round 02/06 | test_acc=0.5000 test_loss=1.1617 | mean_drift=2.1196 max_drift=2.2714 clipped=4/4 | noise_scale=0.2085 noise_l2=4.700 cum_eps=100.00 | drift=[c0:2.246, c1:1.993, c2:2.271, c3:1.969]
10:14:52 | INFO | Round 03/06 | test_acc=0.6250 test_loss=1.1001 | mean_drift=2.0253 max_drift=2.2879 clipped=4/4 | noise_scale=0.2085 noise_l2=5.046 cum_eps=150.00 | drift=[c0:2.288, c1:1.987, c2:1.935, c3:1.892]
10:14:53 | INFO | Round 04/06 | test_acc=0.5700 test_loss=1.6227 | mean_drift=1.7503 max_drift=2.0590 clipped=4/4 | noise_scale=0.2085 noise_l2=5.146 cum_eps=200.00 | drift=[c0:2.059, c1:1.617, c2:1.885, c3:1.440]
10:14:53 | INFO | Round 05/06 | test_acc=0.6400 test_loss=1.2871 | mean_drift=1.8295 max_drift=2.2438 clipped=4/4 | noise_scale=0.2085 noise_l2=4.648 cum_eps=250.00 | drift=[c0:2.244, c1:1.596, c2:2.136, c3:1.342]
10:14:53 | INFO | Round 06/06 | test_acc=0.7450 test_loss=0.7622 | mean_drift=1.5590 max_drift=1.9615 clipped=4/4 | noise_scale=0.2085 noise_l2=4.406 cum_eps=300.00 | drift=[c0:1.799, c1:1.474, c2:1.961, c3:1.001]

Simulation Complete: round_1_acc=0.2850 -> round_6_acc=0.7450 (loss=0.7622, mean_drift=1.5590, cum_eps=300.00)
```

## Running Tests

```bash
pytest -v
```

## License

MIT License — see [LICENSE](LICENSE).

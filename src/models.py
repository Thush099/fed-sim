"""PyTorch models, FederatedClient, and FederatedServer implementations."""

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.privacy import (
    PrivacyAccountant,
    PrivacyStepRecord,
    add_dp_noise,
    clip_update_l2,
    tensor_dict_l2_norm,
)


class TabularClassifier(nn.Module):
    """Lightweight MLP classifier for federated tabular classification."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Sequential(nn.Linear(input_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ClientUpdate:
    """Artifacts returned by a client after one round of local training."""

    client_id: int
    num_samples: int
    delta: Dict[str, torch.Tensor]
    local_state: Dict[str, torch.Tensor]
    pre_clip_drift_l2: float
    train_loss: float
    train_accuracy: float


@dataclass
class RoundMetrics:
    """Aggregated convergence, drift, and privacy metrics for one FL round."""

    round_idx: int
    test_accuracy: float
    test_loss: float
    per_client_drift: Dict[int, float]
    per_client_dispersion: Dict[int, float]
    mean_client_drift: float
    max_client_drift: float
    mean_client_dispersion: float
    clipped_clients: int
    sensitivity: float
    noise_scale: float
    noise_l2_norm: float
    privacy: PrivacyStepRecord


class FederatedClient:
    """Simulates a federated participant holding a persistent local data shard."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        local_epochs: int = 5,
        learning_rate: float = 0.15,
        momentum: float = 0.5,
        weight_decay: float = 1e-3,
        batch_size: int = 64,
        seed: int = 42,
    ) -> None:
        self.client_id = client_id
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self._generator = torch.Generator().manual_seed(seed + client_id + 1)

    @property
    def num_samples(self) -> int:
        return int(self.y_train.shape[0])

    def set_weights(self, global_state: Mapping[str, torch.Tensor]) -> None:
        """Receive broadcast global weights from the server."""
        copied_state = {k: v.detach().clone() for k, v in global_state.items()}
        self.model.load_state_dict(copied_state)

    def train_local(self) -> ClientUpdate:
        """Train locally for ``local_epochs`` and return exact weight delta ``w_local - w_global``."""
        w_global = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        dataset = TensorDataset(self.X_train, self.y_train)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=self._generator,
        )

        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for _ in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_examples = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

                batch_n = int(yb.shape[0])
                epoch_loss += float(loss.item()) * batch_n
                epoch_correct += int((logits.argmax(dim=1) == yb).sum().item())
                epoch_examples += batch_n

            total_loss = epoch_loss / max(epoch_examples, 1)
            total_correct = epoch_correct
            total_examples = epoch_examples

        w_local = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        delta = {k: w_local[k] - w_global[k] for k in w_global}
        drift_l2 = tensor_dict_l2_norm(delta)

        return ClientUpdate(
            client_id=self.client_id,
            num_samples=self.num_samples,
            delta=delta,
            local_state=w_local,
            pre_clip_drift_l2=drift_l2,
            train_loss=total_loss,
            train_accuracy=total_correct / max(total_examples, 1),
        )

    @torch.no_grad()
    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """Evaluate the client's current local model on ``(X, y)``."""
        self.model.eval()
        logits = self.model(X)
        loss = float(nn.CrossEntropyLoss()(logits, y).item())
        accuracy = float((logits.argmax(dim=1) == y).float().mean().item())
        return accuracy, loss


class FederatedServer:
    """Central coordinator performing DP clipping, FedAvg aggregation, broadcast, and drift logging."""

    def __init__(
        self,
        global_model: nn.Module,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        dp_mechanism: str = "laplace",
        epsilon: float = 2.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.global_model = global_model
        self.X_test = X_test
        self.y_test = y_test
        self.dp_mechanism = dp_mechanism
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.round_idx = 0
        self.privacy_accountant = PrivacyAccountant(
            mechanism=dp_mechanism,
            epsilon_per_round=epsilon,
            delta_per_round=delta,
        )
        self._dp_generator = torch.Generator().manual_seed(seed + 10_000)

    def broadcast(self, clients: Sequence[FederatedClient]) -> None:
        """Broadcast current global ``state_dict`` to every client."""
        global_state = {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}
        for client in clients:
            client.set_weights(global_state)

    def aggregate_and_update(self, client_updates: Sequence[ClientUpdate]) -> RoundMetrics:
        """Clip client deltas, aggregate via sample-weighted FedAvg, apply DP noise, and update."""
        if not client_updates:
            raise ValueError("client_updates cannot be empty")
        client_ids = [update.client_id for update in client_updates]
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("client_id values must be unique within a round")
        if any(
            not math.isfinite(update.num_samples) or update.num_samples <= 0
            for update in client_updates
        ):
            raise ValueError("num_samples must be positive and finite for every client update")

        global_state = self.global_model.state_dict()
        expected_keys = set(global_state)
        for update in client_updates:
            if set(update.delta) != expected_keys:
                raise ValueError("delta keys must match global model state")
            for key, value in update.delta.items():
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"delta value for {key!r} must be a tensor")
                if value.shape != global_state[key].shape:
                    raise ValueError(
                        f"delta tensor shape mismatch for {key!r}: "
                        f"expected {tuple(global_state[key].shape)}, got {tuple(value.shape)}"
                    )
                if not torch.isfinite(value).all():
                    raise ValueError("delta tensors must contain only finite values")

        self.round_idx += 1
        prev_global_state = {
            k: v.detach().clone() for k, v in global_state.items()
        }

        total_samples = sum(update.num_samples for update in client_updates)
        if total_samples <= 0:
            raise ValueError("Total client sample count must be positive")
        weights = [update.num_samples / total_samples for update in client_updates]

        clipped_deltas: List[Dict[str, torch.Tensor]] = []
        clipped_count = 0
        per_client_drift: Dict[int, float] = {}

        for update in client_updates:
            clipped, _, was_clipped = clip_update_l2(update.delta, clip_norm=self.clip_norm)
            clipped_deltas.append(clipped)
            if was_clipped:
                clipped_count += 1
            # ||w_k - w_global||_2 measured against the round's broadcast global model.
            per_client_drift[update.client_id] = update.pre_clip_drift_l2

        # Sample-weighted FedAvg aggregation over clipped deltas.
        agg_clean: Dict[str, torch.Tensor] = {}
        for key in prev_global_state:
            acc = torch.zeros_like(prev_global_state[key])
            for weight, c_delta in zip(weights, clipped_deltas):
                acc = acc + weight * c_delta[key]
            agg_clean[key] = acc

        # Un-noised FedAvg center for measuring inter-client dispersion ||w_k - w_fedavg||_2.
        unclipped_mean: Dict[str, torch.Tensor] = {}
        for key in prev_global_state:
            acc = torch.zeros_like(prev_global_state[key])
            for weight, update in zip(weights, client_updates):
                acc = acc + weight * update.delta[key]
            unclipped_mean[key] = acc

        per_client_dispersion: Dict[int, float] = {}
        for update in client_updates:
            diff = {k: update.delta[k] - unclipped_mean[k] for k in prev_global_state}
            per_client_dispersion[update.client_id] = tensor_dict_l2_norm(diff)

        # Replacing one client's clipped update changes the weighted aggregate by at
        # most 2 * clip_norm * max(weight) in L2. Coordinate-wise Laplace noise needs
        # the corresponding L1 bound, which is at most sqrt(dimension) times L2.
        sensitivity = 2.0 * self.clip_norm * max(weights)
        if self.dp_mechanism == "laplace":
            dimension = sum(tensor.numel() for tensor in prev_global_state.values())
            sensitivity *= dimension**0.5
        noisy_agg, noise_scale, noise_l2_norm = add_dp_noise(
            aggregate_delta=agg_clean,
            mechanism=self.dp_mechanism,
            sensitivity=sensitivity,
            epsilon=self.epsilon,
            delta=self.delta,
            generator=self._dp_generator,
        )

        updated_state = {key: prev_global_state[key] + noisy_agg[key] for key in prev_global_state}
        self.global_model.load_state_dict(updated_state)

        test_accuracy, test_loss = self.evaluate_global()
        privacy_record = self.privacy_accountant.step(
            round_idx=self.round_idx,
            sensitivity=sensitivity,
            noise_scale=noise_scale,
        )

        drift_values = list(per_client_drift.values())
        disp_values = list(per_client_dispersion.values())

        return RoundMetrics(
            round_idx=self.round_idx,
            test_accuracy=test_accuracy,
            test_loss=test_loss,
            per_client_drift=per_client_drift,
            per_client_dispersion=per_client_dispersion,
            mean_client_drift=sum(drift_values) / len(drift_values),
            max_client_drift=max(drift_values),
            mean_client_dispersion=sum(disp_values) / len(disp_values),
            clipped_clients=clipped_count,
            sensitivity=sensitivity,
            noise_scale=noise_scale,
            noise_l2_norm=noise_l2_norm,
            privacy=privacy_record,
        )

    @torch.no_grad()
    def evaluate_global(
        self,
        X_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """Evaluate global model accuracy and cross-entropy loss on held-out test data."""
        X_eval = self.X_test if X_test is None else X_test
        y_eval = self.y_test if y_test is None else y_test
        self.global_model.eval()
        logits = self.global_model(X_eval)
        loss = float(nn.CrossEntropyLoss()(logits, y_eval).item())
        accuracy = float((logits.argmax(dim=1) == y_eval).float().mean().item())
        return accuracy, loss

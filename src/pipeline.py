"""End-to-end federated learning simulation pipeline."""

from dataclasses import dataclass
import logging
from typing import List

from config import SimulationConfig
from src.models import (
    ClientUpdate,
    FederatedClient,
    FederatedServer,
    RoundMetrics,
    TabularClassifier,
)
from src.utils import FederatedDataset, generate_federated_dataset, set_seed

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Structured output returned by ``FederatedLearningPipeline.run()``."""

    config: SimulationConfig
    dataset: FederatedDataset
    round_history: List[RoundMetrics]

    @property
    def initial_accuracy(self) -> float:
        return self.round_history[0].test_accuracy

    @property
    def final_accuracy(self) -> float:
        return self.round_history[-1].test_accuracy

    @property
    def final_loss(self) -> float:
        return self.round_history[-1].test_loss


class FederatedLearningPipeline:
    """Orchestrates dataset splitting, client local training, DP-FedAvg, and broadcast."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        set_seed(config.seed)

        self.dataset = generate_federated_dataset(
            num_clients=config.num_clients,
            alpha=config.alpha,
            num_samples=config.num_samples,
            num_features=config.num_features,
            num_informative=config.num_informative,
            num_classes=config.num_classes,
            test_size=config.test_size,
            seed=config.seed,
        )

        global_model = TabularClassifier(
            input_dim=self.dataset.num_features,
            hidden_dim=config.hidden_dim,
            num_classes=self.dataset.num_classes,
        )
        self.server = FederatedServer(
            global_model=global_model,
            X_test=self.dataset.X_test,
            y_test=self.dataset.y_test,
            dp_mechanism=config.dp_mechanism,
            epsilon=config.epsilon,
            delta=config.delta,
            clip_norm=config.clip_norm,
            seed=config.seed,
        )
        self.clients = [
            FederatedClient(
                client_id=client_id,
                model=TabularClassifier(
                    input_dim=self.dataset.num_features,
                    hidden_dim=config.hidden_dim,
                    num_classes=self.dataset.num_classes,
                ),
                X_train=X_k,
                y_train=y_k,
                local_epochs=config.local_epochs,
                learning_rate=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                batch_size=config.batch_size,
                seed=config.seed,
            )
            for client_id, (X_k, y_k) in enumerate(self.dataset.client_splits)
        ]

    def run(self, verbose: bool = True) -> SimulationResult:
        """Run all federated rounds and return convergence, drift, and privacy history."""
        self.server.broadcast(self.clients)
        init_acc, init_loss = self.server.evaluate_global()

        if verbose:
            client_sizes = [client.num_samples for client in self.clients]
            logger.info(
                "Fed-Sim initialized | clients=%d alpha=%.2f skew_tv=%.3f "
                "dp=%s eps=%.2f clip=%.2f sizes=%s | init_acc=%.4f init_loss=%.4f",
                self.config.num_clients,
                self.config.alpha,
                self.dataset.label_skew_tv,
                self.config.dp_mechanism,
                self.config.epsilon,
                self.config.clip_norm,
                client_sizes,
                init_acc,
                init_loss,
            )

        round_history: List[RoundMetrics] = []
        for _ in range(self.config.num_rounds):
            updates: List[ClientUpdate] = [client.train_local() for client in self.clients]
            metrics = self.server.aggregate_and_update(updates)
            self.server.broadcast(self.clients)
            round_history.append(metrics)

            if verbose:
                drift_summary = ", ".join(
                    f"c{cid}:{val:.3f}" for cid, val in sorted(metrics.per_client_drift.items())
                )
                logger.info(
                    "Round %02d/%02d | test_acc=%.4f test_loss=%.4f | "
                    "mean_drift=%.4f max_drift=%.4f clipped=%d/%d | "
                    "noise_scale=%.4f noise_l2=%.3f cum_eps=%.2f | drift=[%s]",
                    metrics.round_idx,
                    self.config.num_rounds,
                    metrics.test_accuracy,
                    metrics.test_loss,
                    metrics.mean_client_drift,
                    metrics.max_client_drift,
                    metrics.clipped_clients,
                    self.config.num_clients,
                    metrics.noise_scale,
                    metrics.noise_l2_norm,
                    metrics.privacy.cumulative_epsilon_composed,
                    drift_summary,
                )

        return SimulationResult(
            config=self.config,
            dataset=self.dataset,
            round_history=round_history,
        )

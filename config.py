"""Configuration dataclasses for the federated learning simulator."""

from dataclasses import dataclass
from typing import Literal

DPMechanism = Literal["laplace", "gaussian", "none"]


@dataclass
class SimulationConfig:
    """Hyperparameters and privacy settings for a DP-FedAvg simulation."""

    num_clients: int = 5
    num_rounds: int = 10
    local_epochs: int = 5
    learning_rate: float = 0.15
    momentum: float = 0.5
    weight_decay: float = 1e-3
    batch_size: int = 64
    alpha: float = 0.5
    dp_mechanism: DPMechanism = "laplace"
    epsilon: float = 2.0
    delta: float = 1e-5
    clip_norm: float = 1.0
    num_samples: int = 2000
    num_features: int = 12
    num_informative: int = 10
    num_classes: int = 4
    hidden_dim: int = 16
    test_size: float = 0.2
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_clients < 1:
            raise ValueError(f"num_clients must be >= 1, got {self.num_clients}")
        if self.num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1, got {self.num_rounds}")
        if self.local_epochs < 1:
            raise ValueError(f"local_epochs must be >= 1, got {self.local_epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.alpha <= 0:
            raise ValueError(f"Dirichlet alpha must be > 0, got {self.alpha}")
        if self.dp_mechanism not in ("laplace", "gaussian", "none"):
            raise ValueError(
                f"dp_mechanism must be 'laplace', 'gaussian', or 'none', got {self.dp_mechanism!r}"
            )
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if not (0.0 < self.delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")
        if self.clip_norm <= 0:
            raise ValueError(f"clip_norm must be > 0, got {self.clip_norm}")
        if not (0.0 < self.test_size < 1.0):
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}")
        if self.num_informative > self.num_features:
            raise ValueError("num_informative cannot exceed num_features")


Config = SimulationConfig

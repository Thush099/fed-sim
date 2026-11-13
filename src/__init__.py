"""Package exports for fed-sim."""

from src.models import (
    ClientUpdate,
    FederatedClient,
    FederatedServer,
    RoundMetrics,
    TabularClassifier,
)
from src.pipeline import FederatedLearningPipeline, SimulationResult
from src.privacy import (
    PrivacyAccountant,
    PrivacyStepRecord,
    add_dp_noise,
    clip_update_l2,
    compute_dp_noise_scale,
    tensor_dict_l2_norm,
)
from src.utils import (
    FederatedDataset,
    compute_client_label_counts,
    compute_label_skew,
    dirichlet_partition,
    generate_federated_dataset,
    set_seed,
)

__all__ = [
    "ClientUpdate",
    "FederatedClient",
    "FederatedDataset",
    "FederatedLearningPipeline",
    "FederatedServer",
    "PrivacyAccountant",
    "PrivacyStepRecord",
    "RoundMetrics",
    "SimulationResult",
    "TabularClassifier",
    "add_dp_noise",
    "clip_update_l2",
    "compute_client_label_counts",
    "compute_dp_noise_scale",
    "compute_label_skew",
    "dirichlet_partition",
    "generate_federated_dataset",
    "set_seed",
    "tensor_dict_l2_norm",
]

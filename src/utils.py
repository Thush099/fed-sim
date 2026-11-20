"""Data generation, Dirichlet non-IID partitioning, and reproducibility helpers."""

from dataclasses import dataclass
import math
from numbers import Real
import random
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class FederatedDataset:
    """Container holding client training shards and the global held-out test set."""

    client_splits: List[Tuple[torch.Tensor, torch.Tensor]]
    X_test: torch.Tensor
    y_test: torch.Tensor
    num_features: int
    num_classes: int
    client_label_counts: np.ndarray
    label_skew_tv: float


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    min_samples_per_client: int = 10,
) -> List[np.ndarray]:
    """Partition sample indices across clients using per-class Dirichlet distributions.

    Lower ``alpha`` (e.g., ``0.2``) produces strongly skewed non-IID client label
    distributions, while higher ``alpha`` (e.g., ``100.0``) approaches IID splits.
    """
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, Real)
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError(f"alpha must be a finite positive real, got {alpha!r}")
    if len(labels) < num_clients:
        raise ValueError("Total number of samples must be at least num_clients")

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    effective_min = min(min_samples_per_client, max(1, len(labels) // (num_clients * 4)))

    best_splits: List[List[int]] = [[] for _ in range(num_clients)]
    for _ in range(50):
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        for cls in classes:
            cls_idx = np.where(labels == cls)[0]
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            counts = rng.multinomial(len(cls_idx), proportions)
            offset = 0
            for client_id in range(num_clients):
                take = int(counts[client_id])
                if take > 0:
                    client_indices[client_id].extend(cls_idx[offset : offset + take].tolist())
                    offset += take

        best_splits = client_indices
        if min(len(shard) for shard in client_indices) >= effective_min:
            break

    # Guarantee no client has zero samples even under extreme alpha values.
    while min(len(shard) for shard in best_splits) == 0:
        empty_id = min(range(num_clients), key=lambda cid: len(best_splits[cid]))
        donor_id = max(range(num_clients), key=lambda cid: len(best_splits[cid]))
        best_splits[empty_id].append(best_splits[donor_id].pop())

    result: List[np.ndarray] = []
    for shard in best_splits:
        arr = np.asarray(shard, dtype=np.int64)
        rng.shuffle(arr)
        result.append(arr)
    return result


def compute_client_label_counts(
    client_splits: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
) -> np.ndarray:
    """Compute a ``(num_clients, num_classes)`` matrix of per-client class counts."""
    counts = np.zeros((len(client_splits), num_classes), dtype=np.int64)
    for client_id, (_, y_client) in enumerate(client_splits):
        labels_np = y_client.detach().cpu().numpy()
        counts[client_id] = np.bincount(labels_np, minlength=num_classes)[:num_classes]
    return counts


def compute_label_skew(client_label_counts: np.ndarray) -> float:
    """Measure mean Total Variation distance between client and global label distributions."""
    total_counts = client_label_counts.sum(axis=0).astype(np.float64)
    global_dist = total_counts / max(total_counts.sum(), 1.0)
    tv_distances: List[float] = []
    for row in client_label_counts:
        row_sum = float(row.sum())
        if row_sum == 0.0:
            continue
        client_dist = row.astype(np.float64) / row_sum
        tv_distances.append(0.5 * float(np.abs(client_dist - global_dist).sum()))
    return float(np.mean(tv_distances)) if tv_distances else 0.0


def generate_federated_dataset(
    num_clients: int,
    alpha: float,
    num_samples: int = 2000,
    num_features: int = 12,
    num_informative: int = 10,
    num_classes: int = 4,
    test_size: float = 0.2,
    seed: int = 42,
) -> FederatedDataset:
    """Generate a reproducible tabular classification dataset and Dirichlet client shards."""
    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=num_informative,
        n_redundant=num_features - num_informative,
        n_classes=num_classes,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    # Fit standardization strictly on training data to avoid leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    shard_indices = dirichlet_partition(
        labels=y_train,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
    )
    client_splits = [
        (
            torch.from_numpy(X_train_scaled[idx]).clone(),
            torch.from_numpy(y_train[idx].astype(np.int64)).clone(),
        )
        for idx in shard_indices
    ]
    label_counts = compute_client_label_counts(client_splits, num_classes=num_classes)
    skew = compute_label_skew(label_counts)

    return FederatedDataset(
        client_splits=client_splits,
        X_test=torch.from_numpy(X_test_scaled),
        y_test=torch.from_numpy(y_test.astype(np.int64)),
        num_features=num_features,
        num_classes=num_classes,
        client_label_counts=label_counts,
        label_skew_tv=skew,
    )

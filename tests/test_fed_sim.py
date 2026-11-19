"""Comprehensive test suite for fed-sim."""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from config import SimulationConfig
from main import main as cli_main
from src.models import ClientUpdate, FederatedClient, FederatedServer, TabularClassifier
from src.pipeline import FederatedLearningPipeline
from src.privacy import (
    PrivacyAccountant,
    add_dp_noise,
    clip_update_l2,
    compute_dp_noise_scale,
    tensor_dict_l2_norm,
)
from src.utils import dirichlet_partition, generate_federated_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_dirichlet_non_iid_vs_iid_skew() -> None:
    """Low Dirichlet alpha creates heavily skewed splits; high alpha approximates IID."""
    low_alpha_ds = generate_federated_dataset(num_clients=5, alpha=0.2, seed=42)
    high_alpha_ds = generate_federated_dataset(num_clients=5, alpha=100.0, seed=42)

    assert low_alpha_ds.label_skew_tv > high_alpha_ds.label_skew_tv
    assert low_alpha_ds.label_skew_tv > 0.30
    assert high_alpha_ds.label_skew_tv < 0.12

    # Verify shards are non-empty and partition all training samples disjointly.
    y = np.repeat(np.arange(4), 100)
    shards = dirichlet_partition(y, num_clients=5, alpha=0.2, seed=7)
    combined = np.concatenate(shards)
    assert len(combined) == len(y)
    assert len(np.unique(combined)) == len(y)
    assert all(len(shard) > 0 for shard in shards)


def test_l2_clipping_enforces_bound() -> None:
    """Per-client L2 norm clipping strictly bounds ||clipped_update||_2 <= clip_norm."""
    large_delta = {
        "layer1.weight": torch.full((16, 12), 2.5),
        "layer1.bias": torch.ones(16) * 1.5,
    }
    for clip_norm in (0.1, 0.5, 1.0):
        clipped, orig_norm, was_clipped = clip_update_l2(large_delta, clip_norm=clip_norm)
        clipped_norm = tensor_dict_l2_norm(clipped)
        assert was_clipped is True
        assert orig_norm > clip_norm
        assert clipped_norm <= clip_norm
        assert math.isclose(clipped_norm, clip_norm, rel_tol=1e-4)

    # Small delta below threshold must remain unclipped.
    small_delta = {"w": torch.full((4,), 0.1)}
    unclipped, small_norm, was_clipped = clip_update_l2(small_delta, clip_norm=1.0)
    assert was_clipped is False
    assert math.isclose(tensor_dict_l2_norm(unclipped), small_norm, rel_tol=1e-6)

    # Zero delta should not produce NaNs.
    zero_delta = {"w": torch.zeros((4, 4))}
    z_clipped, z_norm, z_flag = clip_update_l2(zero_delta, clip_norm=1.0)
    assert z_flag is False
    assert z_norm == 0.0
    assert torch.all(z_clipped["w"] == 0.0)


@pytest.mark.parametrize("mechanism", ["laplace", "gaussian"])
def test_smaller_epsilon_yields_larger_dp_perturbation(mechanism: str) -> None:
    """Verify exact inverse scaling with epsilon for both Laplace and Gaussian DP."""
    sensitivity = 0.25
    delta = 1e-5
    base = {"w": torch.zeros(500, dtype=torch.float32)}

    scale_small_eps = compute_dp_noise_scale(mechanism, sensitivity, epsilon=0.5, delta=delta)
    scale_large_eps = compute_dp_noise_scale(mechanism, sensitivity, epsilon=5.0, delta=delta)
    assert scale_small_eps > scale_large_eps
    assert math.isclose(scale_small_eps / scale_large_eps, 10.0, rel_tol=1e-6)

    if mechanism == "laplace":
        assert math.isclose(scale_small_eps, sensitivity / 0.5, rel_tol=1e-6)
    else:
        expected = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / 0.5
        assert math.isclose(scale_small_eps, expected, rel_tol=1e-6)

    gen_small = torch.Generator().manual_seed(123)
    gen_large = torch.Generator().manual_seed(123)
    _, _, norm_small_eps = add_dp_noise(
        base, mechanism=mechanism, sensitivity=sensitivity, epsilon=0.5, delta=delta, generator=gen_small
    )
    _, _, norm_large_eps = add_dp_noise(
        base, mechanism=mechanism, sensitivity=sensitivity, epsilon=5.0, delta=delta, generator=gen_large
    )
    assert norm_small_eps > norm_large_eps


def test_global_accuracy_improves_and_weights_broadcast() -> None:
    """Global model accuracy strictly improves over rounds and server broadcasts weights."""
    config = SimulationConfig(
        num_clients=4,
        num_rounds=6,
        alpha=0.5,
        dp_mechanism="laplace",
        epsilon=50.0,
        clip_norm=1.0,
        seed=42,
    )
    pipeline = FederatedLearningPipeline(config)
    result = pipeline.run(verbose=False)

    assert len(result.round_history) == 6
    assert result.final_accuracy > result.initial_accuracy
    assert result.final_accuracy > 0.70

    # After final broadcast, all clients must match the global server model exactly.
    global_state = pipeline.server.global_model.state_dict()
    for client in pipeline.clients:
        local_state = client.model.state_dict()
        for key, g_tensor in global_state.items():
            assert torch.allclose(local_state[key], g_tensor)


@pytest.mark.parametrize(
    ("mechanism", "expected_sensitivity"),
    [
        ("laplace", math.sqrt(12.0)),
        ("gaussian", 1.0),
    ],
)
def test_server_uses_mechanism_specific_replacement_sensitivity(
    mechanism: str, expected_sensitivity: float
) -> None:
    model = TabularClassifier(input_dim=2, hidden_dim=2, num_classes=2)
    state = model.state_dict()
    zero_delta = {key: torch.zeros_like(value) for key, value in state.items()}
    updates = [
        ClientUpdate(
            client_id=client_id,
            num_samples=10,
            delta={key: value.clone() for key, value in zero_delta.items()},
            local_state={key: value.clone() for key, value in state.items()},
            pre_clip_drift_l2=0.0,
            train_loss=0.0,
            train_accuracy=0.0,
        )
        for client_id in range(2)
    ]
    server = FederatedServer(
        global_model=model,
        X_test=torch.zeros((2, 2)),
        y_test=torch.zeros(2, dtype=torch.long),
        dp_mechanism=mechanism,
        epsilon=2.0,
        delta=1e-5,
        clip_norm=1.0,
    )

    metrics = server.aggregate_and_update(updates)

    assert metrics.sensitivity == pytest.approx(expected_sensitivity)
    expected_scale = compute_dp_noise_scale(
        mechanism,
        sensitivity=expected_sensitivity,
        epsilon=2.0,
        delta=1e-5,
    )
    assert metrics.noise_scale == pytest.approx(expected_scale)
    assert metrics.privacy.sensitivity == pytest.approx(expected_sensitivity)


def test_server_rejects_nonfinite_client_sample_counts_before_mutating_round_state() -> None:
    model = TabularClassifier(input_dim=2, hidden_dim=2, num_classes=2)
    state = model.state_dict()
    delta = {key: torch.zeros_like(value) for key, value in state.items()}
    update = ClientUpdate(
        client_id=3,
        num_samples=float("nan"),  # type: ignore[arg-type]
        delta=delta,
        local_state=state,
        pre_clip_drift_l2=0.0,
        train_loss=0.0,
        train_accuracy=0.0,
    )
    server = FederatedServer(
        global_model=model,
        X_test=torch.zeros((2, 2)),
        y_test=torch.zeros(2, dtype=torch.long),
        dp_mechanism="none",
    )

    with pytest.raises(ValueError, match="num_samples must be positive and finite"):
        server.aggregate_and_update([update])

    assert server.round_idx == 0
    assert server.privacy_accountant.rounds_completed == 0


def test_server_rejects_duplicate_client_ids_before_mutating_round_state() -> None:
    model = TabularClassifier(input_dim=2, hidden_dim=2, num_classes=2)
    state = model.state_dict()
    delta = {key: torch.zeros_like(value) for key, value in state.items()}
    update = ClientUpdate(
        client_id=3,
        num_samples=5,
        delta=delta,
        local_state=state,
        pre_clip_drift_l2=0.0,
        train_loss=0.0,
        train_accuracy=0.0,
    )
    server = FederatedServer(
        global_model=model,
        X_test=torch.zeros((2, 2)),
        y_test=torch.zeros(2, dtype=torch.long),
        dp_mechanism="none",
    )

    with pytest.raises(ValueError, match="client_id values must be unique"):
        server.aggregate_and_update([update, update])

    assert server.round_idx == 0
    assert server.privacy_accountant.rounds_completed == 0


def test_server_rejects_nonpositive_client_sample_counts_before_mutating_round_state() -> None:
    model = TabularClassifier(input_dim=2, hidden_dim=2, num_classes=2)
    state = model.state_dict()
    delta = {key: torch.zeros_like(value) for key, value in state.items()}
    updates = [
        ClientUpdate(
            client_id=client_id,
            num_samples=num_samples,
            delta={key: value.clone() for key, value in delta.items()},
            local_state={key: value.clone() for key, value in state.items()},
            pre_clip_drift_l2=0.0,
            train_loss=0.0,
            train_accuracy=0.0,
        )
        for client_id, num_samples in enumerate((0, 10))
    ]
    server = FederatedServer(
        global_model=model,
        X_test=torch.zeros((2, 2)),
        y_test=torch.zeros(2, dtype=torch.long),
        dp_mechanism="none",
    )

    with pytest.raises(ValueError, match="num_samples must be positive and finite"):
        server.aggregate_and_update(updates)

    assert server.round_idx == 0
    assert server.privacy_accountant.rounds_completed == 0


def test_gaussian_dp_and_privacy_accountant() -> None:
    """Verify Gaussian DP mode converges under sufficient client count and tracks budget."""
    config = SimulationConfig(
        num_clients=10,
        num_rounds=6,
        alpha=0.5,
        dp_mechanism="gaussian",
        epsilon=3.0,
        delta=1e-4,
        clip_norm=1.0,
        seed=42,
    )
    pipeline = FederatedLearningPipeline(config)
    result = pipeline.run(verbose=False)
    assert result.final_accuracy > result.initial_accuracy

    last_privacy = result.round_history[-1].privacy
    assert last_privacy.cumulative_epsilon_basic == pytest.approx(18.0)
    assert last_privacy.cumulative_delta > 0.0

    # Direct PrivacyAccountant check
    accountant = PrivacyAccountant(mechanism="laplace", epsilon_per_round=1.5)
    rec1 = accountant.step(round_idx=1, sensitivity=0.2, noise_scale=0.2 / 1.5)
    rec2 = accountant.step(round_idx=2, sensitivity=0.2, noise_scale=0.2 / 1.5)
    assert rec1.cumulative_epsilon_basic == pytest.approx(1.5)
    assert rec2.cumulative_epsilon_basic == pytest.approx(3.0)


def test_low_dirichlet_alpha_induces_higher_client_drift() -> None:
    """Non-IID partitioning (alpha=0.2) produces higher client parameter drift than IID (alpha=100.0)."""
    non_iid_cfg = SimulationConfig(
        num_clients=5,
        num_rounds=6,
        alpha=0.2,
        dp_mechanism="laplace",
        epsilon=2.0,
        clip_norm=1.0,
        seed=42,
    )
    iid_cfg = SimulationConfig(
        num_clients=5,
        num_rounds=6,
        alpha=100.0,
        dp_mechanism="laplace",
        epsilon=2.0,
        clip_norm=1.0,
        seed=42,
    )

    non_iid_res = FederatedLearningPipeline(non_iid_cfg).run(verbose=False)
    iid_res = FederatedLearningPipeline(iid_cfg).run(verbose=False)

    non_iid_drift = sum(r.mean_client_drift for r in non_iid_res.round_history) / len(
        non_iid_res.round_history
    )
    iid_drift = sum(r.mean_client_drift for r in iid_res.round_history) / len(
        iid_res.round_history
    )
    non_iid_disp = sum(r.mean_client_dispersion for r in non_iid_res.round_history) / len(
        non_iid_res.round_history
    )
    iid_disp = sum(r.mean_client_dispersion for r in iid_res.round_history) / len(
        iid_res.round_history
    )

    assert non_iid_drift > iid_drift
    assert non_iid_disp > iid_disp


def test_deterministic_reproducibility_and_validation() -> None:
    """Identical seeds produce identical trajectories; invalid configs fail fast."""
    cfg = SimulationConfig(num_clients=4, num_rounds=3, seed=99)
    res1 = FederatedLearningPipeline(cfg).run(verbose=False)
    res2 = FederatedLearningPipeline(cfg).run(verbose=False)

    assert [r.test_accuracy for r in res1.round_history] == [
        r.test_accuracy for r in res2.round_history
    ]
    assert [r.mean_client_drift for r in res1.round_history] == [
        r.mean_client_drift for r in res2.round_history
    ]

    with pytest.raises(ValueError):
        SimulationConfig(alpha=-0.1)
    with pytest.raises(ValueError):
        SimulationConfig(epsilon=0.0)
    with pytest.raises(ValueError):
        SimulationConfig(dp_mechanism="invalid")  # type: ignore[arg-type]


def test_cli_smoke_run() -> None:
    """CLI entrypoint runs cleanly end-to-end with requested flags."""
    exit_code = cli_main(["--num-clients", "4", "--num-rounds", "6", "--epsilon", "2.0"])
    assert exit_code == 0

    proc = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--num-clients",
            "4",
            "--num-rounds",
            "4",
            "--epsilon",
            "2.0",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Simulation Complete:" in proc.stdout

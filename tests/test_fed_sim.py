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
from src.utils import (
    compute_client_label_counts,
    compute_label_skew,
    dirichlet_partition,
    generate_federated_dataset,
    set_seed,
)

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


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), True, "1.0"]
)
def test_compute_dp_noise_scale_rejects_invalid_sensitivity(value: object) -> None:
    with pytest.raises(ValueError, match="sensitivity must be a finite non-negative real"):
        compute_dp_noise_scale("laplace", sensitivity=value, epsilon=1.0)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, True, "1.0"]
)
def test_compute_dp_noise_scale_rejects_invalid_epsilon(value: object) -> None:
    with pytest.raises(ValueError, match="epsilon must be a finite positive real"):
        compute_dp_noise_scale("laplace", sensitivity=1.0, epsilon=value)


@pytest.mark.parametrize("mechanism", ["laplace", "gaussian"])
def test_add_dp_noise_rejects_non_finite_epsilon(mechanism: str) -> None:
    """A non-finite epsilon must not silently disable or corrupt DP noise."""
    delta = {"w": torch.zeros(4)}
    for epsilon in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="epsilon must be a finite positive real"):
            add_dp_noise(delta, mechanism=mechanism, sensitivity=1.0, epsilon=epsilon)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, True, "1.0"]
)
def test_clip_update_l2_rejects_invalid_clip_norm(value: object) -> None:
    with pytest.raises(ValueError, match="clip_norm must be a finite positive real"):
        clip_update_l2({"w": torch.ones(4)}, clip_norm=value)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, True, "0.5"]
)
def test_dirichlet_partition_rejects_invalid_alpha(value: object) -> None:
    with pytest.raises(ValueError, match="alpha must be a finite positive real"):
        dirichlet_partition(np.array([0, 1] * 20), num_clients=2, alpha=value)


def test_dirichlet_partition_accepts_numpy_float_alpha() -> None:
    shards = dirichlet_partition(
        np.array([0, 1] * 20), num_clients=2, alpha=np.float32(0.5), seed=3
    )
    assert sum(len(shard) for shard in shards) == 40


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_key", "delta keys must match global model state"),
        ("wrong_shape", "delta tensor shape mismatch"),
        ("non_finite", "delta tensors must contain only finite values"),
    ],
)
def test_server_rejects_malformed_client_delta_before_state_mutation(
    mutation: str, message: str
) -> None:
    model = TabularClassifier(input_dim=2, hidden_dim=2, num_classes=2)
    state = model.state_dict()
    delta = {key: torch.zeros_like(value) for key, value in state.items()}
    if mutation == "missing_key":
        delta.pop(next(iter(delta)))
    elif mutation == "wrong_shape":
        key = next(iter(delta))
        delta[key] = torch.zeros((1,), dtype=delta[key].dtype)
    else:
        key = next(iter(delta))
        delta[key].view(-1)[0] = float("nan")

    update = ClientUpdate(
        client_id=0,
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
    state_before = {key: value.clone() for key, value in model.state_dict().items()}

    with pytest.raises(ValueError, match=message):
        server.aggregate_and_update([update])

    assert server.round_idx == 0
    assert server.privacy_accountant.rounds_completed == 0
    for key, value in model.state_dict().items():
        assert torch.equal(value, state_before[key])


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


@pytest.mark.parametrize("mechanism", ["lapce", "Laplace", "", "none ", 5, None])
def test_privacy_accountant_rejects_unknown_mechanism(mechanism: object) -> None:
    """A typo'd mechanism must not silently account as Laplace."""
    with pytest.raises(ValueError, match="mechanism must be one of"):
        PrivacyAccountant(mechanism=mechanism, epsilon_per_round=1.0)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), -1.0, True, "1.0"]
)
def test_privacy_accountant_rejects_invalid_epsilon_per_round(value: object) -> None:
    with pytest.raises(ValueError, match="epsilon_per_round must be a finite"):
        PrivacyAccountant(mechanism="laplace", epsilon_per_round=value)


def test_privacy_accountant_rejects_zero_epsilon_for_real_mechanisms() -> None:
    with pytest.raises(ValueError, match="epsilon_per_round must be a finite positive real"):
        PrivacyAccountant(mechanism="laplace", epsilon_per_round=0.0)
    # 'none' releases no noise, so a zero budget is meaningful there.
    accountant = PrivacyAccountant(mechanism="none", epsilon_per_round=0.0)
    assert accountant.cumulative_epsilon_basic == 0.0


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), -0.1, 1.0, True, "0.1"]
)
def test_privacy_accountant_rejects_invalid_delta_per_round(value: object) -> None:
    with pytest.raises(ValueError, match="delta_per_round must be finite and in \[0, 1\)"):
        PrivacyAccountant(
            mechanism="gaussian", epsilon_per_round=1.0, delta_per_round=value
        )


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), 0.0, -0.1, 1.5, True, "0.1"]
)
def test_privacy_accountant_rejects_invalid_slack_delta(value: object) -> None:
    with pytest.raises(ValueError, match="slack_delta must be finite and in \(0, 1\]"):
        PrivacyAccountant(
            mechanism="gaussian", epsilon_per_round=1.0, slack_delta=value
        )


def test_privacy_accountant_composed_bound_handles_large_epsilon() -> None:
    """Advanced composition must not overflow; the basic bound takes over."""
    accountant = PrivacyAccountant(mechanism="gaussian", epsilon_per_round=800.0)
    accountant.step(round_idx=1, sensitivity=1.0, noise_scale=0.5)
    assert accountant.cumulative_epsilon_composed == pytest.approx(800.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sensitivity": float("nan"), "noise_scale": 0.5},
        {"sensitivity": 1.0, "noise_scale": float("inf")},
        {"sensitivity": -1.0, "noise_scale": 0.5},
    ],
)
def test_privacy_accountant_step_rejects_invalid_noise_inputs(kwargs: dict) -> None:
    accountant = PrivacyAccountant(mechanism="laplace", epsilon_per_round=1.0)
    with pytest.raises(ValueError, match="must be a finite non-negative real"):
        accountant.step(round_idx=1, **kwargs)


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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_samples": 0}, "num_samples must be >= 1"),
        ({"num_samples": -5}, "num_samples must be >= 1"),
        ({"num_features": 0}, "num_features must be >= 1"),
        ({"num_informative": 0}, "num_informative must be >= 1"),
        ({"num_classes": 1}, "num_classes must be >= 2"),
        ({"num_classes": -5}, "num_classes must be >= 2"),
    ],
)
def test_config_rejects_invalid_dataset_shape(
    overrides: dict[str, object], message: str
) -> None:
    """Dataset-shape fields are validated at config time, not deep in make_classification."""
    with pytest.raises(ValueError, match=message):
        SimulationConfig(**overrides)  # type: ignore[arg-type]


def test_config_accepts_valid_dataset_shape() -> None:
    for overrides in (
        {"num_samples": 2000, "num_features": 12, "num_informative": 10, "num_classes": 4},
        {"num_samples": 1, "num_features": 1, "num_informative": 1, "num_classes": 2},
    ):
        SimulationConfig(**overrides)  # type: ignore[arg-type]


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


class _FakeResult:
    def __init__(self, final_accuracy: float) -> None:
        self.final_accuracy = final_accuracy


def test_exit_code_no_threshold_always_succeeds() -> None:
    from main import exit_code_for

    assert exit_code_for(_FakeResult(0.0), None) == 0


def test_exit_code_gates_on_min_accuracy() -> None:
    from main import exit_code_for

    assert exit_code_for(_FakeResult(0.5), 0.6) == 1
    assert exit_code_for(_FakeResult(0.6), 0.6) == 0
    assert exit_code_for(_FakeResult(0.9), 0.6) == 0


def test_exit_code_rejects_invalid_threshold() -> None:
    import math

    import pytest

    from main import exit_code_for

    for bad in (float("nan"), float("inf"), float("-inf"), -0.1, 1.5):
        with pytest.raises(ValueError, match="min_accuracy must be a finite value"):
            exit_code_for(_FakeResult(1.0), bad)


def test_cli_min_accuracy_gates_the_run() -> None:
    argv = ["--num-clients", "3", "--num-rounds", "3", "--local-epochs", "2", "--seed", "42"]
    assert cli_main(argv + ["--min-accuracy", "0.0"]) == 0
    assert cli_main(argv + ["--min-accuracy", "1.0"]) == 1


class TestConfigNonFiniteRejection:
    @pytest.mark.parametrize(
        "field",
        ["learning_rate", "momentum", "weight_decay", "alpha", "epsilon", "delta", "clip_norm", "test_size"],
    )
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_config_values_fail_fast(self, field: str, bad: float) -> None:
        from config import SimulationConfig

        with pytest.raises(ValueError, match=f"{field} must be finite"):
            SimulationConfig(**{field: bad})  # type: ignore[arg-type]

    def test_non_finite_epsilon_does_not_run_a_nan_simulation(self) -> None:
        with pytest.raises(ValueError, match="epsilon must be finite"):
            cli_main(["--num-clients", "2", "--num-rounds", "2", "--epsilon", "nan"])


class TestLabelSkewUtilities:
    def test_balanced_clients_have_zero_skew(self) -> None:
        counts = np.array([[10, 10], [10, 10]])
        assert compute_label_skew(counts) == pytest.approx(0.0)

    def test_fully_skewed_two_class_clients_skew_half(self) -> None:
        # Each client is point-mass on one class; global is uniform, so each
        # client's TV distance to the global distribution is 0.5.
        counts = np.array([[20, 0], [0, 20]])
        assert compute_label_skew(counts) == pytest.approx(0.5)

    def test_fully_skewed_three_class_clients_skew_two_thirds(self) -> None:
        counts = np.eye(3) * 10.0
        assert compute_label_skew(counts) == pytest.approx(2.0 / 3.0)

    def test_empty_client_rows_are_skipped(self) -> None:
        counts = np.array([[10, 10], [0, 0]])
        assert compute_label_skew(counts) == pytest.approx(0.0)

    def test_all_empty_clients_yield_zero(self) -> None:
        assert compute_label_skew(np.zeros((3, 4))) == 0.0

    def test_client_label_counts_shape_and_totals(self) -> None:
        splits = [
            (torch.zeros(5, 2), torch.tensor([0, 0, 1, 1, 1])),
            (torch.zeros(4, 2), torch.tensor([1, 1, 1, 1])),
        ]
        counts = compute_client_label_counts(splits, num_classes=2)
        assert counts.shape == (2, 2)
        assert counts[0].tolist() == [2, 3]
        assert counts[1].tolist() == [0, 4]

    def test_set_seed_makes_torch_and_numpy_reproducible(self) -> None:
        set_seed(123)
        a = torch.randn(3).tolist()
        b = np.random.rand(3).tolist()
        set_seed(123)
        assert torch.randn(3).tolist() == a
        assert np.random.rand(3).tolist() == b


def test_noise_scale_dp_contracts() -> None:
    """Noise calibration contracts: the no-noise mode is explicit, invalid deltas and
    unknown mechanisms are rejected, and formulas match the documented ones."""
    import math

    from src.privacy import compute_dp_noise_scale

    assert compute_dp_noise_scale("none", 1.0, 1.0, 1e-5) == 0.0
    for bad_delta in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="delta must be in"):
            compute_dp_noise_scale("laplace", 1.0, 1.0, bad_delta)
    with pytest.raises(ValueError, match="Unsupported DP mechanism"):
        compute_dp_noise_scale("reroute", 1.0, 1.0, 1e-5)
    assert compute_dp_noise_scale("laplace", 2.0, 4.0, 1e-5) == 0.5
    assert compute_dp_noise_scale("gaussian", 1.0, 1.0, 1e-5) == pytest.approx(
        math.sqrt(2.0 * math.log(1.25 / 1e-5))
    )


def test_zero_noise_release_is_an_exact_clone() -> None:
    """mechanism='none' and a zero sensitivity must release the update UNMODIFIED —
    these are the only documented zero-noise paths and both are explicit opt-ins."""
    base = {"w": torch.tensor([1.0, -2.0, 3.5])}

    for mechanism, sensitivity in (("none", 1.0), ("laplace", 0.0)):
        noisy, scale, norm = add_dp_noise(
            base, mechanism=mechanism, sensitivity=sensitivity, epsilon=1.0, delta=1e-5,
        )
        assert scale == 0.0 and norm == 0.0
        assert torch.equal(noisy["w"], base["w"]), f"{mechanism} must not alter the update"
        assert noisy["w"] is not base["w"], "release must be a clone, not the original tensor"


def test_accountant_composition_and_delta_slack_contracts() -> None:
    """Composition accounting: no rounds or mechanism='none' compose to zero, and the
    gaussian advanced-composition path adds exactly the documented slack delta."""
    from src.privacy import PrivacyAccountant

    fresh = PrivacyAccountant(mechanism="laplace", epsilon_per_round=0.5)
    assert fresh.cumulative_epsilon_composed == 0.0

    none_acc = PrivacyAccountant(mechanism="none", epsilon_per_round=0.0)
    none_acc.step(round_idx=1, sensitivity=1.0, noise_scale=0.0)
    assert none_acc.cumulative_epsilon_basic == 0.0
    assert none_acc.cumulative_epsilon_composed == 0.0

    gauss = PrivacyAccountant(
        mechanism="gaussian", epsilon_per_round=0.1, delta_per_round=1e-6, slack_delta=1e-5,
    )
    for i in range(1, 101):
        gauss.step(round_idx=i, sensitivity=1.0, noise_scale=2.0)
    assert gauss.cumulative_epsilon_composed < gauss.cumulative_epsilon_basic, (
        "precondition: the advanced bound must win for the slack path to apply"
    )
    assert gauss.cumulative_delta == pytest.approx(min(1.0, 100 * 1e-6 + 1e-5))

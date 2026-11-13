"""Differential privacy clipping, noise calibration, and privacy accounting."""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Mapping, Optional, Tuple

import torch


def tensor_dict_l2_norm(tensors: Mapping[str, torch.Tensor]) -> float:
    """Compute the global Euclidean (L2) norm across a dictionary of tensors."""
    sq_sum = 0.0
    for tensor in tensors.values():
        sq_sum += float(torch.sum(tensor.detach().to(dtype=torch.float64) ** 2).item())
    return math.sqrt(sq_sum)


def clip_update_l2(
    delta: Mapping[str, torch.Tensor],
    clip_norm: float,
) -> Tuple[Dict[str, torch.Tensor], float, bool]:
    """Clip a parameter update dictionary to have global L2 norm at most ``clip_norm``.

    Args:
        delta: Mapping from parameter names to update tensors.
        clip_norm: Maximum allowed L2 norm threshold ``C > 0``.

    Returns:
        Tuple of ``(clipped_delta, original_l2_norm, was_clipped)``.
    """
    if clip_norm <= 0:
        raise ValueError(f"clip_norm must be positive, got {clip_norm}")

    original_norm = tensor_dict_l2_norm(delta)
    if original_norm <= clip_norm or original_norm == 0.0:
        return {k: v.detach().clone() for k, v in delta.items()}, original_norm, False

    scale = clip_norm / original_norm
    clipped = {k: v.detach().clone() * scale for k, v in delta.items()}

    # Guard against float32 rounding pushing the realized norm infinitesimally above clip_norm.
    post_norm = tensor_dict_l2_norm(clipped)
    if post_norm > clip_norm:
        correction = (clip_norm * (1.0 - 1e-6)) / post_norm
        clipped = {k: v * correction for k, v in clipped.items()}

    return clipped, original_norm, True


def compute_dp_noise_scale(
    mechanism: str,
    sensitivity: float,
    epsilon: float,
    delta: float = 1e-5,
) -> float:
    """Compute calibrated per-coordinate DP noise scale.

    For Laplace: ``scale = sensitivity / epsilon``.
    For Gaussian: ``sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon``.
    """
    if sensitivity < 0:
        raise ValueError(f"sensitivity must be non-negative, got {sensitivity}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    if mechanism == "none":
        return 0.0
    if mechanism == "laplace":
        return sensitivity / epsilon
    if mechanism == "gaussian":
        return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    raise ValueError(f"Unsupported DP mechanism: {mechanism!r}")


def _sample_laplace_like(
    template: torch.Tensor,
    scale: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample i.i.d. Laplace(0, scale) noise via inverse CDF with generator support."""
    if scale == 0.0:
        return torch.zeros_like(template)
    u = (
        torch.rand(
            template.shape,
            dtype=torch.float64,
            device=template.device,
            generator=generator,
        )
        - 0.5
    )
    # Avoid log(0) at exact +/-0.5 endpoints while preserving symmetry.
    u = u.clamp(min=-0.499999999999, max=0.499999999999)
    noise = -scale * torch.sign(u) * torch.log1p(-2.0 * torch.abs(u))
    return noise.to(dtype=template.dtype)


def add_dp_noise(
    aggregate_delta: Mapping[str, torch.Tensor],
    mechanism: str,
    sensitivity: float,
    epsilon: float,
    delta: float = 1e-5,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Dict[str, torch.Tensor], float, float]:
    """Perturb an aggregated update dictionary with calibrated DP noise.

    Returns:
        Tuple of ``(noisy_aggregate, noise_scale, realized_noise_l2_norm)``.
    """
    noise_scale = compute_dp_noise_scale(
        mechanism=mechanism,
        sensitivity=sensitivity,
        epsilon=epsilon,
        delta=delta,
    )
    if mechanism == "none" or noise_scale == 0.0:
        return (
            {k: v.detach().clone() for k, v in aggregate_delta.items()},
            0.0,
            0.0,
        )

    noisy: Dict[str, torch.Tensor] = {}
    noise_tensors: Dict[str, torch.Tensor] = {}
    for key, tensor in aggregate_delta.items():
        if mechanism == "laplace":
            noise = _sample_laplace_like(tensor, noise_scale, generator=generator)
        else:
            noise = (
                torch.randn(
                    tensor.shape,
                    dtype=tensor.dtype,
                    device=tensor.device,
                    generator=generator,
                )
                * noise_scale
            )
        noise_tensors[key] = noise
        noisy[key] = tensor.detach().clone() + noise

    noise_l2_norm = tensor_dict_l2_norm(noise_tensors)
    return noisy, noise_scale, noise_l2_norm


@dataclass(frozen=True)
class PrivacyStepRecord:
    """Privacy expenditure snapshot after a single communication round."""

    round_idx: int
    mechanism: str
    round_epsilon: float
    round_delta: float
    sensitivity: float
    noise_scale: float
    cumulative_epsilon_basic: float
    cumulative_epsilon_composed: float
    cumulative_delta: float


@dataclass
class PrivacyAccountant:
    """Tracks per-round and cumulative differential privacy expenditure across FL rounds."""

    mechanism: str
    epsilon_per_round: float
    delta_per_round: float = 1e-5
    slack_delta: float = 1e-5
    history: List[PrivacyStepRecord] = field(default_factory=list)

    @property
    def rounds_completed(self) -> int:
        return len(self.history)

    @property
    def cumulative_epsilon_basic(self) -> float:
        if self.mechanism == "none":
            return 0.0
        return self.rounds_completed * self.epsilon_per_round

    @property
    def cumulative_epsilon_composed(self) -> float:
        """Return tightest of basic and advanced composition bounds."""
        t = self.rounds_completed
        if t == 0 or self.mechanism == "none":
            return 0.0
        basic = t * self.epsilon_per_round
        if self.mechanism == "gaussian":
            eps = self.epsilon_per_round
            adv = math.sqrt(2.0 * t * math.log(1.0 / self.slack_delta)) * eps + t * eps * (
                math.exp(eps) - 1.0
            )
            return min(basic, adv)
        return basic

    @property
    def cumulative_delta(self) -> float:
        if self.mechanism != "gaussian" or self.rounds_completed == 0:
            return 0.0
        basic_delta = self.rounds_completed * self.delta_per_round
        if self.cumulative_epsilon_composed < self.cumulative_epsilon_basic:
            return min(1.0, basic_delta + self.slack_delta)
        return min(1.0, basic_delta)

    def step(self, round_idx: int, sensitivity: float, noise_scale: float) -> PrivacyStepRecord:
        """Record one round of DP noise release and return the updated budget snapshot."""
        round_eps = 0.0 if self.mechanism == "none" else self.epsilon_per_round
        round_del = self.delta_per_round if self.mechanism == "gaussian" else 0.0
        # Append placeholder first to compute updated totals cleanly.
        self.history.append(
            PrivacyStepRecord(
                round_idx=round_idx,
                mechanism=self.mechanism,
                round_epsilon=round_eps,
                round_delta=round_del,
                sensitivity=sensitivity,
                noise_scale=noise_scale,
                cumulative_epsilon_basic=0.0,
                cumulative_epsilon_composed=0.0,
                cumulative_delta=0.0,
            )
        )
        record = PrivacyStepRecord(
            round_idx=round_idx,
            mechanism=self.mechanism,
            round_epsilon=round_eps,
            round_delta=round_del,
            sensitivity=sensitivity,
            noise_scale=noise_scale,
            cumulative_epsilon_basic=self.cumulative_epsilon_basic,
            cumulative_epsilon_composed=self.cumulative_epsilon_composed,
            cumulative_delta=self.cumulative_delta,
        )
        self.history[-1] = record
        return record

"""CLI entry point for the DP-FedAvg simulator."""

import argparse
import logging
import math
from typing import Optional, Sequence

from config import SimulationConfig
from src.pipeline import FederatedLearningPipeline, SimulationResult


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with sensible federated learning defaults."""
    parser = argparse.ArgumentParser(
        description="Federated Learning Simulator with Dirichlet non-IID splits & Differential Privacy"
    )
    parser.add_argument("--num-clients", type=int, default=5, help="Number of federated clients")
    parser.add_argument("--num-rounds", type=int, default=10, help="Number of communication rounds")
    parser.add_argument("--local-epochs", type=int, default=5, help="Local SGD epochs per round")
    parser.add_argument("--lr", type=float, default=0.15, help="Client SGD learning rate")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet concentration parameter (low=non-IID, high=IID)",
    )
    parser.add_argument(
        "--dp-mechanism",
        type=str,
        choices=["laplace", "gaussian", "none"],
        default="laplace",
        help="Differential privacy noise mechanism",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=2.0,
        help="Per-round differential privacy budget epsilon",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Privacy failure probability delta (used by Gaussian mechanism)",
    )
    parser.add_argument(
        "--clip-norm",
        type=float,
        default=1.0,
        help="Per-client L2 update clipping threshold",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help=(
            "Exit non-zero when the final round's test accuracy is below this "
            "threshold in [0, 1], so the simulation can gate a CI job."
        ),
    )
    return parser


def exit_code_for(result: "SimulationResult", min_accuracy: Optional[float]) -> int:
    """Map a simulation result to a process exit code under an accuracy gate.

    Returns 0 when no threshold is given or the final accuracy meets it, and 1
    when the final round's test accuracy is below the threshold. The threshold
    must be a finite value in [0, 1]; anything else raises ``ValueError`` so a
    typo cannot silently disable gating.
    """
    if min_accuracy is None:
        return 0
    if not math.isfinite(min_accuracy) or not 0.0 <= min_accuracy <= 1.0:
        raise ValueError(
            "min_accuracy must be a finite value in [0, 1]; "
            f"got {min_accuracy!r}"
        )
    return 1 if result.final_accuracy < min_accuracy else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse CLI arguments, run the FL simulation, and print summary metrics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    config = SimulationConfig(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        alpha=args.alpha,
        dp_mechanism=args.dp_mechanism,
        epsilon=args.epsilon,
        delta=args.delta,
        clip_norm=args.clip_norm,
        seed=args.seed,
    )
    pipeline = FederatedLearningPipeline(config)
    result = pipeline.run(verbose=True)

    last = result.round_history[-1]
    print(
        f"\nSimulation Complete: round_1_acc={result.initial_accuracy:.4f} -> "
        f"round_{config.num_rounds}_acc={result.final_accuracy:.4f} "
        f"(loss={result.final_loss:.4f}, mean_drift={last.mean_client_drift:.4f}, "
        f"cum_eps={last.privacy.cumulative_epsilon_composed:.2f})"
    )
    return exit_code_for(result, args.min_accuracy)


if __name__ == "__main__":
    raise SystemExit(main())

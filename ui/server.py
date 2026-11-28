"""Lightweight local web UI server for the DP-FedAvg federated simulator.

Serves the federation console in ``ui/index.html`` and exposes a JSON API that runs the
real ``FederatedLearningPipeline``. Standard library only. Start with
``python ui/server.py`` and open the printed URL.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import SimulationConfig  # noqa: E402
from src.pipeline import FederatedLearningPipeline  # noqa: E402

INDEX_HTML = (REPO_ROOT / "ui" / "index.html").read_text(encoding="utf-8")

DP_MECHANISMS = ("laplace", "gaussian")

# Privacy budgets swept to draw the privacy-utility tradeoff. Lower epsilon = stronger
# privacy = more noise = lower accuracy. Spanning 1..32 shows collapse -> recovery.
DEFAULT_EPSILON_SWEEP = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def run_simulation(
    num_clients: int = 5,
    num_rounds: int = 10,
    local_epochs: int = 3,
    learning_rate: float = 0.15,
    alpha: float = 0.5,
    dp_mechanism: str = "gaussian",
    epsilon: float = 8.0,
    delta: float = 1e-5,
    clip_norm: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a DP-FedAvg simulation and return per-round metrics.

    Raises ValueError with a clear message for invalid parameters, matching the CLI's
    boundary validation.
    """
    if isinstance(num_clients, bool) or not isinstance(num_clients, int):
        raise ValueError(f"num_clients must be an integer, got {num_clients!r}")
    if isinstance(num_rounds, bool) or not isinstance(num_rounds, int):
        raise ValueError(f"num_rounds must be an integer, got {num_rounds!r}")
    if num_clients < 2:
        raise ValueError(f"num_clients must be >= 2, got {num_clients}")
    if num_rounds < 1:
        raise ValueError(f"num_rounds must be >= 1, got {num_rounds}")
    if dp_mechanism not in DP_MECHANISMS:
        raise ValueError(f"dp_mechanism must be one of {list(DP_MECHANISMS)}, got {dp_mechanism!r}")

    config = SimulationConfig(
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        alpha=alpha,
        dp_mechanism=dp_mechanism,
        epsilon=epsilon,
        delta=delta,
        clip_norm=clip_norm,
        seed=seed,
    )
    pipeline = FederatedLearningPipeline(config)
    result = pipeline.run(verbose=False)

    rounds = [
        {
            "round_idx": m.round_idx,
            "test_accuracy": round(m.test_accuracy, 6),
            "test_loss": round(m.test_loss, 6),
            "mean_client_drift": round(m.mean_client_drift, 6),
            "max_client_drift": round(m.max_client_drift, 6),
            "clipped_clients": m.clipped_clients,
            "sensitivity": round(m.sensitivity, 6),
            "noise_scale": round(m.noise_scale, 6),
            "noise_l2_norm": round(m.noise_l2_norm, 6),
        }
        for m in result.round_history
    ]
    return {
        "config": {
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "alpha": alpha,
            "dp_mechanism": dp_mechanism,
            "epsilon": epsilon,
            "delta": delta,
            "clip_norm": clip_norm,
            "seed": seed,
        },
        "initial_accuracy": round(result.initial_accuracy, 6),
        "final_accuracy": round(result.final_accuracy, 6),
        "final_loss": round(result.final_loss, 6),
        "rounds": rounds,
    }


def privacy_utility_curve(
    dp_mechanism: str = "gaussian",
    epsilons: Optional[Sequence[float]] = None,
    num_clients: int = 5,
    num_rounds: int = 5,
    local_epochs: int = 3,
    learning_rate: float = 0.15,
    alpha: float = 0.5,
    delta: float = 1e-5,
    clip_norm: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Sweep the privacy budget (epsilon) and report the resulting model utility.

    Runs the real simulation once per epsilon at a fixed seed and fixed model/data
    config, so the only varying factor is the DP noise level. Lower epsilon means
    stronger privacy, more injected noise, and lower accuracy -- the privacy-utility
    tradeoff this simulator exists to expose. Every point is a genuine run; nothing is
    smoothed or interpolated.

    Raises ValueError with a clear message for invalid mechanism or epsilon values.
    """
    if dp_mechanism not in DP_MECHANISMS:
        raise ValueError(f"dp_mechanism must be one of {list(DP_MECHANISMS)}, got {dp_mechanism!r}")
    if epsilons is None:
        epsilons = DEFAULT_EPSILON_SWEEP
    if not isinstance(epsilons, (list, tuple)) or len(epsilons) < 2:
        raise ValueError("epsilons must be a list of at least 2 positive epsilon values")

    points: list[dict[str, Any]] = []
    for eps in epsilons:
        if (
            isinstance(eps, bool)
            or not isinstance(eps, (int, float))
            or not math.isfinite(eps)
            or eps <= 0
        ):
            raise ValueError(f"each epsilon must be a finite positive number, got {eps!r}")
        run = run_simulation(
            num_clients=num_clients,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            alpha=alpha,
            dp_mechanism=dp_mechanism,
            epsilon=float(eps),
            delta=delta,
            clip_norm=clip_norm,
            seed=seed,
        )
        last = run["rounds"][-1]
        points.append(
            {
                "epsilon": float(eps),
                "final_accuracy": run["final_accuracy"],
                "final_loss": run["final_loss"],
                "noise_scale": last["noise_scale"],
                "noise_l2_norm": last["noise_l2_norm"],
            }
        )
    return {
        "dp_mechanism": dp_mechanism,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "seed": seed,
        "points": points,
    }


MAX_REQUEST_BYTES = 4_000_000


def build_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Create (but do not start) the UI HTTP server bound to ``host:port``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/health":
                self._send_json({"status": "ok", "dp_mechanisms": list(DP_MECHANISMS)})
            else:
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)

        def do_POST(self) -> None:
            if self.path not in ("/api/run", "/api/privacy_curve"):
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_REQUEST_BYTES:
                    self._send_json(
                        {"error": f"request body exceeds {MAX_REQUEST_BYTES} bytes"},
                        status=413,
                    )
                    return
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                if self.path == "/api/run":
                    result = run_simulation(
                        num_clients=payload.get("num_clients", 5),
                        num_rounds=payload.get("num_rounds", 10),
                        local_epochs=payload.get("local_epochs", 3),
                        learning_rate=payload.get("learning_rate", 0.15),
                        alpha=payload.get("alpha", 0.5),
                        dp_mechanism=payload.get("dp_mechanism", "gaussian"),
                        epsilon=payload.get("epsilon", 8.0),
                        delta=payload.get("delta", 1e-5),
                        clip_norm=payload.get("clip_norm", 1.0),
                        seed=payload.get("seed", 42),
                    )
                else:
                    result = privacy_utility_curve(
                        dp_mechanism=payload.get("dp_mechanism", "gaussian"),
                        epsilons=payload.get("epsilons"),
                        num_clients=payload.get("num_clients", 5),
                        num_rounds=payload.get("num_rounds", 5),
                        local_epochs=payload.get("local_epochs", 3),
                        learning_rate=payload.get("learning_rate", 0.15),
                        alpha=payload.get("alpha", 0.5),
                        delta=payload.get("delta", 1e-5),
                        clip_norm=payload.get("clip_norm", 1.0),
                        seed=payload.get("seed", 42),
                    )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the DP-FedAvg federation console UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8129)
    args = parser.parse_args(argv)

    server = build_server(args.host, args.port)
    print(f"DP-FedAvg federation console running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
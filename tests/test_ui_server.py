"""Tests for the DP-FedAvg federation console UI server (ui/server.py)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from ui.server import build_server, privacy_utility_curve, run_simulation


@pytest.fixture()
def server():
    srv = build_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _post(base: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base}/api/run",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request)
        return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_run_endpoint_runs_real_simulation(server):
    status, payload = _post(
        server,
        {"num_clients": 3, "num_rounds": 3, "local_epochs": 1},
    )
    assert status == 200
    assert len(payload["rounds"]) == 3
    assert payload["config"]["num_clients"] == 3
    for metric in payload["rounds"]:
        assert "test_accuracy" in metric and "noise_l2_norm" in metric
        assert "clipped_clients" in metric


def test_result_is_json_serializable(server):
    _, payload = _post(server, {"num_clients": 3, "num_rounds": 2, "local_epochs": 1})
    json.dumps(payload)  # must not raise


def test_health_and_index_served(server):
    health = json.loads(urllib.request.urlopen(server + "/api/health").read())
    assert health["dp_mechanisms"] == ["laplace", "gaussian"]
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert "federation" in html and "/api/run" in html


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"num_clients": "x"}, "num_clients must be an integer"),
        ({"num_clients": 1}, "num_clients must be >= 2"),
        ({"num_rounds": 0}, "num_rounds must be >= 1"),
        ({"num_rounds": True}, "num_rounds must be an integer"),
        ({"dp_mechanism": "magic"}, "dp_mechanism must be one of"),
    ],
)
def test_invalid_inputs_rejected_with_clear_message(server, payload, message):
    status, body = _post(server, payload)
    assert status == 400
    assert message in body["error"]


def test_run_simulation_rejects_bad_type_directly():
    with pytest.raises(ValueError, match="num_clients must be an integer"):
        run_simulation(num_clients="x")

def test_index_has_accessibility_features(server):
    """The served page ships skip-link navigation, a main landmark, and reduced-motion guard."""
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert 'class="skip-link"' in html, "missing skip link for keyboard users"
    assert '<main class="layout">' in html, "missing main landmark"
    assert 'role="alert"' in html, "error box must announce to assistive tech"
    assert "prefers-reduced-motion" in html, "missing reduced-motion guard"


def _post_path(base: str, path: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request)
        return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_privacy_utility_curve_shows_honest_tradeoff():
    """Accuracy must be higher under a weaker privacy budget (larger epsilon)."""
    curve = privacy_utility_curve(
        dp_mechanism="gaussian",
        epsilons=(2.0, 32.0),
        num_clients=3,
        num_rounds=3,
        seed=7,
    )
    assert curve["dp_mechanism"] == "gaussian"
    assert len(curve["points"]) == 2
    low, high = curve["points"]
    assert low["epsilon"] == 2.0 and high["epsilon"] == 32.0
    # Stronger privacy (eps=2) must not beat weaker privacy (eps=32) on accuracy.
    assert high["final_accuracy"] > low["final_accuracy"] + 0.2
    # Lower epsilon injects strictly more noise.
    assert low["noise_l2_norm"] > high["noise_l2_norm"]


def test_privacy_utility_curve_is_json_serializable():
    curve = privacy_utility_curve(dp_mechanism="laplace", epsilons=(1.0, 4.0), num_rounds=2, seed=1)
    assert json.dumps(curve)
    assert set(curve["points"][0]) == {
        "epsilon", "final_accuracy", "final_loss", "noise_scale", "noise_l2_norm"
    }


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"dp_mechanism": "nope"}, "dp_mechanism"),
        ({"epsilons": [2.0, -1.0]}, "finite positive"),
        ({"epsilons": [2.0]}, "at least 2"),
    ],
)
def test_privacy_curve_rejects_bad_input_with_clear_message(server, payload, message):
    status, body = _post_path(server, "/api/privacy_curve", payload)
    assert status == 400
    assert message in body["error"]


def test_privacy_curve_endpoint_returns_real_runs(server):
    status, body = _post_path(
        server, "/api/privacy_curve",
        {"dp_mechanism": "gaussian", "epsilons": [2.0, 32.0], "num_clients": 3, "num_rounds": 3, "seed": 7},
    )
    assert status == 200
    assert len(body["points"]) == 2
    assert body["points"][1]["final_accuracy"] > body["points"][0]["final_accuracy"]


def test_index_has_loading_and_theme_states(server):
    """The page shows busy feedback during a run and declares its dark-first color scheme."""
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert 'aria-busy' in html, "form must announce busy state during a run"
    assert 'class="spinner"' in html, "missing visible busy indicator"
    assert "color-scheme: dark" in html, "dark-first page must declare its color scheme"


def test_oversized_declared_body_is_rejected_not_hung(server):
    """A lying Content-Length must get a fast 413, never park a handler thread."""
    import socket
    from urllib.parse import urlparse

    u = urlparse(server)
    with socket.create_connection((u.hostname, u.port), timeout=5) as sock:
        sock.sendall(
            b"POST /api/run HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n"
            b"Content-Length: 999999999\r\n\r\n{}"
        )
        first = sock.recv(4096).split(b"\r\n", 1)[0]
    assert b"413" in first, f"expected 413 for oversized body, got {first!r}"


def test_index_has_baseline_overlay_controls(server):
    """The tradeoff chart supports pinning a baseline curve for run comparison."""
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert 'id="pin-baseline"' in html, "missing pin-baseline control"
    assert 'id="clear-baseline"' in html, "missing clear-baseline control"
    assert "pinned baseline" in html, "missing the overlay legend"


def test_tradeoff_busy_wiring_is_form_independent(server):
    """computeTradeoff's button lives outside the form; its busy wiring must not
    rely on btn.form (a null reference that silently killed the handler)."""
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    start = html.index("async function computeTradeoff")
    end = html.index('tradeoff-btn").addEventListener')
    handler = html[start:end]
    assert "btn.form" not in handler, "computeTradeoff must not depend on btn.form"
    assert 'querySelector(".spinner")' in handler, "the tradeoff spinner must be scoped to its own row"

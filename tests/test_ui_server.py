"""Tests for the DP-FedAvg federation console UI server (ui/server.py)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from ui.server import build_server, run_simulation


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

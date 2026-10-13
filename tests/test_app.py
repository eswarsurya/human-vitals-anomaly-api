import pytest

from app import create_app

NORMAL = {"heart_rate": 75, "bp_systolic": 120, "bp_diastolic": 80, "temperature": 36.6, "spo2": 98}
EXTREME = {"heart_rate": 150, "bp_systolic": 180, "bp_diastolic": 110, "temperature": 39.5, "spo2": 85}


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["trained_rows"] > 100_000


def test_normal_reading_is_not_anomaly(client):
    p = client.post("/predict", json=NORMAL).get_json()["predictions"][0]
    assert p["is_anomaly"] is False


def test_extreme_reading_is_anomaly(client):
    p = client.post("/predict", json=EXTREME).get_json()["predictions"][0]
    assert p["is_anomaly"] is True


def test_batch_accepts_objects_and_lists(client):
    body = {"data": [NORMAL, [150, 180, 110, 39.5, 85]]}
    r = client.post("/predict", json=body).get_json()
    assert r["total_samples"] == 2
    assert r["anomalies_detected"] == 1


def test_explain_ranks_most_unusual_feature_first(client):
    r = client.post("/explain", json=EXTREME).get_json()
    assert r["contributions"][0]["feature"] == "spo2"


@pytest.mark.parametrize(
    "body, message",
    [
        ({"heart_rate": 75}, "missing"),
        ({**NORMAL, "spo2": "high"}, "must be a number"),
        ({**NORMAL, "temperature": 60}, "outside the possible range"),
        ({"data": []}, "non-empty"),
        ({"data": [[1, 2, 3]]}, "list of 5 numbers"),
    ],
)
def test_bad_input_is_rejected_with_a_clear_message(client, body, message):
    r = client.post("/predict", json=body)
    assert r.status_code == 400
    assert message in r.get_json()["error"]


def test_non_json_body_is_rejected(client):
    r = client.post("/predict", data="not json", content_type="text/plain")
    assert r.status_code == 400


def test_batch_size_is_capped(client):
    r = client.post("/predict", json={"data": [NORMAL] * 1001})
    assert r.status_code == 400

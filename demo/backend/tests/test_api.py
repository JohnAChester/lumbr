import pytest
from fastapi.testclient import TestClient

from conftest import ANSWERABLE
from app.main import create_app

KEY = {"Authorization": "Bearer k"}


@pytest.fixture
def client(retriever):
    return TestClient(create_app(retriever, "k"))


def test_health_open(client):
    body = client.get("/health").json()
    assert body["ok"] is True and body["passages"] == 11 and body["retriever"] == "test"
    assert body["threshold"] == 0.2 and len(body["prompt_sha256"]) == 64


def test_auth_required(client):
    assert client.post("/retrieve", json={"query": ANSWERABLE}).status_code == 401
    assert client.post("/retrieve", json={"query": ANSWERABLE}, headers={"Authorization": "Bearer x"}).status_code == 401
    assert client.post("/passages", json={"ids": []}).status_code == 401


def test_retrieve_ok(client):
    body = client.post("/retrieve", json={"query": ANSWERABLE}, headers=KEY).json()
    assert body["passed"] is True and body["sources"][0]["n"] == 1 and body["messages"][0]["role"] == "system"


@pytest.mark.parametrize("payload", [
    {"query": "   "},
    {"query": "x" * 301},
    {"query": "ok", "k": 0},
    {"query": "ok", "k": 21},
    {"query": "ok", "context_ids": ["dsid_aaa#000"] * 9},
    {"query": "ok", "context_ids": ["dsid_aaa#0"]},
    {"query": "ok", "history": [{"q": "a", "a": "b"}] * 7},
    {"query": "ok", "history": [{"q": "a", "a": "b" * 601}]},
    {"query": "ok", "history": [{"q": "a"}]},
])
def test_validation_errors(client, payload):
    assert client.post("/retrieve", json=payload, headers=KEY).status_code == 422


def test_passages_limits_and_unknown(client):
    assert client.post("/passages", json={"ids": ["dsid_aaa#000"] * 17}, headers=KEY).status_code == 422
    body = client.post("/passages", json={"ids": ["dsid_aaa#000", "dsid_zzz#000", "bad"]}, headers=KEY).json()
    assert [p["passage_id"] for p in body["passages"]] == ["dsid_aaa#000"]

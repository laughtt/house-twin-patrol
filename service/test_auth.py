"""Login, tokens and the maps list. From house-twin-patrol/:
uv run --python 3.12 --with-requirements service/requirements.txt --with pytest --with httpx python -m pytest service/test_auth.py -q
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from passwd import make  # noqa: E402

os.environ.update(API_KEY="test-key", TOKEN_SECRET="test-secret", APP_USER="demo", APP_PASSWORD_HASH=make("test-password"))
os.environ.setdefault("UPLOAD_DIR", "/nonexistent")
import app as service  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(service.app)


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "UPLOADS", tmp_path)
    service.failed.clear()


def login(password="test-password", user="demo"):
    return client.post("/auth/login", json={"user": user, "password": password})


def test_login_ok_and_token_opens_protected_endpoints():
    r = login()
    assert r.status_code == 200 and r.json()["user"] == "demo"
    token = r.json()["token"]
    assert client.get("/lidar/sessions", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    assert client.get("/lidar/sessions", headers={"X-API-Key": token}).status_code == 200  # how the web pages send ?k=
    assert client.get("/lidar/sessions", headers={"X-API-Key": "test-key"}).status_code == 200  # legacy key still works
    assert client.get("/lidar/sessions").status_code == 401


def test_wrong_password_or_user():
    assert login("nope").status_code == 401
    assert login(user="admin").status_code == 401


def test_login_disabled_without_hash(monkeypatch):
    token = login().json()["token"]
    monkeypatch.setattr(service, "APP_PASSWORD_HASH", "")
    assert login().status_code == 401
    assert client.get("/lidar/sessions", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_rate_limit_after_five_failures():
    for _ in range(5):
        assert login("nope").status_code == 401
    assert login().status_code == 429  # even the right password waits
    assert client.post("/auth/login", json={"user": "demo", "password": "test-password"},
                       headers={"X-Forwarded-For": "203.0.113.9"}).status_code == 200  # another client is not blocked


def test_expired_tampered_and_foreign_tokens_rejected(monkeypatch):
    token = login().json()["token"]
    kind, exp, sig = token.split(".")
    past, flip = int(time.time()) - 1, sig[:-1] + ("1" if sig.endswith("0") else "0")
    for bad in (f"{kind}.{past}.{service.sign(past)}", f"{kind}.{int(exp) + 1}.{sig}", f"{kind}.{exp}.{flip}", "garbage"):
        assert client.get("/lidar/sessions", headers={"Authorization": f"Bearer {bad}"}).status_code == 401
    assert not service.token_ok("t1.²³.x")
    monkeypatch.setattr(service, "TOKEN_SECRET", "another-secret")
    assert client.get("/lidar/sessions", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_maps_list_with_states(tmp_path):
    for s in ("lidar-1", "lidar-2", "lidar-3"):
        (tmp_path / s / "house").mkdir(parents=True)
        (tmp_path / s / "lidar.zip").write_bytes(b"zip")
    spaces = {"Kitchen": {"area_m2": 9.5}, "Hall": {"area_m2": 4.25}}
    (tmp_path / "lidar-3" / "house" / "report.json").write_text(json.dumps({"spaces": spaces}))
    (tmp_path / "lidar-2" / "build.log").write_text("Traceback: boom")
    r = client.get("/lidar/maps", headers={"Authorization": f"Bearer {login().json()['token']}"})
    assert r.json() == [{"session": "lidar-3", "state": "done", "rooms": 2, "area_m2": 13.8},
                        {"session": "lidar-2", "state": "failed"}, {"session": "lidar-1", "state": "not built"}]
    assert client.get("/lidar/maps").status_code == 401

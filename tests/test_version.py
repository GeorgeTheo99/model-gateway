"""Version and capability contract for products that attach to this gateway."""

import json
import subprocess
import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

from src import discovery
from src.server import app
from src.version import CAPABILITIES, VERSION

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def test_version_matches_the_project_and_capabilities_are_unique():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert VERSION == project["version"]
    assert len(set(CAPABILITIES)) == len(CAPABILITIES)


def test_health_body_stays_exact_for_older_installers():
    # Home Server 0.4.x verifiers compare the whole body; never add fields here.
    assert client.get("/health").json() == {"status": "ok", "service": "model-gateway"}


def test_admin_status_reports_version_and_capabilities(monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_KEY", "admin-key")
    status = client.get("/admin/api/status", headers={"Authorization": "Bearer admin-key"}).json()
    assert status["version"] == VERSION
    assert status["capabilities"] == dict.fromkeys(CAPABILITIES, True)


def test_endpoint_document_reports_version_and_capabilities(monkeypatch):
    monkeypatch.setattr(discovery.providers, "_load_config", lambda: {})
    document = discovery.endpoint_document()
    assert document["version"] == discovery.ENDPOINT_VERSION == 1
    assert document["gateway_version"] == VERSION
    assert document["capabilities"] == list(CAPABILITIES)


def test_cli_version_works_without_a_running_gateway():
    cli = ROOT / "bin" / "model-gateway"
    plain = subprocess.run([cli, "version"], capture_output=True, text=True, check=True)
    assert plain.stdout.strip() == VERSION
    detailed = json.loads(subprocess.run([cli, "version", "--json"], capture_output=True, text=True, check=True).stdout)
    assert detailed == {"service": "model-gateway", "version": VERSION, "capabilities": list(CAPABILITIES)}

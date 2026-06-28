"""Shared test fixtures.

Isolate the request ledger to a per-test temp database by default so no test
ever writes to the production ledger at ~/srv/cloud-gateway/shared/ledger.db.
Tests that need to inspect the ledger can still override the path.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CLOUD_GATEWAY_LEDGER_PATH", str(tmp_path / "test-ledger.db"))
    yield

"""Model gateway entrypoint."""

import os

import uvicorn

from src.auth import check_bind_safety
from src.server import app  # noqa: F401

if __name__ == "__main__":
    host = os.environ.get("MODEL_GATEWAY_HOST", "127.0.0.1")
    check_bind_safety(host)
    uvicorn.run(
        "src.server:app",
        host=host,
        port=int(os.environ.get("MODEL_GATEWAY_PORT", "9111")),
        log_level="info",
        # Request URLs are not operational logs: they may carry temporary
        # capabilities or future query credentials. The usage ledger provides
        # structured request accounting without recording those URLs.
        access_log=False,
    )

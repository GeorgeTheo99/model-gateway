"""Model gateway entrypoint."""

import os

import uvicorn

from src.server import app  # noqa: F401

if __name__ == "__main__":
    uvicorn.run(
        "src.server:app",
        host=os.environ.get("MODEL_GATEWAY_HOST", "127.0.0.1"),
        port=int(os.environ.get("MODEL_GATEWAY_PORT", "9111")),
        log_level="info",
    )

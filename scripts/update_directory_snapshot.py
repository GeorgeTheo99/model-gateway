#!/usr/bin/env python3
"""Refresh the release's bundled models.dev snapshot (config/directory/models-dev-snapshot.json).

An install serves this snapshot until its first directory sync succeeds
(docs/adr/0005-models-dev-directory.md). Run before a release:

    uv run python scripts/update_directory_snapshot.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import directory  # noqa: E402


def main() -> int:
    corrections = directory.load_corrections()
    api_bytes = directory._download(directory.API_URL)
    labs_bytes = directory._download(directory.LABS_URL)
    plan_bytes = directory._download(directory.PLAN_URL)
    api = {**json.loads(api_bytes), directory.PLAN_PROVIDER: directory.plan_models(json.loads(plan_bytes))}
    snapshot = directory.build_snapshot(api, json.loads(labs_bytes), corrections,
                                        source_sha256=hashlib.sha256(api_bytes + plan_bytes).hexdigest())
    previous = None
    if directory.BUNDLED_SNAPSHOT.exists():
        previous = json.loads(directory.BUNDLED_SNAPSHOT.read_text(encoding="utf-8"))
    directory.validate_update(snapshot, previous)
    directory.BUNDLED_SNAPSHOT.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {directory.BUNDLED_SNAPSHOT} ({directory.model_count(snapshot)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

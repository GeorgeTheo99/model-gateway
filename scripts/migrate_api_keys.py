#!/usr/bin/env python3
"""Move static inline provider API keys from config.yaml into mode-0600 key files.

Run through ``model-gateway secrets migrate`` so the gateway's config, secret,
and backup paths match the running service. Prints provider IDs and key-file
paths only, never key values. Exits 3 when a real run had nothing to move.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

NOTHING_TO_MIGRATE = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    # src.providers resolves CONFIG_PATH from the environment at import time.
    os.environ["MODEL_GATEWAY_CONFIG"] = str(args.config.expanduser())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config_io

    migrated = config_io.migrate_inline_api_keys(dry_run=args.dry_run)
    verb = "would move" if args.dry_run else "moved"
    for row in migrated:
        print(f"{verb} {row['provider']} -> {row['api_key_file']}")
    if not migrated:
        print("no inline static provider keys found")
        return 0 if args.dry_run else NOTHING_TO_MIGRATE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

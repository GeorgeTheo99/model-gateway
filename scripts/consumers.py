#!/usr/bin/env python3
"""Manage identity-aware consumer credentials (``auth.consumer_credentials``).

Run through ``model-gateway consumer`` so the config, client-key, and backup
paths match the running service. Prints credential IDs and key-file paths
only, never key values. Exits 3 when add changed nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

UNCHANGED = 3


def main() -> int:
    parser = argparse.ArgumentParser(prog="model-gateway consumer", description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add", help="create <consumer>-<role> with a generated 0600 key file")
    add.add_argument("consumer", help="consumer id, e.g. myai")
    add.add_argument("--role", required=True, choices=["runtime", "deployer"],
                     help="runtime: profiles read+invoke; deployer: profiles read+write")
    add.add_argument("--namespace", action="append", dest="namespaces",
                     help="profile namespace (repeatable; default: the consumer id)")
    add.add_argument("--allow-direct-models", action="store_true",
                     help="also allow explicit catalog models outside profiles")

    listing = commands.add_parser("list", help="show credentials without key values")
    listing.add_argument("--json", action="store_true")

    revoke = commands.add_parser("revoke", help="remove a credential and delete its key file")
    revoke.add_argument("credential_id", help="credential id, e.g. myai-runtime")

    args = parser.parse_args()
    # src.providers resolves CONFIG_PATH from the environment at import time.
    os.environ["MODEL_GATEWAY_CONFIG"] = str(args.config.expanduser())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config_io

    try:
        if args.command == "list":
            rows = config_io.list_consumer_credentials()
            if args.json:
                print(json.dumps(rows, indent=2))
            elif not rows:
                print("no consumer credentials configured")
            for row in rows if not args.json else ():
                direct = " direct-models" if row["allow_direct_models"] else ""
                print(f"{row['id']:<24} consumer={row['consumer']} namespaces={','.join(row['namespaces'])} "
                      f"permissions={','.join(row['permissions'])}{direct} key={row['key_status']} "
                      f"{row['key_file'] or ''}".rstrip())
            return 0
        if args.command == "add":
            row = config_io.add_consumer_credential(
                args.consumer, args.role, namespaces=args.namespaces,
                allow_direct_models=args.allow_direct_models)
            print(f"{row['status']} {row['id']} -> {row['key_file']}")
            if row["enables_client_auth"]:
                print("warning: /v1 previously accepted requests without a key; it now requires "
                      "a client or consumer key", file=sys.stderr)
            return UNCHANGED if row["status"] == "unchanged" else 0
        row = config_io.revoke_consumer_credential(args.credential_id)
        deleted = "deleted" if row["key_file_deleted"] else "kept"
        print(f"revoked {row['id']} (key file {deleted}: {row['key_file'] or 'none'})")
        if row.get("warning"):
            print(f"warning: {row['warning']}", file=sys.stderr)
        return 0
    except (KeyError, ValueError, OSError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"error: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

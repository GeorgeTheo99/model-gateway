#!/usr/bin/env python3
"""Export or import a secret-free gateway bundle (config, catalog, profiles).

Run through ``model-gateway bundle`` so paths match the running service.
Never reads or prints key values. Import exits 3 for a dry run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from pathlib import Path

DRY_RUN = 3


def main() -> int:
    parser = argparse.ArgumentParser(prog="model-gateway bundle", description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="write a bundle tarball")
    export.add_argument("--out", type=Path,
                        default=Path(f"model-gateway-bundle-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.tar.gz"))
    load = commands.add_parser("import", help="replace providers, catalog, and profiles from a bundle")
    load.add_argument("bundle", type=Path)
    load.add_argument("--force", action="store_true", help="replace existing providers/models/profiles")
    load.add_argument("--dry-run", action="store_true")
    load.add_argument("--rollback-file", type=Path, help=argparse.SUPPRESS)
    undo = commands.add_parser("restore-rollback", help="restore the files saved before a failed import")
    undo.add_argument("rollback_file", type=Path)
    args = parser.parse_args()
    # src.providers resolves CONFIG_PATH from the environment at import time.
    os.environ["MODEL_GATEWAY_CONFIG"] = str(args.config.expanduser())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import logging

    import yaml

    from src import bundle
    from src.profiles import ProfileError

    # Import-time validation loads the server module, which configures INFO logging.
    logging.getLogger().setLevel(logging.WARNING)

    try:
        if args.command == "export":
            manifest = bundle.export_bundle(args.out)
            print(f"wrote {args.out} ({', '.join(manifest['files'])})")
            if manifest["removed"]:
                print(f"left out: {', '.join(manifest['removed'])}")
            return 0
        if args.command == "restore-rollback":
            for restored in bundle.restore_rollback(args.rollback_file):
                print(f"restored {restored}")
            return 0
        summary = bundle.import_bundle(args.bundle, force=args.force, dry_run=args.dry_run,
                                       rollback_file=args.rollback_file)
        print(json.dumps(summary, indent=2))
        for row in summary["missing_key_files"]:
            print(f"warning: {row['owner']} needs key file {row['api_key_file']}", file=sys.stderr)
        return DRY_RUN if args.dry_run else 0
    except (ValueError, RuntimeError, OSError, tarfile.TarError, yaml.YAMLError, ProfileError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

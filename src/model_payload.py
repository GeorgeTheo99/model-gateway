"""Read-only, dependency-free verification against an independently trusted manifest.

Ported from Home Server's verify-model-payload.py for the gateway-owned local
runtime (``src.local_runtime``); also runnable as ``python -m src.model_payload``.

The manifest must be outside the payload directory. Verify a quiescent directory;
this is an integrity check, not a sandbox for concurrently modified files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys


class VerificationError(ValueError):
    """The manifest or payload failed verification."""


# These become a directory name and a download URL, so they must be plain tokens.
MODEL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
HF_REPO_RE = re.compile(r"[\w.-]+/[\w.-]+", re.ASCII)
HF_REVISION_RE = re.compile(r"[0-9a-f]{40}")


def safe_path(value: object) -> str:
    """Require a canonical relative POSIX path, without platform aliases."""
    if (not isinstance(value, str) or not value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
            or "\\" in value or ":" in value
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise VerificationError("invalid relative file path; remove traversal or ambiguous separators")
    return value


def check_path(path: Path, *, directory: bool = False) -> None:
    """Use lstat without resolving away symlinks in any ancestor."""
    path = Path(os.path.join(os.getcwd(), path))
    for ancestor in reversed(path.parents):
        mode = ancestor.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise VerificationError("path ancestor is not a real directory; remove symlinks")
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise VerificationError("symlink found; use real directories and regular files")
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise VerificationError("expected a directory" if directory else "expected a regular file")


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError("duplicate JSON key; regenerate the manifest or index")
        result[key] = value
    return result


def read_json(path: Path, label: str) -> dict:
    try:
        check_path(path)
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=unique_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise VerificationError(f"{label} is not valid JSON") from None
    except OSError:
        raise VerificationError(f"cannot read {label}; check existence and permissions") from None
    if not isinstance(value, dict):
        raise VerificationError(f"{label} must be a JSON object")
    return value


def validate_manifest(manifest: dict) -> dict[str, dict]:
    if (not isinstance(manifest, dict)
            or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1):
        raise VerificationError("manifest schema_version must be 1")
    for field in ("model_id", "hf_repo", "hf_revision", "license"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise VerificationError(f"manifest {field} must be a nonempty string")
    if MODEL_ID_RE.fullmatch(manifest["model_id"]) is None or manifest["model_id"].endswith(".partial"):
        raise VerificationError("manifest model_id must be a plain name of letters, digits, '.', '_', or '-'")
    if HF_REPO_RE.fullmatch(manifest["hf_repo"]) is None:
        raise VerificationError("manifest hf_repo must be 'owner/name'")
    if HF_REVISION_RE.fullmatch(manifest["hf_revision"]) is None:
        raise VerificationError("manifest hf_revision must be a 40-character lowercase commit hash")
    if type(manifest.get("total_bytes")) is not int or manifest["total_bytes"] < 0:
        raise VerificationError("manifest total_bytes must be a nonnegative integer")
    if not isinstance(manifest.get("files"), list) or not manifest["files"]:
        raise VerificationError("manifest files must be a nonempty list")
    files = {}
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise VerificationError("manifest file entry must be an object")
        name = safe_path(entry.get("path"))
        if name in files:
            raise VerificationError(f"duplicate manifest path: {name!r}")
        if type(entry.get("size_bytes")) is not int or entry["size_bytes"] < 0:
            raise VerificationError(f"invalid size_bytes for {name!r}")
        if (not isinstance(entry.get("sha256"), str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", entry["sha256"]) is None):
            raise VerificationError(f"invalid sha256 for {name!r}")
        files[name] = entry
    if sum(entry["size_bytes"] for entry in files.values()) != manifest["total_bytes"]:
        raise VerificationError("manifest total_bytes does not equal file sizes")
    if "model.safetensors.index.json" not in files:
        raise VerificationError("manifest must list model.safetensors.index.json")
    return files


def verify_payload(directory: str | Path, manifest: dict) -> dict:
    """Verify every listed file and index coverage; return a small JSON-ready summary."""
    files = validate_manifest(manifest)
    root = Path(directory)
    try:
        check_path(root, directory=True)
        actual = set()

        def walk_error(error: OSError) -> None:
            raise error

        for current, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
            for name in dirs:
                check_path(Path(current) / name, directory=True)
            for name in names:
                path = Path(current) / name
                check_path(path)
                actual.add(path.relative_to(root).as_posix())
        missing, extra = set(files) - actual, actual - set(files)
        if missing:
            raise VerificationError(f"missing file: {min(missing)!r}")
        if extra:
            raise VerificationError(f"extra file: {min(extra)!r}; remove unlisted files")

        total = 0
        for name, entry in files.items():
            path = root / name
            check_path(path)
            digest = hashlib.sha256()
            size = 0
            # O_NOFOLLOW also prevents following a leaf replaced since lstat.
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise VerificationError(f"not a regular file: {name!r}")
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    digest.update(chunk)
            if size != entry["size_bytes"]:
                raise VerificationError(f"size mismatch: {name!r}; restore the trusted payload")
            if digest.hexdigest() != entry["sha256"].lower():
                raise VerificationError(f"sha256 mismatch: {name!r}; restore the trusted payload")
            total += size
        if total != manifest["total_bytes"]:
            raise VerificationError("payload total_bytes mismatch")

        index = read_json(root / "model.safetensors.index.json", "safetensors index")
        weight_map = index.get("weight_map")
        if (not isinstance(weight_map, dict) or not weight_map
                or any(not key.strip() for key in weight_map)):
            raise VerificationError("index weight_map must be a nonempty object with tensor names")
        references = {safe_path(value) for value in weight_map.values()}
        shards = {name for name in files if name.endswith(".safetensors")}
        if not shards or references != shards:
            raise VerificationError("index weight_map must reference exactly the manifest safetensors shards")
    except OSError:
        raise VerificationError("cannot read payload; check existence, permissions, and file types") from None
    return {"ok": True, "model_id": manifest["model_id"],
            "file_count": len(files), "total_bytes": total}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_payload(args.directory, read_json(args.manifest, "manifest"))
    except VerificationError as error:
        print(f"verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

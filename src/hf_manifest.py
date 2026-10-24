"""Build a verified payload manifest for a Hugging Face MLX repository at a pinned revision.

The manifest (``src.model_payload``) records every file's size and SHA-256.
LFS files carry the SHA-256 Hugging Face reports for the revision; every other
file is downloaded and checked against its Git blob id before it is hashed, so
nothing is trusted on first download. Used to curate ``local-models/`` and for
owner-pinned open models
(docs/adr/0003-scoped-access-and-lightweight-local-models.md section 6).

Only weights, configuration, tokenizer, template and documentation files are
accepted; a repository with code (``*.py``, ``auto_map``, ``trust_remote_code``)
is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request

from src.model_payload import HF_REPO_RE, HF_REVISION_RE, MODEL_ID_RE, VerificationError, safe_path, validate_manifest

HF = "https://huggingface.co"
ALLOWED_SUFFIXES = frozenset({".safetensors", ".json", ".jinja", ".txt", ".model", ".tiktoken", ".md"})
ALLOWED_NAMES = frozenset({".gitattributes"})
CODE_KEYS = ("auto_map", "trust_remote_code")
MAX_SMALL_FILE_BYTES = 64 * 1024 * 1024
MAX_LISTING_PAGES = 20
MAX_FILES = 2000
_NEXT_LINK_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


class ManifestError(ValueError):
    """The repository cannot be pinned as a local model."""


def _opener():
    from src.local_runtime import HttpsOnlyRedirect

    return urllib.request.build_opener(urllib.request.ProxyHandler({}), HttpsOnlyRedirect)


def _get(opener, url: str, *, limit: int) -> tuple[bytes, dict]:
    with opener.open(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=60) as response:
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ManifestError(f"{url} is larger than expected")
        return data, dict(response.headers)


def _tree(opener, repo: str, revision: str) -> list[dict]:
    url = f"{HF}/api/models/{repo}/tree/{revision}?recursive=true&expand=false"
    entries: list[dict] = []
    for _page in range(MAX_LISTING_PAGES):
        data, headers = _get(opener, url, limit=16 * 1024 * 1024)
        page = _json(data, "repository listing")
        if not isinstance(page, list):
            raise ManifestError("Hugging Face returned an unexpected repository listing")
        entries.extend(page)
        if len(entries) > MAX_FILES:
            raise ManifestError("the repository has too many files")
        match = _NEXT_LINK_RE.search(headers.get("Link") or headers.get("link") or "")
        if not match:
            return entries
        url = match.group(1)
        parts, expected = urllib.parse.urlsplit(url), urllib.parse.urlsplit(HF)
        if (parts.scheme, parts.netloc) != (expected.scheme, expected.netloc):
            raise ManifestError("Hugging Face returned an unexpected listing link")
    raise ManifestError("the repository listing has too many pages")


def _json(data: bytes, label: str):
    try:
        return json.loads(data)
    except (UnicodeError, ValueError):
        raise ManifestError(f"Hugging Face returned an invalid {label}") from None


def _allowed(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name in ALLOWED_NAMES or any(name.endswith(suffix) for suffix in ALLOWED_SUFFIXES)


def _blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def build_manifest(repo: str, revision: str, model_id: str, *, opener=None, single_file_ok: bool = False) -> dict:
    """The manifest for ``repo`` at ``revision``; raises ManifestError when it cannot be pinned."""
    if not HF_REPO_RE.fullmatch(repo or ""):
        raise ManifestError("repository must be 'owner/name'")
    if not HF_REVISION_RE.fullmatch(revision or ""):
        raise ManifestError("revision must be a 40-character lowercase commit hash")
    if not MODEL_ID_RE.fullmatch(model_id or "") or model_id.endswith(".partial"):
        raise ManifestError("model id must be a plain name of letters, digits, '.', '_', or '-'")
    opener = opener or _opener()
    info, _ = _get(opener, f"{HF}/api/models/{repo}/revision/{revision}", limit=4 * 1024 * 1024)
    info = _json(info, "model description")
    if not isinstance(info, dict) or info.get("sha") != revision:
        raise ManifestError("Hugging Face did not confirm the pinned revision")
    if info.get("gated") not in (False, None):
        raise ManifestError("gated repositories cannot be pinned")
    license_id = (info.get("cardData") or {}).get("license") if isinstance(info.get("cardData"), dict) else None
    if not isinstance(license_id, str) or not license_id.strip():
        raise ManifestError("the repository declares no license")

    files, refused = [], []
    for entry in _tree(opener, repo, revision):
        if not isinstance(entry, dict) or entry.get("type") != "file":
            continue
        try:
            path = safe_path(entry.get("path"))
        except VerificationError:
            raise ManifestError("the repository has a file with an unsafe path") from None
        if not _allowed(path):
            refused.append(path)
            continue
        size, lfs = entry.get("size"), entry.get("lfs")
        if isinstance(lfs, dict):
            sha256, size = lfs.get("oid"), lfs.get("size")
            if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
                raise ManifestError(f"{path} has no SHA-256 from Hugging Face")
        else:
            if type(size) is not int or size > MAX_SMALL_FILE_BYTES:
                raise ManifestError(f"{path} is too large to hash as a non-LFS file")
            data, _ = _get(opener, f"{HF}/{repo}/resolve/{revision}/{urllib.parse.quote(path)}", limit=size)
            if len(data) != size or _blob_id(data) != entry.get("oid"):
                raise ManifestError(f"{path} did not match its Git blob id")
            if path.endswith(".json"):
                config = _json(data, path)
                if isinstance(config, dict) and any(config.get(key) for key in CODE_KEYS):
                    raise ManifestError(f"{path} requires repository code ({', '.join(CODE_KEYS)})")
            sha256 = hashlib.sha256(data).hexdigest()
        if type(size) is not int or size < 0:
            raise ManifestError(f"{path} has no size")
        files.append({"path": path, "size_bytes": size, "sha256": sha256})
    if refused:
        raise ManifestError(f"the repository has files a local model may not contain: {', '.join(sorted(refused)[:5])}")
    files.sort(key=lambda item: item["path"])
    manifest = {"schema_version": 1, "model_id": model_id, "hf_repo": repo, "hf_revision": revision,
                "license": license_id.strip(), "total_bytes": sum(item["size_bytes"] for item in files),
                "files": files}
    try:
        validate_manifest(manifest, single_file_ok=single_file_ok)
    except VerificationError as exc:
        raise ManifestError(f"the repository is not a supported MLX model: {exc}") from None
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repo")
    parser.add_argument("revision")
    parser.add_argument("model_id")
    args = parser.parse_args(argv)
    try:
        manifest = build_manifest(args.repo, args.revision, args.model_id)
    except (ManifestError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

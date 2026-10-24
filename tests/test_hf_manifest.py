"""Building verified payload manifests from a pinned Hugging Face revision, offline."""

from __future__ import annotations

import hashlib
import json

import pytest

from src import hf_manifest
from src.hf_manifest import ManifestError, build_manifest

REPO, REV = "org/model", "a" * 40
INDEX = json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}}).encode()
CONFIG = json.dumps({"model_type": "qwen"}).encode()
WEIGHTS_SHA = "b" * 64


def blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class Response:
    def __init__(self, data: bytes, headers=None):
        self.data, self.headers = data, headers or {}

    def read(self, limit):
        return self.data[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Host:
    """A fake huggingface.co serving one repository revision."""

    def __init__(self, files: dict[str, bytes], *, lfs: dict[str, tuple[str, int]] | None = None,
                 info: dict | None = None, pages: int = 1):
        self.files, self.lfs = files, lfs or {}
        self.info = info if info is not None else {"sha": REV, "gated": False, "cardData": {"license": "apache-2.0"}}
        self.pages, self.requests = pages, []

    def tree(self) -> list[dict]:
        entries = [{"type": "directory", "path": "sub"}]
        for path, data in self.files.items():
            entries.append({"type": "file", "path": path, "size": len(data), "oid": blob(data)})
        for path, (sha, size) in self.lfs.items():
            entries.append({"type": "file", "path": path, "size": size, "oid": "0" * 40,
                            "lfs": {"oid": sha, "size": size, "pointerSize": 134}})
        return entries

    def open(self, request, timeout):
        url = request.full_url
        self.requests.append(url)
        base = f"{hf_manifest.HF}/api/models/{REPO}"
        if url == f"{base}/revision/{REV}":
            return Response(json.dumps(self.info).encode())
        if url.startswith(f"{base}/tree/{REV}"):
            entries = self.tree()
            if self.pages == 2 and "cursor" not in url:
                return Response(json.dumps(entries[:2]).encode(),
                                {"Link": f'<{base}/tree/{REV}?recursive=true&cursor=2>; rel="next"'})
            return Response(json.dumps(entries[2:] if self.pages == 2 else entries).encode())
        prefix = f"{hf_manifest.HF}/{REPO}/resolve/{REV}/"
        assert url.startswith(prefix), url
        return Response(self.files[url[len(prefix):]])


def host(**kwargs) -> Host:
    files = {"config.json": CONFIG, "model.safetensors.index.json": INDEX, "README.md": b"# model\n"}
    return Host({**files, **kwargs.pop("files", {})},
                lfs={"model-00001-of-00001.safetensors": (WEIGHTS_SHA, 1000), **kwargs.pop("lfs", {})}, **kwargs)


def test_lfs_files_use_the_reported_sha256_and_small_files_are_hashed_after_their_blob_check():
    fake = host(pages=2)
    manifest = build_manifest(REPO, REV, "model-x", opener=fake)
    assert manifest["license"] == "apache-2.0" and manifest["hf_revision"] == REV
    by_path = {entry["path"]: entry for entry in manifest["files"]}
    assert [entry["path"] for entry in manifest["files"]] == sorted(by_path)
    assert by_path["model-00001-of-00001.safetensors"] == {
        "path": "model-00001-of-00001.safetensors", "size_bytes": 1000, "sha256": WEIGHTS_SHA}
    assert by_path["config.json"]["sha256"] == hashlib.sha256(CONFIG).hexdigest()
    assert manifest["total_bytes"] == 1000 + len(CONFIG) + len(INDEX) + len(b"# model\n")
    # LFS weights are never downloaded to build the manifest.
    assert not any("safetensors" in url and "/resolve/" in url and "index" not in url for url in fake.requests)


@pytest.mark.parametrize("change, message", [
    ({"files": {"modeling_qwen.py": b"import os\n"}}, "may not contain: modeling_qwen.py"),
    ({"files": {"config.json": json.dumps({"auto_map": {"AutoModel": "x.Y"}}).encode()}}, "requires repository code"),
    ({"files": {"tokenizer_config.json": json.dumps({"trust_remote_code": True}).encode()}},
     "requires repository code"),
    ({"info": {"sha": REV, "gated": "auto", "cardData": {"license": "mit"}}}, "gated"),
    ({"info": {"sha": REV, "gated": False, "cardData": {}}}, "declares no license"),
    ({"info": {"sha": "c" * 40, "cardData": {"license": "mit"}}}, "did not confirm"),
])
def test_repositories_that_cannot_be_pinned_are_refused(change, message):
    with pytest.raises(ManifestError, match=message):
        build_manifest(REPO, REV, "model-x", opener=host(**change))


def test_a_small_file_that_does_not_match_its_blob_id_is_refused():
    fake = host()
    original = fake.open

    def tampered(request, timeout):
        response = original(request, timeout)
        if request.full_url.endswith("/config.json"):
            response.data = json.dumps({"model_type": "evil"}).encode()[: len(CONFIG)].ljust(len(CONFIG), b" ")
        return response

    fake.open = tampered
    with pytest.raises(ManifestError, match="config.json did not match its Git blob id"):
        build_manifest(REPO, REV, "model-x", opener=fake)


@pytest.mark.parametrize("repo, revision, model_id, message", [
    ("not a repo", REV, "m", "owner/name"),
    (REPO, "main", "m", "40-character"),
    (REPO, REV, "../m", "plain name"),
])
def test_inputs_are_validated_before_any_request(repo, revision, model_id, message):
    with pytest.raises(ManifestError, match=message):
        build_manifest(repo, revision, model_id, opener=object())


def test_a_repository_without_a_sharded_index_is_not_supported():
    fake = Host({"config.json": CONFIG}, lfs={"model.safetensors": (WEIGHTS_SHA, 10)})
    with pytest.raises(ManifestError, match="not a supported MLX model"):
        build_manifest(REPO, REV, "model-x", opener=fake)


def test_the_shipped_manifests_record_what_the_builder_produces():
    """Every curated manifest names a pinned revision, an Apache-2.0 license and only allowed files."""
    from src import local_runtime

    for local in local_runtime.MODELS.values():
        manifest = local_runtime.manifest(local)
        assert manifest["license"] == "apache-2.0"
        assert all(hf_manifest._allowed(entry["path"]) for entry in manifest["files"])
        assert [entry["path"] for entry in manifest["files"]] == sorted(entry["path"] for entry in manifest["files"])


@pytest.mark.parametrize("link", ["http://huggingface.co/api/next", "https://evil.example/api/next"])
def test_listing_links_must_stay_on_https_huggingface(link):
    fake = host()
    original = fake.open

    def linked(request, timeout):
        response = original(request, timeout)
        if "/tree/" in request.full_url:
            response.headers = {"Link": f'<{link}>; rel="next"'}
        return response

    fake.open = linked
    with pytest.raises(ManifestError, match="unexpected listing link"):
        build_manifest(REPO, REV, "model-x", opener=fake)


def test_listing_pages_are_capped():
    fake = host()
    original = fake.open
    base = f"{hf_manifest.HF}/api/models/{REPO}/tree/{REV}"

    def endless(request, timeout):
        if "/tree/" in request.full_url:
            return Response(b"[]", {"Link": f'<{base}?cursor=again>; rel="next"'})
        return original(request, timeout)

    fake.open = endless
    with pytest.raises(ManifestError, match="too many pages"):
        build_manifest(REPO, REV, "model-x", opener=fake)


def test_any_json_file_may_not_require_repository_code():
    fake = host(files={"processor_config.json": json.dumps({"auto_map": {"AutoProcessor": "p.P"}}).encode()})
    with pytest.raises(ManifestError, match="processor_config.json requires repository code"):
        build_manifest(REPO, REV, "model-x", opener=fake)

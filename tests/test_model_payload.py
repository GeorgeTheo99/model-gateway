"""Payload verifier (ported from Home Server): tiny synthetic payloads only."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


from src import model_payload as verifier

ROOT = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "src.model_payload"]


def run(args):
    return subprocess.run([*CLI, *args], capture_output=True, text=True, cwd=ROOT)


class ModelPayloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # macOS temp paths may themselves use /var -> /private/var.
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "model"
        self.root.mkdir()
        (self.root / "shards").mkdir()
        (self.root / "model-00001.safetensors").write_bytes(b"tiny first shard")
        (self.root / "shards/model-00002.safetensors").write_bytes(b"tiny second shard")
        (self.root / "config.json").write_text('{"model_type":"fixture"}', encoding="utf-8")
        self.write_index({"layer.0": "model-00001.safetensors",
                          "layer.1": "shards/model-00002.safetensors"})
        self.manifest = {"schema_version": 1, "model_id": "tiny-model",
                         "hf_repo": "test/tiny-model", "hf_revision": "0123456789abcdef0123456789abcdef01234567",
                         "license": "apache-2.0"}
        self.refresh_manifest()
        self.manifest_path = self.base / "manifest.json"

    def write_index(self, weight_map):
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8")

    def refresh_manifest(self):
        entries = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                data = path.read_bytes()
                entries.append({"path": path.relative_to(self.root).as_posix(),
                                "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        self.manifest["files"] = entries
        self.manifest["total_bytes"] = sum(entry["size_bytes"] for entry in entries)

    def verify(self):
        return verifier.verify_payload(self.root, self.manifest)

    def run_cli(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        return run(["--manifest", str(self.manifest_path), "--directory", str(self.root)])

    def test_good_payload_and_cli_are_read_only(self):
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.root.rglob("*") if p.is_file()}
        expected = {"ok": True, "model_id": "tiny-model", "file_count": 4,
                    "total_bytes": self.manifest["total_bytes"]}
        self.assertEqual(self.verify(), expected)
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(json.loads(result.stdout), expected)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns)
                                  for p in self.root.rglob("*") if p.is_file()})

    def test_corrupt_hash(self):
        path = self.root / "model-00001.safetensors"
        path.write_bytes(b"X" * path.stat().st_size)
        with self.assertRaisesRegex(verifier.VerificationError, "sha256 mismatch"):
            self.verify()
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("sha256 mismatch", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_wrong_file_size(self):
        (self.root / "model-00001.safetensors").write_bytes(b"short")
        with self.assertRaisesRegex(verifier.VerificationError, "size mismatch"):
            self.verify()

    def test_wrong_total(self):
        self.manifest["total_bytes"] += 1
        with self.assertRaisesRegex(verifier.VerificationError, "total_bytes"):
            self.verify()

    def test_extra_file(self):
        (self.root / "shards/extra").write_bytes(b"extra")
        with self.assertRaisesRegex(verifier.VerificationError, "extra file"):
            self.verify()

    def test_missing_file(self):
        (self.root / "shards/model-00002.safetensors").unlink()
        with self.assertRaisesRegex(verifier.VerificationError, "missing file"):
            self.verify()

    def test_symlink_file(self):
        path = self.root / "model-00001.safetensors"
        target = self.base / "external"
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaisesRegex(verifier.VerificationError, "symlink"):
            self.verify()

    def test_symlink_directory(self):
        path = self.root / "shards"
        target = self.base / "external"
        path.rename(target)
        path.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(verifier.VerificationError, "symlink"):
            self.verify()

    def test_symlink_root(self):
        target = self.base / "external"
        self.root.rename(target)
        self.root.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(verifier.VerificationError, "symlink"):
            self.verify()

    def test_symlink_ancestor_even_before_dotdot(self):
        alias = self.base / "alias"
        alias.symlink_to(self.base, target_is_directory=True)
        for root in (alias / "model", alias / ".." / self.base.name / "model"):
            with self.subTest(root=root), self.assertRaisesRegex(verifier.VerificationError, "symlink"):
                verifier.verify_payload(root, self.manifest)

    def test_broken_symlink(self):
        (self.root / "broken").symlink_to(self.base / "missing")
        with self.assertRaisesRegex(verifier.VerificationError, "symlink"):
            self.verify()

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX FIFO")
    def test_nonregular_file(self):
        os.mkfifo(self.root / "fifo")
        with self.assertRaisesRegex(verifier.VerificationError, "regular file"):
            self.verify()

    def test_invalid_relative_paths(self):
        for path in ("", "/absolute", "../outside", "a/../../outside", ".", "..",
                     "a/./b", "./file", "a//b", "a/", "a\\b", "C:/file", "a\x00b", "a\nb"):
            manifest = copy.deepcopy(self.manifest)
            manifest["files"][0]["path"] = path
            with self.subTest(path=path), self.assertRaisesRegex(verifier.VerificationError, "relative file path"):
                verifier.verify_payload(self.root, manifest)

    def test_duplicate_manifest_path(self):
        self.manifest["files"].append(copy.deepcopy(self.manifest["files"][0]))
        with self.assertRaisesRegex(verifier.VerificationError, "duplicate manifest path"):
            self.verify()

    def test_invalid_manifest_fields(self):
        for field, value in (("schema_version", 2), ("schema_version", True), ("model_id", 1),
                             ("hf_repo", ""), ("hf_revision", None), ("license", {}),
                             ("total_bytes", True), ("total_bytes", -1), ("files", []),
                             ("files", {}), ("files", [None])):
            manifest = copy.deepcopy(self.manifest)
            manifest[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(verifier.VerificationError):
                verifier.verify_payload(self.root, manifest)
        for field, value in (("size_bytes", True), ("size_bytes", -1), ("size_bytes", 1.5),
                             ("sha256", "xyz"), ("sha256", None), ("path", 1)):
            manifest = copy.deepcopy(self.manifest)
            manifest["files"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(verifier.VerificationError):
                verifier.verify_payload(self.root, manifest)

    def test_identifiers_that_become_paths_or_urls_must_be_plain_tokens(self):
        for field, value in (("model_id", "../escape"), ("model_id", "a/b"), ("model_id", ".hidden"),
                             ("model_id", "tiny-model.partial"), ("model_id", "x" * 129),
                             ("model_id", "tiny\n"), ("hf_repo", "org"), ("hf_repo", "org/name/extra"),
                             ("hf_repo", "org/na me"), ("hf_repo", "org/n\u00e4me"), ("hf_revision", "main"),
                             ("hf_revision", "0123456789ABCDEF0123456789ABCDEF01234567"),
                             ("hf_revision", "0123456789abcdef0123456789abcdef0123456")):
            manifest = copy.deepcopy(self.manifest)
            manifest[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(verifier.VerificationError):
                verifier.validate_manifest(manifest)
        manifest = copy.deepcopy(self.manifest)
        manifest.update(model_id="Qwen3.8-27B_8bit-30gb", hf_repo="mlx-community/Qwen3.8-27B-8bit")
        verifier.validate_manifest(manifest)

    def test_bad_index_coverage_and_shape(self):
        for weight_map in (None, [], {}, {"": "model-00001.safetensors"},
                           {"a": "model-00001.safetensors"},
                           {"a": "missing.safetensors", "b": "shards/model-00002.safetensors"},
                           {"a": "config.json"}, {"a": "../outside.safetensors"}, {"a": 1}):
            with self.subTest(weight_map=weight_map):
                self.write_index(weight_map)
                self.refresh_manifest()  # Index bytes are trusted; semantic validation must still fail.
                with self.assertRaises(verifier.VerificationError):
                    self.verify()

    def test_repeated_shard_references_are_valid(self):
        self.write_index({"a": "model-00001.safetensors", "b": "model-00001.safetensors",
                          "c": "shards/model-00002.safetensors"})
        self.refresh_manifest()
        self.assertTrue(self.verify()["ok"])

    def test_index_required(self):
        (self.root / "model.safetensors.index.json").unlink()
        self.refresh_manifest()
        with self.assertRaisesRegex(verifier.VerificationError, "must list"):
            self.verify()

    def test_malformed_index_json(self):
        for data in (b"{", b"\xff", b"[]", b'{"weight_map":{},"weight_map":{}}'):
            with self.subTest(data=data):
                (self.root / "model.safetensors.index.json").write_bytes(data)
                self.refresh_manifest()
                with self.assertRaises(verifier.VerificationError):
                    self.verify()

    def test_malformed_manifest_json(self):
        for data in (b"{private contents", b"\xff", b"[]", b'{"schema_version":1,"schema_version":1}'):
            with self.subTest(data=data):
                self.manifest_path.write_bytes(data)
                result = run(["--manifest", str(self.manifest_path), "--directory", str(self.root)])
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertNotIn("private contents", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_missing_directory(self):
        with self.assertRaisesRegex(verifier.VerificationError, "cannot read payload"):
            verifier.verify_payload(self.base / "missing", self.manifest)

    def test_cli_requires_arguments(self):
        for args in ([], ["--directory", str(self.root)], ["--manifest", str(self.manifest_path)]):
            with self.subTest(args=args):
                result = run(args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("required", result.stderr)


if __name__ == "__main__":
    unittest.main()

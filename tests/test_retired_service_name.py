from pathlib import Path
import re
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    + "cloud"
    + r"(?:[-_ ]*)"
    + "gateway"
    + r"(?![A-Za-z0-9])"
    + r"|(?<![A-Za-z0-9])"
    + "cloud_"
    + "gw"
    + r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
EXPECTED_NOTICE = (
    "`model-gateway` is the canonical gateway service. `cloud-"
    "gateway` is retired; do not recreate legacy checkouts, bare repos, "
    "LaunchAgents, or environment variables."
)
Finding = tuple[str, int | None, str]


def _repository_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.fail(result.stderr.decode("utf-8", errors="replace"))
    return [
        path.decode("utf-8", errors="surrogateescape")
        for path in result.stdout.split(b"\0")
        if path
    ]


def _scan_repository(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for relative_path in _repository_paths(root):
        if SERVICE_PATTERN.search(relative_path):
            findings.append((relative_path, None, "forbidden tracked or untracked path"))

        path = root / relative_path
        if not path.is_file() or path.is_symlink():
            continue
        content = path.read_bytes()
        if b"\0" in content:
            continue
        for line_number, line in enumerate(
            content.decode("utf-8", errors="replace").splitlines(), start=1
        ):
            if SERVICE_PATTERN.search(line):
                findings.append((relative_path, line_number, line))
    return findings


def test_retired_service_name_is_limited_to_canonical_notice() -> None:
    assert _scan_repository(REPO_ROOT) == [
        ("docs/deployment.md", 3, EXPECTED_NOTICE)
    ]


def test_scanner_covers_paths_untracked_content_and_name_variants(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    legacy_name = "cloud" + "-gateway"
    legacy_env = "CLOUD_GATE" + "WAY_CONFIG=1"
    legacy_camel = "CloudGate" + "way"
    legacy_abbreviation = "CLOUD_" + "GW"

    forbidden_path = tmp_path / f"com.local.{legacy_name}.plist"
    forbidden_path.write_text("safe\n", encoding="utf-8")
    (tmp_path / "untracked.env").write_text(legacy_env + "\n", encoding="utf-8")
    (tmp_path / "abbreviation.env").write_text(
        legacy_abbreviation + "=1\n", encoding="utf-8"
    )
    (tmp_path / "camel.txt").write_text(legacy_camel + "\n", encoding="utf-8")
    (tmp_path / "safe.txt").write_text("CLOUD_GWEN_CONFIG=1\n", encoding="utf-8")

    assert _scan_repository(tmp_path) == [
        ("abbreviation.env", 1, legacy_abbreviation + "=1"),
        ("camel.txt", 1, legacy_camel),
        (forbidden_path.name, None, "forbidden tracked or untracked path"),
        ("untracked.env", 1, legacy_env),
    ]

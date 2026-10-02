#!/bin/bash
# Verify a Model Gateway component package (or a product archive holding only it)
# without executing anything from the package: the payload is checked with this
# checkout's postinstall, which must be byte-identical to the package's.

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDENTIFIER="com.local.model-gateway.component"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ $# -eq 1 ] || { printf 'usage: %s PACKAGE.pkg\n' "$0" >&2; exit 2; }
PKG="$1"
[ -f "$PKG" ] && [ ! -L "$PKG" ] || fail "not a package file: $PKG"
for tool in pkgutil plutil python3 shasum; do
  command -v "$tool" >/dev/null 2>&1 || fail "missing required tool: $tool"
done

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/model-gateway-verify.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT
pkgutil --expand-full "$PKG" "$WORK_DIR/expanded" >/dev/null
COMPONENT="$WORK_DIR/expanded"
if [ ! -f "$COMPONENT/PackageInfo" ]; then
  # A product archive: exactly one component, ours.
  components=()
  while IFS= read -r info; do components+=("$(dirname "$info")"); done \
    < <(find "$COMPONENT" -mindepth 2 -maxdepth 2 -name PackageInfo -type f -print)
  [ "${#components[@]}" -eq 1 ] || fail "expected exactly one component package, found ${#components[@]}"
  COMPONENT="${components[0]}"
fi
SUPPORT="$COMPONENT/Payload/Library/Application Support/ModelGateway"
RELEASE_PLIST="$SUPPORT/package/release.plist"
[ -f "$RELEASE_PLIST" ] || fail "package has no release metadata"
VERSION="$(plutil -extract package_version raw -o - "$RELEASE_PLIST")"

python3 - "$COMPONENT/PackageInfo" "$IDENTIFIER" "$VERSION" <<'PY'
import sys
import xml.etree.ElementTree as ET

path, identifier, version = sys.argv[1:]
info = ET.parse(path).getroot()
if info.get("identifier") != identifier:
    raise SystemExit(f"package identifier is {info.get('identifier')!r}, not {identifier!r}")
if info.get("version") != version:
    raise SystemExit("package version differs from its release metadata")
if info.get("install-location", "/") != "/":
    raise SystemExit("package must install at /")
if info.find("./scripts/postinstall") is None or info.find("./scripts/preinstall") is not None:
    raise SystemExit("package scripts must be exactly a postinstall")
PY
[ "$(cd "$COMPONENT/Payload" && find . -mindepth 1 -maxdepth 3 -print | LC_ALL=C sort)" = \
  "$(printf '%s\n' ./Library './Library/Application Support' './Library/Application Support/ModelGateway')" ] \
  || fail "payload installs outside /Library/Application Support/ModelGateway"
[ "$(cd "$COMPONENT/Scripts" && find . -mindepth 1 -print)" = "./postinstall" ] || fail "package scripts must be exactly postinstall"
cmp -s "$COMPONENT/Scripts/postinstall" "$SCRIPT_DIR/postinstall" || \
  fail "package postinstall differs from this checkout's; verify with the release's own checkout"
"$SCRIPT_DIR/postinstall" --verify-payload "$SUPPORT" >/dev/null
printf 'verified: %s (%s %s, release %s)\n' "$PKG" "$IDENTIFIER" "$VERSION" \
  "$(plutil -extract release_name raw -o - "$RELEASE_PLIST")"

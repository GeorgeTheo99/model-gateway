#!/bin/bash
# Build the unsigned Model Gateway component package (com.local.model-gateway.component).

set -euo pipefail
umask 022

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_TREE="$(cd "$SCRIPT_DIR/../../.." && pwd)"
IDENTIFIER="com.local.model-gateway.component"
REF="HEAD"
VERSION=""
OUT_DIR=""
DRY_RUN=0
PRODUCT=0
HELPER_TIMEOUT=1500
POSTINSTALL_TIMEOUT=1560
# Everything the installed gateway needs at runtime, and nothing else.
RUNTIME_PATHS=(LICENSE README.md bin/model-gateway config local-models local-runtime pyproject.toml scripts src uv.lock)

usage() {
  cat <<'USAGE'
Usage: packaging/component/scripts/build-component-pkg.sh --version X --out-dir DIR [options]

Builds an unsigned Installer component package from a clean, committed
revision. The payload stages the gateway's runtime files, the per-user helper,
a manifest, and release metadata in /Library/Application Support/ModelGateway.

Options:
  --version X     package version; must equal VERSION in src/version.py at REF
  --out-dir DIR   output directory
  --ref REF       committed revision to package (default: HEAD)
  --product       also build a product archive (productbuild) around the component
  --dry-run       stage and verify the payload in DIR/ModelGateway-X.dry-run without pkgbuild
  -h, --help      show this help
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="${2:?missing value for --version}"; shift ;;
    --out-dir) OUT_DIR="${2:?missing value for --out-dir}"; shift ;;
    --ref) REF="${2:?missing value for --ref}"; shift ;;
    --product) PRODUCT=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done

say() { printf '%s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

[ -n "$VERSION" ] && [ -n "$OUT_DIR" ] || { usage >&2; exit 2; }
[[ "$VERSION" =~ ^[0-9]+(\.[0-9]+){1,3}$ ]] || fail "version must be numeric dotted form (for example 0.5.0): $VERSION"
case "$REF" in *[!A-Za-z0-9._/@{}^~:+-]*|'') fail "unsafe ref: $REF" ;; esac
tools=(git plutil python3 shasum tar)
[ "$DRY_RUN" -eq 1 ] || tools+=(pkgbuild pkgutil)
[ "$PRODUCT" -eq 0 ] || [ "$DRY_RUN" -eq 1 ] || tools+=(productbuild)
for tool in "${tools[@]}"; do
  have "$tool" || fail "missing required build tool: $tool"
done

[ -z "$(git -C "$SOURCE_TREE" status --porcelain)" ] || fail "working tree is dirty; package only committed release inputs"
SOURCE_COMMIT="$(git -C "$SOURCE_TREE" rev-parse --verify "$REF^{commit}")" || fail "unknown ref: $REF"
RELEASE_NAME="$VERSION-${SOURCE_COMMIT:0:12}"

# Only regular files from Git: no symlinks, submodules, or unsafe names.
file_count=0
while IFS=$'\t' read -r metadata path; do
  case "${metadata%% *}:$(cut -d' ' -f2 <<<"$metadata")" in
    100644:blob|100755:blob) ;;
    *) fail "runtime source contains a symlink, submodule, or other non-regular entry: $path" ;;
  esac
  [[ "$path" =~ ^[A-Za-z0-9._/-]+$ ]] && [[ "$path" != *..* ]] || fail "unsafe runtime source path: $path"
  file_count=$((file_count + 1))
done < <(git -C "$SOURCE_TREE" ls-tree -r "$SOURCE_COMMIT" -- "${RUNTIME_PATHS[@]}")
[ "$file_count" -gt 0 ] || fail "selected ref has no runtime files"
for required in "${RUNTIME_PATHS[@]}" packaging/component/scripts/postinstall \
  packaging/component/bin/model-gateway-install-from-pkg; do
  git -C "$SOURCE_TREE" cat-file -e "$SOURCE_COMMIT:$required" 2>/dev/null || \
    fail "selected ref does not contain required path: $required"
done
SOURCE_VERSION="$(git -C "$SOURCE_TREE" show "$SOURCE_COMMIT:src/version.py" | sed -n 's/^VERSION = "\(.*\)"$/\1/p')"
[ "$SOURCE_VERSION" = "$VERSION" ] || fail "--version $VERSION does not match src/version.py ($SOURCE_VERSION) at $REF"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/model-gateway-component.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT
PAYLOAD_DIR="$WORK_DIR/payload"
SCRIPTS_DIR="$WORK_DIR/scripts"
SUPPORT_DIR="$PAYLOAD_DIR/Library/Application Support/ModelGateway"
GATEWAY_DIR="$SUPPORT_DIR/package/gateway"
RELEASE_PLIST="$SUPPORT_DIR/package/release.plist"
mkdir -p "$GATEWAY_DIR" "$SUPPORT_DIR/bin" "$SCRIPTS_DIR"

git -C "$SOURCE_TREE" archive "$SOURCE_COMMIT" -- "${RUNTIME_PATHS[@]}" | tar -xf - -C "$GATEWAY_DIR"
git -C "$SOURCE_TREE" show "$SOURCE_COMMIT:packaging/component/bin/model-gateway-install-from-pkg" \
  > "$SUPPORT_DIR/bin/model-gateway-install-from-pkg"
git -C "$SOURCE_TREE" show "$SOURCE_COMMIT:packaging/component/scripts/postinstall" > "$SCRIPTS_DIR/postinstall"
chmod 755 "$SUPPORT_DIR/bin/model-gateway-install-from-pkg" "$SCRIPTS_DIR/postinstall"
[ -z "$(find "$PAYLOAD_DIR" "$SCRIPTS_DIR" ! -type f ! -type d -print -quit)" ] || \
  fail "package input contains a symlink or other non-regular entry"
# Immutable once installed: root-owned (pkgbuild's recommended ownership), not group/other writable.
chmod -R u+w,go-w,a+rX "$PAYLOAD_DIR" "$SCRIPTS_DIR"

(
  cd "$SUPPORT_DIR"
  find ./bin ./package/gateway -type f -print | LC_ALL=C sort | while IFS= read -r file; do
    shasum -a 256 "$file"
  done > manifest.sha256
)
python3 - "$RELEASE_PLIST" "$GATEWAY_DIR/src/version.py" <<PY
import ast, plistlib, sys

values = {node.targets[0].id: ast.literal_eval(node.value)
          for node in ast.parse(open(sys.argv[2]).read()).body
          if isinstance(node, ast.Assign) and len(node.targets) == 1}
metadata = {
    "format_version": 1,
    "package_identifier": "$IDENTIFIER",
    "package_version": "$VERSION",
    "release_name": "$RELEASE_NAME",
    "source_commit": "$SOURCE_COMMIT",
    "manifest_sha256": "$(shasum -a 256 "$SUPPORT_DIR/manifest.sha256" | awk '{print $1}')",
    "helper_sha256": "$(shasum -a 256 "$SUPPORT_DIR/bin/model-gateway-install-from-pkg" | awk '{print $1}')",
    "postinstall_sha256": "$(shasum -a 256 "$SCRIPTS_DIR/postinstall" | awk '{print $1}')",
    "capabilities": sorted(values["CAPABILITIES"]),
    "helper_timeout": $HELPER_TIMEOUT,
    "postinstall_timeout": $POSTINSTALL_TIMEOUT,
}
with open(sys.argv[1], "wb") as handle:
    plistlib.dump(metadata, handle, sort_keys=True)
PY
chmod 644 "$RELEASE_PLIST" "$SUPPORT_DIR/manifest.sha256"
# The installer's own check, against the exact bytes being packaged.
"$SCRIPTS_DIR/postinstall" --verify-payload "$SUPPORT_DIR" >/dev/null

say "source commit: $SOURCE_COMMIT"
say "version:       $VERSION"
say "release:       $RELEASE_NAME"
say "files:         $(wc -l < "$SUPPORT_DIR/manifest.sha256" | tr -d ' ')"
say "manifest:      $(shasum -a 256 "$SUPPORT_DIR/manifest.sha256" | awk '{print $1}')"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

if [ "$DRY_RUN" -eq 1 ]; then
  STAGED="$OUT_DIR/ModelGateway-$VERSION.dry-run"
  rm -rf "$STAGED"
  mkdir -p "$STAGED"
  cp -pR "$PAYLOAD_DIR" "$SCRIPTS_DIR" "$STAGED/"
  say "dry-run:       staged payload and scripts in $STAGED (pkgbuild not run)"
  exit 0
fi

PKG_PATH="$OUT_DIR/ModelGateway-component-$VERSION.pkg"
PRODUCT_PATH="$OUT_DIR/ModelGateway-$VERSION.pkg"
[ ! -e "$PKG_PATH" ] || fail "package already exists: $PKG_PATH"
[ "$PRODUCT" -eq 0 ] || [ ! -e "$PRODUCT_PATH" ] || fail "package already exists: $PRODUCT_PATH"
pkgbuild --root "$PAYLOAD_DIR" --scripts "$SCRIPTS_DIR" --identifier "$IDENTIFIER" \
  --version "$VERSION" --install-location / --ownership recommended "$WORK_DIR/raw.pkg" >/dev/null
# The helper may build a Python environment; give the postinstall room beyond Installer's default.
pkgutil --expand "$WORK_DIR/raw.pkg" "$WORK_DIR/expanded"
python3 - "$WORK_DIR/expanded/PackageInfo" "$POSTINSTALL_TIMEOUT" <<'PY'
import sys
import xml.etree.ElementTree as ET

path, timeout = sys.argv[1:]
tree = ET.parse(path)
postinstall = tree.find("./scripts/postinstall")
if postinstall is None:
    raise SystemExit("generated package is missing postinstall metadata")
postinstall.set("timeout", timeout)
tree.write(path, encoding="utf-8", xml_declaration=True)
PY
pkgutil --flatten "$WORK_DIR/expanded" "$WORK_DIR/component.pkg"
"$SCRIPT_DIR/verify-component-pkg.sh" "$WORK_DIR/component.pkg" >/dev/null
mv -n "$WORK_DIR/component.pkg" "$PKG_PATH"
(cd "$OUT_DIR" && shasum -a 256 "$(basename "$PKG_PATH")" > "$(basename "$PKG_PATH").sha256")
say "built:         $PKG_PATH"
if [ "$PRODUCT" -eq 1 ]; then
  productbuild --package "$PKG_PATH" "$PRODUCT_PATH" >/dev/null
  (cd "$OUT_DIR" && shasum -a 256 "$(basename "$PRODUCT_PATH")" > "$(basename "$PRODUCT_PATH").sha256")
  say "product:       $PRODUCT_PATH"
fi

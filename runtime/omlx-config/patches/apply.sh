#!/bin/bash
# Verify/apply legacy oMLX DeepSeek V4 patches after upgrade.
#
# Current oMLX releases include DeepSeek V4 support under
#   omlx/patches/deepseek_v4/
# In that case this script is a no-op verifier.  The legacy patch path below is
# retained for older installs that carried DeepSeek V4 directly in mlx_lm.
#
# Usage:
#   ./apply.sh           # apply missing legacy patches, if applicable
#   ./apply.sh --check   # verify support/patches are present
#
# After applying legacy patches, restart oMLX:
#   server-ci restart --omlx

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECK=0
if [ "${1:-}" = "--check" ]; then
    CHECK=1
elif [ "${1:-}" != "" ]; then
    echo "Usage: $SCRIPT_DIR/apply.sh [--check]" >&2
    exit 2
fi

OMLX_SITE="$(find "$HOME/.local/share/uv/tools/omlx/lib" -path '*/site-packages/omlx' -type d -print -quit 2>/dev/null | sed 's#/omlx$##')"

if [ -z "$OMLX_SITE" ] || [ ! -d "$OMLX_SITE/omlx" ]; then
    echo "ERROR: oMLX site-packages not found" >&2
    exit 1
fi

INTEGRATED_DSV4="$OMLX_SITE/omlx/patches/deepseek_v4/deepseek_v4_model.py"
if [ -f "$INTEGRATED_DSV4" ]; then
    echo "DeepSeek V4 support present in oMLX integrated patch tree: $INTEGRATED_DSV4"
    exit 0
fi

DEEPSEEK_V4="$OMLX_SITE/mlx_lm/models/deepseek_v4.py"
if [ ! -f "$DEEPSEEK_V4" ]; then
    echo "ERROR: DeepSeek V4 support not found in integrated oMLX patches or legacy mlx_lm module" >&2
    exit 1
fi

has_attn_remap() {
    grep -q 'attn_hc' "$DEEPSEEK_V4"
}

has_rope_offset_fix() {
    grep -q 'isinstance(offset, mx.array)' "$DEEPSEEK_V4"
}

if [ "$CHECK" -eq 1 ]; then
    missing=0
    if ! has_attn_remap; then
        echo "Missing legacy DeepSeek V4 attn_hc/ffn_hc remap"
        missing=1
    fi
    if ! has_rope_offset_fix; then
        echo "Missing legacy DeepSeek V4 RoPE offset scalar conversion"
        missing=1
    fi
    if [ "$missing" -eq 0 ]; then
        echo "Legacy DeepSeek V4 patches present: $DEEPSEEK_V4"
    fi
    exit "$missing"
fi

if ! has_attn_remap; then
    python3 - "$DEEPSEEK_V4" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
needle = '                    nk = nk.replace(f".hc_{sub}_{param}", f".hc_{sub}.{param}")\n'
insert = needle + (
    "\n"
    "            # mlx-community/DeepSeek-V4-Flash-8bit uses attn_hc/ffn_hc\n"
    "            # names for the same HyperConnection modules.\n"
    '            nk = nk.replace(".attn_hc.", ".hc_attn.")\n'
    '            nk = nk.replace(".ffn_hc.", ".hc_ffn.")\n'
)
if needle not in text:
    raise SystemExit(f"patch anchor not found in {path}")
path.write_text(text.replace(needle, insert, 1))
print(f"Patched {path}: DeepSeek V4 attn_hc/ffn_hc remap")
PY
else
    echo "DeepSeek V4 attn_hc/ffn_hc remap already present."
fi

if ! has_rope_offset_fix; then
    python3 - "$DEEPSEEK_V4" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
needle = (
    "        dtype = x.dtype\n"
    "        T = x.shape[-2]\n"
    "        pos = mx.arange(offset, offset + T, dtype=mx.float32)\n"
)
insert = (
    "        dtype = x.dtype\n"
    "        T = x.shape[-2]\n"
    "        if isinstance(offset, mx.array):\n"
    "            offset = int(offset)\n"
    "        pos = mx.arange(offset, offset + T, dtype=mx.float32)\n"
)
if needle not in text:
    raise SystemExit(f"patch anchor not found in {path}")
path.write_text(text.replace(needle, insert, 1))
print(f"Patched {path}: DeepSeek V4 RoPE offset scalar conversion")
PY
else
    echo "DeepSeek V4 RoPE offset scalar conversion already present."
fi

echo "Legacy DeepSeek V4 patches applied."

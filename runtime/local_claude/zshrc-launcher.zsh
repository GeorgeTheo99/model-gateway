# DEPRECATED compat shim for older shell configs.
# Sources the legacy runtime/pi-launcher.zsh, which is superseded by
# pi-shared/bin/pi-catalog (pi-* only, no claude-*/codex-*). Not sourced by
# the active ~/.zshrc on ls99. New machines should source the pi-catalog-
# generated launcher instead (see pi-shared README).
source "${HOME}/local_code/model-gateway/runtime/pi-launcher.zsh"

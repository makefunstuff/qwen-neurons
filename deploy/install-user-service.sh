#!/usr/bin/env bash
# Install and start a systemd user service that runs this checkout (settings come from config.local.env).
set -euo pipefail
cd "$(dirname "$0")/.."
./play setup
unit="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/evil-pugs.service"
mkdir -p "$(dirname "$unit")"
sed "s#@REPO@#$PWD#g" deploy/evil-pugs.service > "$unit"
systemctl --user daemon-reload
systemctl --user enable --now evil-pugs.service
systemctl --user --no-pager status evil-pugs.service | head -5

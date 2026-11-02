#!/usr/bin/env bash
# Fetches the Kronos model code (not a pip package) into vendor/Kronos
set -euo pipefail
mkdir -p vendor
if [ ! -d vendor/Kronos ]; then
  git clone --depth 1 https://github.com/shiyu-coder/Kronos vendor/Kronos
fi
echo "Kronos ready in vendor/Kronos"

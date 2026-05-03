#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Shadow Loom — fresh-clone bootstrap
#
# Brings a freshly-cloned repo to a runnable state:
#   1. Verifies Python >= 3.13 is available.
#   2. Creates .venv if missing and activates it.
#   3. Installs the package in editable mode.
#   4. Copies .env.example -> .env if .env is missing.
#   5. Checks for Ollama and reports whether the default model is pulled.
#
# Idempotent — re-running is safe.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.6:27b}"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${VENV:-.venv}"

# Pretty headings without depending on tput.
say() { printf '\n\033[1;36m▸ %s\033[0m\n' "$*"; }
ok()  { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn(){ printf '  \033[33m!\033[0m %s\n' "$*"; }
err() { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; }

# ─────────────────────────────────────────────────────────────────
# 1. Python version
# ─────────────────────────────────────────────────────────────────
say "Checking Python version"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    err "$PYTHON_BIN not found on PATH."
    err "Install Python 3.13+ from https://www.python.org/downloads/ or your OS package manager."
    exit 1
fi
PY_VER="$("$PYTHON_BIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
PY_MAJOR="${PY_VER%%.*}"
PY_MINOR="${PY_VER##*.}"
if (( PY_MAJOR < 3 || (PY_MAJOR == 3 && PY_MINOR < 13) )); then
    err "Found Python $PY_VER; Shadow Loom needs >= 3.13."
    err "Tip: pyenv install 3.13 && pyenv local 3.13"
    exit 1
fi
ok "Python $PY_VER"

# ─────────────────────────────────────────────────────────────────
# 2. Virtualenv
# ─────────────────────────────────────────────────────────────────
say "Setting up virtualenv at $VENV_DIR"
if [[ -d "$VENV_DIR" ]]; then
    ok "$VENV_DIR already exists"
else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    ok "Created $VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
ok "Activated $VENV_DIR"

# ─────────────────────────────────────────────────────────────────
# 3. Install package
# ─────────────────────────────────────────────────────────────────
say "Installing Shadow Loom (editable)"
python -m pip install --upgrade pip --quiet
python -m pip install -e . --quiet
ok "Editable install complete"

# ─────────────────────────────────────────────────────────────────
# 4. .env
# ─────────────────────────────────────────────────────────────────
say "Configuring .env"
if [[ -f .env ]]; then
    ok ".env already exists — left untouched"
else
    cp .env.example .env
    ok "Wrote .env from .env.example"
    warn "Edit .env to set OPENROUTER_API_KEY / OAUTH credentials if you need them."
fi

# ─────────────────────────────────────────────────────────────────
# 5. Ollama check
# ─────────────────────────────────────────────────────────────────
say "Checking Ollama (LLM backend)"
if ! command -v ollama >/dev/null 2>&1; then
    warn "ollama not found on PATH."
    warn "Install from https://ollama.com/download, then run:"
    warn "    ollama pull $OLLAMA_MODEL"
    warn "Or switch DEFAULT_MODEL in .env to an OpenRouter / OpenAI model."
else
    ok "ollama installed"
    if ollama list 2>/dev/null | grep -q "$OLLAMA_MODEL"; then
        ok "Model $OLLAMA_MODEL is available"
    else
        warn "Model $OLLAMA_MODEL is NOT pulled yet."
        warn "Run:  ollama pull $OLLAMA_MODEL"
        warn "(Or change DEFAULT_MODEL in .env to a smaller / hosted model.)"
    fi
fi

# ─────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────
say "Setup complete"
cat <<EOF

  Activate the venv in new shells:
      source $VENV_DIR/bin/activate

  Common next steps:
      make ui          # start the NiceGUI workspace (http://localhost:7860)
      make mcp         # start the MCP server on stdio
      make pipeline    # run the bundled end-to-end demo
      make test        # run the fast test suite

  Configuration reference:
      .env             # local overrides (yours, git-ignored)
      docs/settings.md # full list of every knob

EOF

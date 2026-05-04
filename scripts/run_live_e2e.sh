#!/usr/bin/env bash
# scripts/run_live_e2e.sh
#
# Run the live end-to-end test suite (tests/test_live_e2e.py) detached
# under nohup so it survives shell exit. All output is appended to a
# timestamped log file under logs/.
#
# Usage:
#   scripts/run_live_e2e.sh                 # default: full live suite
#   scripts/run_live_e2e.sh -k Directive    # pytest -k filter
#   scripts/run_live_e2e.sh tests/test_live_e2e.py::TestEvaluationE2E
#
# Environment overrides:
#   PYTHON       — python interpreter to use (default: $(which python))
#   LOG_DIR      — log directory (default: ./logs)
#   PYTEST_ARGS  — extra args appended to pytest command

set -euo pipefail

# Resolve repo root from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-$(command -v python)}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_DIR}/live_e2e_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/live_e2e_${TIMESTAMP}.pid"

# Default target: the live test file. Caller args (filters, paths) win.
TARGET=("tests/test_live_e2e.py")
if [[ $# -gt 0 ]]; then
  TARGET=("$@")
fi

# Pre-flight: warn if Ollama isn't reachable (tests will skip otherwise).
if ! curl -sf -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "WARNING: Ollama is not reachable at localhost:11434." >&2
  echo "         All @requires_ollama tests will be SKIPPED." >&2
fi

PYTEST_CMD=(
  "${PYTHON}" -m pytest
  -v
  --tb=short
  -o "log_cli=true"
  -o "log_cli_level=INFO"
  "${TARGET[@]}"
  ${PYTEST_ARGS:-}
)

echo "Repo:       ${REPO_ROOT}"
echo "Python:     ${PYTHON}"
echo "Log:        ${LOG_FILE}"
echo "PID file:   ${PID_FILE}"
echo "Command:    ${PYTEST_CMD[*]}"
echo

# Launch detached. nohup + setsid makes the child immune to SIGHUP and
# detaches it from the controlling terminal so the test run keeps going
# even if the SSH session dies.
nohup setsid "${PYTEST_CMD[@]}" >"${LOG_FILE}" 2>&1 < /dev/null &
PID=$!
echo "${PID}" > "${PID_FILE}"

echo "Started live e2e suite as PID ${PID}."
echo
echo "Tail the log:   tail -f ${LOG_FILE}"
echo "Stop the run:   kill \$(cat ${PID_FILE})"

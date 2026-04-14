#!/usr/bin/env bash
set -euo pipefail

# Run a script under scripts/ in a detached screen session and persist logs.
# Usage:
#   bash scripts/run_with_screen.sh eval_geometry_forcing.sh
#   bash scripts/run_with_screen.sh eval_geometry_forcing.sh my_session
#   bash scripts/run_with_screen.sh scripts/eval_rpe.sh my_session logs

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${ROOT_DIR}/scripts"
DEFAULT_LOG_DIR="${ROOT_DIR}/logs"

if [[ "${1:-}" == "" ]] || [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
  echo "Usage: $0 <script_name_or_path> [session_name] [log_dir]"
  echo "Example: $0 eval_geometry_forcing.sh"
  echo "Example: $0 scripts/eval_rpe.sh eval_rpe_1 logs"
  echo ""
  echo "Available scripts in ${SCRIPTS_DIR}:"
  ls -1 "${SCRIPTS_DIR}"/*.sh | xargs -n1 basename
  exit 0
fi

if ! command -v screen >/dev/null 2>&1; then
  echo "Error: screen is not installed. Install it with: sudo apt-get install screen"
  exit 1
fi

input_script="$1"
user_session_name="${2:-}"
log_dir_input="${3:-${DEFAULT_LOG_DIR}}"

if [[ "${input_script}" = /* ]]; then
  script_path="${input_script}"
elif [[ "${input_script}" == scripts/* ]]; then
  script_path="${ROOT_DIR}/${input_script}"
else
  script_path="${SCRIPTS_DIR}/${input_script}"
fi

if [[ ! -f "${script_path}" ]]; then
  echo "Error: script not found: ${script_path}"
  exit 1
fi

mkdir -p "${log_dir_input}"

script_base="$(basename "${script_path}" .sh)"
timestamp="$(date +%Y%m%d_%H%M%S)"
default_session_name="${script_base}_${timestamp}"
session_name="${user_session_name:-${default_session_name}}"
log_file="${log_dir_input}/${session_name}.log"

# Ensure no stale session with the same name.
if screen -list | grep -q "[.]${session_name}[[:space:]]"; then
  echo "Error: screen session '${session_name}' already exists."
  echo "Use a different session name or stop it first: screen -S ${session_name} -X quit"
  exit 1
fi

launch_cmd="cd '${ROOT_DIR}' && bash '${script_path}' 2>&1 | tee -a '${log_file}'"
screen -dmS "${session_name}" bash -lc "${launch_cmd}"

echo "Started."
echo "  Session : ${session_name}"
echo "  Script  : ${script_path}"
echo "  Log     : ${log_file}"
echo ""
echo "Useful commands:"
echo "  Attach session : screen -r ${session_name}"
echo "  Tail logs      : tail -f '${log_file}'"
echo "  Stop session   : screen -S ${session_name} -X quit"

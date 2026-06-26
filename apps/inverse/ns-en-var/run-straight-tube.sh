#!/usr/bin/env bash
set -euo pipefail

usage()
{
  cat <<'EOF'
Usage: run-straight-tube.sh [BUILD_DIR]

Runs the straight-tube EnVar workflow:
  1. make-ensemble
  2. make-obs-ensemble
  3. ns-en-var

Environment:
  FEMX_BUILD_DIR  Build directory containing the apps.
  NP              MPI ranks for forward apps. Default: 4.
  MODES           Boundary ensemble modes. Default: 2.
  INIT_MODES      Initial velocity ensemble modes. Default: 2.
  AMPLITUDE       Relative perturbation amplitude. Default: 0.35.
  MAX_ITERATIONS  Override optimizer max iterations.
  ABS_TOL         Override optimizer absolute gradient tolerance.
  REL_TOL         Override optimizer relative tolerance.
  STEP_TOL        Override optimizer step tolerance.
  LOG_DIR         Directory for logs. Default: straight-tube output/logs.
  BUILD_TARGETS   Set to 1 to build required targets first.
  JOBS            Build parallelism when BUILD_TARGETS=1. Default: 4.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]
then
  usage
  exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
case_dir="${script_dir}/inputs/straight-tube"
config="${CONFIG:-${case_dir}/Config.json}"
out_dir="${case_dir}/output"
log_dir="${LOG_DIR:-${out_dir}/logs}"

build_dir="${FEMX_BUILD_DIR:-${1:-}}"
if [[ -z "${build_dir}" ]]
then
  for candidate in "${repo_root}/build" "/tmp/femx-ns-en-var-build"
  do
    if [[ -x "${candidate}/apps/inverse/ns-en-var/ns-en-var" ]]
    then
      build_dir="${candidate}"
      break
    fi
  done
fi

if [[ -z "${build_dir}" ]]
then
  echo "run-straight-tube.sh: build directory not found" >&2
  echo "Set FEMX_BUILD_DIR or pass BUILD_DIR as the first argument." >&2
  exit 1
fi

make_ensemble="${build_dir}/apps/inverse/make-ensemble/make-ensemble"
make_obs_ensemble="${build_dir}/apps/inverse/make-obs-ensemble/make-obs-ensemble"
ns_en_var="${build_dir}/apps/inverse/ns-en-var/ns-en-var"

for exe in "${make_ensemble}" "${make_obs_ensemble}" "${ns_en_var}"
do
  if [[ ! -x "${exe}" ]]
  then
    echo "run-straight-tube.sh: executable not found: ${exe}" >&2
    exit 1
  fi
done

np="${NP:-4}"
modes="${MODES:-2}"
init_modes="${INIT_MODES:-2}"
amplitude="${AMPLITUDE:-0.35}"
ns_args=(--config "${config}")
if [[ -n "${MAX_ITERATIONS:-}" ]]
then
  ns_args+=(--max-iterations "${MAX_ITERATIONS}")
fi
if [[ -n "${ABS_TOL:-}" ]]
then
  ns_args+=(--abs-tol "${ABS_TOL}")
fi
if [[ -n "${REL_TOL:-}" ]]
then
  ns_args+=(--rel-tol "${REL_TOL}")
fi
if [[ -n "${STEP_TOL:-}" ]]
then
  ns_args+=(--step-tol "${STEP_TOL}")
fi

mkdir -p "${log_dir}"

if [[ "${BUILD_TARGETS:-0}" == "1" ]]
then
  cmake --build "${build_dir}" --target make-ensemble make-obs-ensemble ns-en-var -j "${JOBS:-4}"
fi

mpi=()
if [[ "${np}" != "1" ]]
then
  mpi=(mpirun -n "${np}")
fi

run_log()
{
  local log_file="$1"
  shift
  echo "==> $*" | tee "${log_file}"
  "$@" 2>&1 | tee -a "${log_file}"
}

run_log "${log_dir}/make-ensemble.log" \
  "${make_ensemble}" \
  --config "${config}" \
  --modes "${modes}" \
  --initial-modes "${init_modes}" \
  --amplitude "${amplitude}"

run_log "${log_dir}/make-obs-ensemble.log" \
  "${mpi[@]}" "${make_obs_ensemble}" \
  --config "${config}"

run_log "${log_dir}/ns-en-var.log" \
  "${mpi[@]}" "${ns_en_var}" \
  "${ns_args[@]}"

echo
echo "Straight-tube EnVar workflow complete."
echo "Logs: ${log_dir}"
echo "Visualization: ${out_dir}/ns-en-var-straight-tube.xdmf"

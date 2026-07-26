#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
web_dir="${DEMO_WEB_DIR:-$root/web}"
port="${DEMO_PORT:-4173}"

case $# in
  0) check_only=false ;;
  1)
    if [[ "$1" != "--check" ]]; then
      echo "Usage: scripts/prepare-demo.sh [--check]" >&2
      exit 2
    fi
    check_only=true
    ;;
  *)
    echo "Usage: scripts/prepare-demo.sh [--check]" >&2
    exit 2
    ;;
esac

if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
  echo "DEMO_PORT must be an integer from 1 through 65535." >&2
  exit 2
fi

validate_replay() {
  node - "$1" <<'NODE'
const replay = require(process.argv[2]);
const run = replay.optimized || replay;
const comparison = replay.comparison || {};
const close = (actual, expected) => Number.isFinite(actual) && Math.abs(actual - expected) < 0.01;
const tools = new Set((run.audit || []).map((event) => event.tool));
const valid = run.metadata?.verified === true
  && run.actions?.length === 168
  && run.errors?.length === 0
  && run.telemetry?.length === 672
  && tools.has("inspect_model")
  && tools.has("read_telemetry")
  && tools.has("read_runtime_errors")
  && tools.has("evaluate_action")
  && tools.has("apply_setpoints")
  && close(run.summary?.pmv_compliance_pct, 98.030303)
  && close(comparison.energy_change_pct, -0.350973)
  && close(comparison.pmv_compliance_delta_pct, -1.212121);
if (!valid) process.exit(1);
console.log(`Replay: verified
Validated actions: ${run.actions.length}
Runtime errors: ${run.errors.length}
PMV compliance: ${run.summary.pmv_compliance_pct.toFixed(2)}%
PMV baseline delta: ${comparison.pmv_compliance_delta_pct.toFixed(2)} points`);
NODE
}

for file in demo-run.json index.html app.js style.css architecture.md; do
  if [[ ! -f "$web_dir/$file" ]]; then
    echo "Missing demo asset: $web_dir/$file" >&2
    exit 1
  fi
done

if [[ -e "$web_dir/latest.json" ]]; then
  echo "Remove web/latest.json before recording; live output may be stale or failed." >&2
  exit 1
fi

summary="$(validate_replay "$web_dir/demo-run.json")" || {
  echo "Replay verification failed; recording contract does not match verified metrics." >&2
  exit 1
}

printf '%s\n' "$summary"
echo "Dashboard URL: http://127.0.0.1:$port/"
echo "Architecture URL: http://127.0.0.1:$port/architecture.md"

if $check_only; then
  exit 0
fi

log="${DEMO_LOG:-$(mktemp "${TMPDIR:-/tmp}/honeywell-dashboard.XXXXXX.log")}"

verify_server_replay() {
  local downloaded
  downloaded="$(mktemp "${TMPDIR:-/tmp}/honeywell-replay.XXXXXX.json")"
  if ! curl --fail --silent --output "$downloaded" "http://127.0.0.1:$port/demo-run.json" \
    || ! cmp --silent "$web_dir/demo-run.json" "$downloaded" \
    || ! validate_replay "$downloaded" >/dev/null; then
    rm -f "$downloaded"
    return 1
  fi
  rm -f "$downloaded"
}

if curl --fail --silent --output /dev/null "http://127.0.0.1:$port/"; then
  if ! verify_server_replay; then
    echo "Port $port serves different or invalid demo content; choose DEMO_PORT." >&2
    exit 1
  fi
else
  nohup python3 -m http.server "$port" --bind 127.0.0.1 --directory "$web_dir" >"$log" 2>&1 &
  server_pid=$!
  for _ in {1..20}; do
    if curl --fail --silent --output /dev/null "http://127.0.0.1:$port/"; then
      break
    fi
    sleep 0.1
  done
  if ! curl --fail --silent --output /dev/null "http://127.0.0.1:$port/"; then
    kill "$server_pid" 2>/dev/null || true
    echo "Dashboard failed to start; inspect $log" >&2
    exit 1
  fi
  if ! verify_server_replay; then
    kill "$server_pid" 2>/dev/null || true
    echo "Dashboard started but replay identity check failed; inspect $log" >&2
    exit 1
  fi
  echo "Dashboard PID: $server_pid (stop with: kill $server_pid)"
fi

for path in / /demo-run.json /architecture.md; do
  if ! status="$(curl --silent --output /dev/null --write-out '%{http_code}' "http://127.0.0.1:$port$path")"; then
    if [[ -n "${server_pid:-}" ]]; then
      kill "$server_pid" 2>/dev/null || true
    fi
    echo "Dashboard smoke check failed: could not fetch $path" >&2
    exit 1
  fi
  if [[ "$status" != "200" ]]; then
    if [[ -n "${server_pid:-}" ]]; then
      kill "$server_pid" 2>/dev/null || true
    fi
    echo "Dashboard smoke check failed: $path returned $status" >&2
    exit 1
  fi
done

echo "Dashboard: ready"
echo "Record stable replay only; do not use ?live=1 for this video."

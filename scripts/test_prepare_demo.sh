#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/web"
for file in demo-run.json index.html app.js style.css architecture.md; do
  cp "$root/web/$file" "$tmp/web/$file"
done

output="$(DEMO_WEB_DIR="$tmp/web" DEMO_PORT=4199 "$root/scripts/prepare-demo.sh" --check)"
grep -Fq "Replay: verified" <<<"$output"
grep -Fq "Validated actions: 168" <<<"$output"
grep -Fq "Runtime errors: 0" <<<"$output"
grep -Fq "Dashboard URL: http://127.0.0.1:4199/" <<<"$output"

if DEMO_WEB_DIR="$tmp/web" "$root/scripts/prepare-demo.sh" --check unexpected >"$tmp/out" 2>"$tmp/err"; then
  echo "expected extra argument check to fail" >&2
  exit 1
fi
grep -Fq "Usage:" "$tmp/err"

if DEMO_WEB_DIR="$tmp/web" DEMO_PORT=invalid "$root/scripts/prepare-demo.sh" --check >"$tmp/out" 2>"$tmp/err"; then
  echo "expected invalid port check to fail" >&2
  exit 1
fi
grep -Fq "DEMO_PORT" "$tmp/err"

node - "$tmp/web/demo-run.json" <<'NODE'
const fs = require("fs");
const path = process.argv[2];
const replay = JSON.parse(fs.readFileSync(path));
replay.comparison.pmv_compliance_delta_pct = 0;
fs.writeFileSync(path, JSON.stringify(replay));
NODE
if DEMO_WEB_DIR="$tmp/web" "$root/scripts/prepare-demo.sh" --check >"$tmp/out" 2>"$tmp/err"; then
  echo "expected false replay metric check to fail" >&2
  exit 1
fi
grep -Fq "Replay verification failed" "$tmp/err"
cp "$root/web/demo-run.json" "$tmp/web/demo-run.json"

node - "$tmp/web/demo-run.json" <<'NODE'
const fs = require("fs");
const path = process.argv[2];
const replay = JSON.parse(fs.readFileSync(path));
replay.optimized.telemetry.pop();
replay.optimized.audit = replay.optimized.audit.filter((event) => event.tool !== "inspect_model");
fs.writeFileSync(path, JSON.stringify(replay));
NODE
if DEMO_WEB_DIR="$tmp/web" "$root/scripts/prepare-demo.sh" --check >"$tmp/out" 2>"$tmp/err"; then
  echo "expected incomplete replay contract check to fail" >&2
  exit 1
fi
grep -Fq "Replay verification failed" "$tmp/err"
cp "$root/web/demo-run.json" "$tmp/web/demo-run.json"

touch "$tmp/web/latest.json"
if DEMO_WEB_DIR="$tmp/web" "$root/scripts/prepare-demo.sh" --check >"$tmp/out" 2>"$tmp/err"; then
  echo "expected stale latest.json check to fail" >&2
  exit 1
fi
grep -Fq "Remove web/latest.json" "$tmp/err"
rm "$tmp/web/latest.json"

mkdir "$tmp/bin"
cat >"$tmp/bin/curl" <<'SH'
#!/usr/bin/env bash
url="${*: -1}"
if [[ -f "$CURL_STATE_DIR/replay-checked" && "$url" == */ ]]; then
  exit 7
fi
if [[ "$url" == */demo-run.json ]]; then
  touch "$CURL_STATE_DIR/replay-checked"
fi
exec /usr/bin/curl "$@"
SH
chmod +x "$tmp/bin/curl"
if PATH="$tmp/bin:$PATH" CURL_STATE_DIR="$tmp" DEMO_WEB_DIR="$tmp/web" DEMO_PORT=4201 \
  "$root/scripts/prepare-demo.sh" >"$tmp/start-out" 2>"$tmp/start-err"; then
  echo "expected simulated curl transport failure" >&2
  exit 1
fi
sleep 0.2
if /usr/bin/curl --fail --silent --output /dev/null "http://127.0.0.1:4201/"; then
  pid="$(sed -n 's/^Dashboard PID: \([0-9]*\).*/\1/p' "$tmp/start-out")"
  [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  echo "expected spawned dashboard cleanup after curl failure" >&2
  exit 1
fi

echo "prepare-demo tests passed"

#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL=60
WORKERS_PER_REPO=8
REPORT_FILE="$REPO_ROOT/download_reports/hf_download_progress.log"
LINKS_DIR="$REPO_ROOT/hf_assets"
STATE_DIR="$REPO_ROOT/.hf_download_state"

usage() {
    cat <<'EOF'
Usage: scripts/download_hf_assets.sh [options]

Download the requested Hugging Face assets into the configured HF cache.
Existing cache entries are reused, so rerunning resumes interrupted downloads.

Options:
  --interval SECONDS          Progress report interval (default: 60)
  --workers-per-repo COUNT    Parallel file downloads per repository (default: 8)
  --report PATH               Progress report file
  --links-dir PATH            Directory for cache snapshot symlinks
  -h, --help                  Show this help
EOF
}

while (($#)); do
    case "$1" in
        --interval) INTERVAL="$2"; shift 2 ;;
        --workers-per-repo) WORKERS_PER_REPO="$2"; shift 2 ;;
        --report) REPORT_FILE="$2"; shift 2 ;;
        --links-dir) LINKS_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' '--interval must be a positive integer' >&2; exit 2; }
[[ "$WORKERS_PER_REPO" =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' '--workers-per-repo must be a positive integer' >&2; exit 2; }
command -v hf >/dev/null || { printf '%s\n' 'hf CLI is not installed or not on PATH' >&2; exit 1; }

mkdir -p "$(dirname "$REPORT_FILE")" "$LINKS_DIR" "$STATE_DIR"

ASSETS=(
    "lingbot-va-posttrain-robotwin|robbyant/lingbot-va-posttrain-robotwin|model"
    "FlashWAM-RoboTwin|NU-World-Model-Embodied-AI/FlashWAM-RoboTwin|model"
    "robotwin-clean-and-aug-lerobot|robbyant/robotwin-clean-and-aug-lerobot|dataset"
)

declare -A PIDS STATES OUTPUTS
TOTAL=${#ASSETS[@]}
COMPLETED=0
FAILED=0

asset_cache_dir() {
    local repo_type="$1"
    local repo_id="$2"
    local cache_root="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}}"
    printf '%s/%ss--%s\n' "$cache_root" "$repo_type" "${repo_id//\//--}"
}

cached_bytes() {
    local total_bytes=0
    local entry name repo repo_type cache_dir bytes
    for entry in "${ASSETS[@]}"; do
        IFS='|' read -r name repo repo_type <<<"$entry"
        cache_dir="$(asset_cache_dir "$repo_type" "$repo")"
        [[ -d "$cache_dir" ]] || continue
        bytes="$(du -sB1 "$cache_dir" | awk '{print $1}')"
        total_bytes=$((total_bytes + bytes))
    done
    printf '%s\n' "$total_bytes"
}

format_size() {
    awk -v bytes="$1" 'BEGIN { printf "%.1f MiB", bytes / 1024 / 1024 }'
}

initialize_report() {
    [[ -s "$REPORT_FILE" ]] && return
    {
        printf '# Hugging Face Download Progress\n\n'
        printf '| Time | Event | Completed | Failed | Active tasks | Cache | Delta | Interval | Rate |\n'
        printf '| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |\n'
    } >"$REPORT_FILE"
}

report() {
    local event="$1"
    local active="$2"
    local bytes="$3"
    local delta="$4"
    local seconds="$5"
    local rate=0
    ((delta < 0)) && delta=0
    ((seconds > 0)) && rate=$((delta / seconds / 1024 / 1024))
    printf '| %s | %s | %d/%d | %d | %s | %s | +%s | %ss | %d MiB/s |\n' \
        "$(date '+%Y-%m-%d %H:%M:%S %z')" "$event" "$COMPLETED" "$TOTAL" "$FAILED" "$active" \
        "$(format_size "$bytes")" "$(format_size "$delta")" "$seconds" "$rate" \
        >>"$REPORT_FILE"
}

link_snapshot() {
    local name="$1"
    local snapshot="$2"
    local destination="$LINKS_DIR/$name"
    [[ -d "$snapshot" ]] || return 1
    if [[ -e "$destination" && ! -L "$destination" ]]; then
        printf 'Refusing to replace non-symlink: %s\n' "$destination" >&2
        return 1
    fi
    ln -sfn "$snapshot" "$destination"
}

start_download() {
    local name="$1"
    local repo="$2"
    local repo_type="$3"
    local output="$STATE_DIR/$name.path"
    local -a args=(download "$repo" --repo-type "$repo_type" --max-workers "$WORKERS_PER_REPO" --quiet)
    if [[ "$repo_type" == "dataset" ]]; then
        args+=(--include "lerobot_robotwin_eef_clean_50/**" "empty_emb/**")
    fi
    hf "${args[@]}" >"$output" 2>"$STATE_DIR/$name.stderr" &
    PIDS["$name"]=$!
    OUTPUTS["$name"]="$output"
    STATES["$name"]=running
}

initialize_report

for entry in "${ASSETS[@]}"; do
    IFS='|' read -r name repo repo_type <<<"$entry"
    start_download "$name" "$repo" "$repo_type"
done

stop_downloads() {
    local pid
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait
    exit 130
}
trap stop_downloads INT TERM

previous_bytes="$(cached_bytes)"
report started "$(IFS=,; echo "${!PIDS[*]}")" "$previous_bytes" 0 0
previous_time="$(date +%s)"

while :; do
    active=()
    for entry in "${ASSETS[@]}"; do
        IFS='|' read -r name repo repo_type <<<"$entry"
        [[ "${STATES[$name]}" == "running" ]] || continue
        pid="${PIDS[$name]}"
        if kill -0 "$pid" 2>/dev/null; then
            active+=("$name")
            continue
        fi
        if wait "$pid"; then
            snapshot="$(tail -n 1 "${OUTPUTS[$name]}")"
            if link_snapshot "$name" "$snapshot"; then
                STATES["$name"]=completed
                COMPLETED=$((COMPLETED + 1))
            else
                STATES["$name"]=failed
                FAILED=$((FAILED + 1))
            fi
        else
            STATES["$name"]=failed
            FAILED=$((FAILED + 1))
        fi
    done

    now="$(date +%s)"
    current_bytes="$(cached_bytes)"
    elapsed=$((now - previous_time))
    active_names="$(IFS=,; echo "${active[*]:-none}")"
    if ((${#active[@]} == 0)); then
        report finished "$active_names" "$current_bytes" "$((current_bytes - previous_bytes))" "$elapsed"
        break
    fi
    report progress "$active_names" "$current_bytes" "$((current_bytes - previous_bytes))" "$elapsed"
    previous_bytes="$current_bytes"
    previous_time="$now"
    sleep "$INTERVAL"
done

((FAILED == 0))

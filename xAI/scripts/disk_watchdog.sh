#!/bin/bash
# PhenoVision preadapt DISK WATCHDOG (plan component C5 — disk safety: WARN + AUTO-PAUSE).
#
# A standalone, CPU-only watchdog for the /blue group quota during a preadapt wave. It does NOT
# use a GPU and requests NO --gres, so it does NOT count against the FIRM ~3-concurrent-GPU cap
# of the guralnick allocation (run it alongside the 2 trainers + 1 collector that DO use the 3
# GPUs). It polls the group quota via `blue_quota`, and:
#
#   * SOFT threshold crossed (free < SOFT_GB, default 300 GB): EMAIL a one-time warning to
#     r.dinnage@gmail.com (debounced — re-armed only after free climbs back above SOFT_GB).
#   * HARD threshold crossed (free < HARD_GB, default 80 GB): create the PAUSE sentinel so
#     trainers STOP EMITTING checkpoints (the trainer's emit path polls this sentinel; training
#     itself keeps running — only checkpoint emission is gated).
#   * RELEASE threshold reached (free >= RELEASE_GB, default 150 GB) while paused: remove the
#     PAUSE sentinel so emission resumes.
#
# Disk frugality is critical: the group /blue is at ~170 GB free and the model-only retention
# ladder keeps each kept checkpoint ~1.2 GB, but a full optimizer-state checkpoint (kept for
# resume) is larger — a runaway producer can fill the filesystem and corrupt EVERY group job.
#
# Usage (standalone or launched by run_preadapt_wave.sh):
#     sbatch xAI/scripts/disk_watchdog.sh
#     # env overrides:
#     SOFT_GB=300 HARD_GB=80 RELEASE_GB=150 POLL=120 sbatch xAI/scripts/disk_watchdog.sh
#     # run in the foreground (e.g. inside the wave launcher) for a quick check:
#     POLL=10 bash xAI/scripts/disk_watchdog.sh
#
# run_preadapt_wave.sh can launch it as a 4th, GPU-free job:
#     sbatch --job-name=preadapt_diskwatch xAI/scripts/disk_watchdog.sh
# (No --gres => it never competes for the 3 GPUs; safe to run for the whole wave.)
#
# Stop it cleanly by: touch xAI/output/preadapt/_DISK_WATCHDOG_STOP   (or scancel / SIGTERM).
#
#SBATCH --job-name=preadapt_diskwatch
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=96:00:00
#SBATCH --output xAI/logs/%x-%j.out
#SBATCH --error  xAI/logs/%x-%j.err
#SBATCH --mail-user=r.dinnage@gmail.com
#SBATCH --mail-type=FAIL,END
# NOTE: deliberately NO --partition and NO --gres — this is a CPU-only job that must NOT consume
# one of the 3 GPUs. It runs on whatever CPU node SLURM places it on under the guralnick QOS.

# Env init FIRST with NO `set -u`: a bare SLURM shell sourcing .bashrc trips over unbound vars
# (e.g. BASHRCSOURCED) under `set -u`. Sourcing the profile makes `module`/`blue_quota` available
# in batch context. Tolerate a missing .bashrc when run interactively in a login shell.
if [[ -f "/home/${USER}/.bashrc" ]]; then
    # shellcheck disable=SC1090
    source "/home/${USER}/.bashrc" || true
fi
set -eo pipefail

PROJECT_ROOT=/blue/guralnick/share/r.dinnage/Projects/phenovision
cd "$PROJECT_ROOT"
mkdir -p xAI/logs xAI/output/preadapt

# --- thresholds (env-overridable), all in GB of FREE space (Limit - Use) ---
SOFT_GB="${SOFT_GB:-300}"      # email a warning below this
HARD_GB="${HARD_GB:-80}"       # create PAUSE sentinel below this (trainers stop emitting)
RELEASE_GB="${RELEASE_GB:-150}"  # remove PAUSE sentinel at/above this (emission resumes)
POLL="${POLL:-120}"            # seconds between polls
GROUP="${WATCHDOG_GROUP:-guralnick}"
MAIL_TO="${WATCHDOG_MAIL_TO:-r.dinnage@gmail.com}"

PREADAPT_BASE="xAI/output/preadapt"
PAUSE_SENTINEL="${PREADAPT_BASE}/_DISK_PAUSE"
WARN_SENTINEL="${PREADAPT_BASE}/_DISK_WARN"
STOP_SENTINEL="${PREADAPT_BASE}/_DISK_WATCHDOG_STOP"

echo "$(date)"
echo "host=$(hostname)"
echo "=== preadapt disk watchdog ==="
echo "group=$GROUP  SOFT_GB=$SOFT_GB  HARD_GB=$HARD_GB  RELEASE_GB=$RELEASE_GB  POLL=${POLL}s"
echo "pause_sentinel=$PAUSE_SENTINEL"
echo "warn_sentinel =$WARN_SENTINEL"
echo "stop_sentinel =$STOP_SENTINEL"

# Sanity: thresholds must be ordered HARD < RELEASE <= SOFT for the hysteresis to make sense.
if ! awk -v h="$HARD_GB" -v r="$RELEASE_GB" -v s="$SOFT_GB" 'BEGIN{exit !(h < r && r <= s)}'; then
    echo "WARNING: thresholds out of order (expected HARD < RELEASE <= SOFT): " \
         "HARD_GB=$HARD_GB RELEASE_GB=$RELEASE_GB SOFT_GB=$SOFT_GB. Continuing anyway." >&2
fi

# Clean shutdown on SIGTERM/SIGINT (scancel sends SIGTERM): note it and exit 0.
_TERM=0
trap '_TERM=1; echo "[diskwatch] caught signal -> will exit after this cycle."' TERM INT

# --- convert a quota size token (e.g. 51.83T, 52T, 870.5G, 1024000k, 0k) to GB (float) ---
# blue_quota uses binary-ish suffixes T (TiB), G (GiB), M (MiB), k (KiB); a bare number is bytes.
# We normalize everything to GB (=GiB here) for thresholding. Robust to upper/lower-case suffix.
to_gb() {
    awk -v tok="$1" '
        BEGIN {
            s = tok
            # strip a trailing "B" if present (e.g. "GB")
            sub(/[Bb]$/, "", s)
            unit = ""
            if (s ~ /[TtGgMmKk]$/) { unit = substr(s, length(s), 1); s = substr(s, 1, length(s)-1) }
            val = s + 0.0
            u = toupper(unit)
            if      (u == "T") gb = val * 1024.0
            else if (u == "G") gb = val
            else if (u == "M") gb = val / 1024.0
            else if (u == "K") gb = val / 1024.0 / 1024.0
            else               gb = val / 1024.0 / 1024.0 / 1024.0   # bare number => bytes
            printf("%.4f", gb)
        }'
}

# --- parse `blue_quota` and return "USE_GB LIMIT_GB" for the GROUP section ---
# Output of blue_quota has TWO sections ("group 'guralnick'" then "user '...'"), each with a
# `blue2  <Use> <Quota> <Limit> <Grace> <Files>` data row. We must read the GROUP one, so we only
# consider data rows AFTER the line matching `group '<GROUP>'` and BEFORE the next "quotas for"
# header. Columns: $1=fs $2=Use $3=Quota $4=Limit. We use Use and Limit.
read_group_quota() {
    local raw
    raw="$(blue_quota 2>/dev/null)" || { echo "ERR blue_quota_failed"; return 1; }
    printf '%s\n' "$raw" | awk -v grp="$GROUP" '
        /quotas for/ { ingroup = ($0 ~ ("group .?" grp)) ? 1 : 0; next }
        ingroup && $1 ~ /^blue/ && $2 != "" {
            print $2, $4    # Use, Limit (raw tokens, e.g. 51.83T 52T)
            found = 1
            exit
        }
        END { if (!found) print "ERR no_group_row" }'
}

# --- email helper: prefer `mail`, else `sendmail`, else loud log + WARN sentinel ---
send_warn_email() {
    local subj="$1" body="$2"
    if command -v mail >/dev/null 2>&1; then
        printf '%s\n' "$body" | mail -s "$subj" "$MAIL_TO" \
            && { echo "[diskwatch] WARN email sent via mail to $MAIL_TO"; return 0; }
        echo "[diskwatch] mail failed; falling back to sendmail." >&2
    fi
    if command -v sendmail >/dev/null 2>&1; then
        printf 'To: %s\nSubject: %s\n\n%s\n' "$MAIL_TO" "$subj" "$body" \
            | sendmail -t \
            && { echo "[diskwatch] WARN email sent via sendmail to $MAIL_TO"; return 0; }
        echo "[diskwatch] sendmail failed; logging loudly + touching WARN sentinel." >&2
    fi
    # No working mailer: log loudly and drop the WARN sentinel so a human/other tooling notices.
    echo "[diskwatch] !!! DISK WARNING (no mailer available) !!! $subj :: $body" >&2
    { printf '%s\n%s\n' "$subj" "$body"; } > "$WARN_SENTINEL" 2>/dev/null || true
    return 0
}

# --- state for SOFT-warning debounce: warn once per downward crossing, re-arm above SOFT_GB ---
warned=0
paused=0
[[ -f "$PAUSE_SENTINEL" ]] && paused=1   # adopt any pre-existing pause state (idempotent restart)

# --- watchdog loop ---
while true; do
    if [[ -f "$STOP_SENTINEL" ]]; then
        echo "[diskwatch] STOP sentinel present ($STOP_SENTINEL) -> exiting cleanly."
        break
    fi

    qline="$(read_group_quota || true)"
    if [[ -z "$qline" || "$qline" == ERR* ]]; then
        echo "[diskwatch] $(date '+%F %T') could not read group '$GROUP' quota ('$qline'); retrying next cycle." >&2
    else
        use_tok="${qline%% *}"
        lim_tok="${qline##* }"
        use_gb="$(to_gb "$use_tok")"
        lim_gb="$(to_gb "$lim_tok")"
        free_gb="$(awk -v l="$lim_gb" -v u="$use_gb" 'BEGIN{printf("%.4f", l-u)}')"

        # Decide state transitions (all comparisons via awk for float safety).
        lt_soft="$(awk -v f="$free_gb" -v t="$SOFT_GB" 'BEGIN{print (f<t)?1:0}')"
        ge_soft="$(awk -v f="$free_gb" -v t="$SOFT_GB" 'BEGIN{print (f>=t)?1:0}')"
        lt_hard="$(awk -v f="$free_gb" -v t="$HARD_GB" 'BEGIN{print (f<t)?1:0}')"
        ge_release="$(awk -v f="$free_gb" -v t="$RELEASE_GB" 'BEGIN{print (f>=t)?1:0}')"

        state="OK"
        [[ "$lt_soft" == 1 ]] && state="SOFT"
        [[ "$lt_hard" == 1 ]] && state="HARD"

        echo "[diskwatch] $(date '+%F %T') free=${free_gb}GB (use=${use_gb}GB limit=${lim_gb}GB) " \
             "state=$state paused=$paused warned=$warned " \
             "[SOFT<$SOFT_GB HARD<$HARD_GB RELEASE>=$RELEASE_GB]"

        # SOFT: email once per downward crossing; re-arm only after free recovers above SOFT_GB.
        if [[ "$lt_soft" == 1 && "$warned" == 0 ]]; then
            send_warn_email \
                "[PhenoVision diskwatch] /blue group '$GROUP' LOW: free=${free_gb}GB < ${SOFT_GB}GB" \
                "Group '$GROUP' /blue free space is ${free_gb}GB (use=${use_gb}GB / limit=${lim_gb}GB), below the SOFT threshold of ${SOFT_GB}GB.
HARD threshold (auto-pause checkpoint emission) is ${HARD_GB}GB; emission resumes at ${RELEASE_GB}GB.
Host=$(hostname). Free up space (thin non-ladder checkpoints) before HARD is reached.
This is an automated, debounced warning from xAI/scripts/disk_watchdog.sh."
            warned=1
        elif [[ "$ge_soft" == 1 && "$warned" == 1 ]]; then
            echo "[diskwatch] free recovered above SOFT_GB=${SOFT_GB}GB -> re-arming SOFT warning."
            warned=0
            # Clear a fallback WARN sentinel if we'd dropped one (no mailer path).
            [[ -f "$WARN_SENTINEL" ]] && { rm -f "$WARN_SENTINEL" || true; }
        fi

        # HARD: create the PAUSE sentinel so trainers stop EMITTING (training keeps running).
        if [[ "$lt_hard" == 1 && "$paused" == 0 ]]; then
            {
                printf 'reason=disk_hard_threshold\nfree_gb=%s\nhard_gb=%s\nrelease_gb=%s\nhost=%s\nts=%s\n' \
                    "$free_gb" "$HARD_GB" "$RELEASE_GB" "$(hostname)" "$(date '+%F %T')"
            } > "$PAUSE_SENTINEL" 2>/dev/null || true
            echo "[diskwatch] HARD threshold crossed (free=${free_gb}GB < ${HARD_GB}GB) -> created PAUSE sentinel: $PAUSE_SENTINEL"
            paused=1
        # RELEASE: remove the PAUSE sentinel so emission resumes.
        elif [[ "$ge_release" == 1 && "$paused" == 1 ]]; then
            rm -f "$PAUSE_SENTINEL" 2>/dev/null || true
            echo "[diskwatch] free recovered to >= RELEASE_GB=${RELEASE_GB}GB (free=${free_gb}GB) -> removed PAUSE sentinel; emission may resume."
            paused=0
        fi
    fi

    # Exit promptly if a signal arrived during the cycle.
    [[ "$_TERM" == 1 ]] && { echo "[diskwatch] exiting on signal."; break; }

    # Sleep in 1s slices so a STOP sentinel / signal is honored within ~1s, not a full POLL.
    slept=0
    while [[ "$slept" -lt "$POLL" ]]; do
        [[ "$_TERM" == 1 ]] && break
        [[ -f "$STOP_SENTINEL" ]] && break
        sleep 1
        slept=$(( slept + 1 ))
    done
done

echo "Done: $(date)"

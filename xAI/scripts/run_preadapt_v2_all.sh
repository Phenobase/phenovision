#!/bin/bash
# PhenoVision preadapt v2 ALL-RUNS launcher (CPU/GPU two-pass collector split, round-replicated).
#
# Chains the 6 v2 runs (stable_evo only: {mae,plantclef,naive} x {seed 42, seed 43}) as TWO ROUNDS
# with a HARD BARRIER between them, so the FIRST replicate of all three conditions finishes before
# any SECOND replicate begins:
#
#     ROUND 1 (@s42, grid lines 1,2,3):  R1W1 = wave(1,2)  then  R1W2 = wave(3 solo)
#         ============== BARRIER: round 2 waits afterany on ALL of round 1's jobs ==============
#     ROUND 2 (@s43, grid lines 4,5,6):  R2W1 = wave(4,5)  then  R2W2 = wave(6 solo)
#
# WAVE STRUCTURE (run_preadapt_v2_wave.sh): each wave = up to 2 trainers (B200) + 1 shared GPU
# two-pass collector (L4) + N CPU two-pass collectors (hpg-default, 0 GPU) + per-run posthoc.
#
#   GPU INVARIANT (FIRM <= 3), enforced two ways:
#     * PER WAVE: 2 trainers + 1 GPU collector = 3 GPU (a solo wave = 1 + 1 = 2). CPU collectors +
#       posthoc add 0.
#     * WITHIN A ROUND: wave 2's trainers + collectors depend (afterany) on wave 1's collectors +
#       posthoc, so the two waves of a round are NOT co-resident -> still <= 3 GPU at any instant.
#     * ACROSS ROUNDS: round 2's first wave depends (afterany) on EVERY job of BOTH round-1 waves
#       (the BARRIER), so no round-2 GPU job starts until round 1 is entirely finished.
#
# TWO-PASS / TREE ISOLATION (FIRM): all v2 collectors run --require-passes gpu,cpu and watch ONLY the
# DISTINCT v2 tree xAI/output/preadapt_v2/. The LIVE v1 experiment (xAI/output/preadapt/, single-pass
# legacy collectors) is untouched. Never point a legacy collector at a v2 dir or vice-versa.
#
# DEFAULT BEHAVIOR: PRINT-ONLY (print the full DAG + the wave invocations; submit nothing). Pass
# --submit to actually submit; --dry-run is an explicit alias for the default.
#
# Usage:
#     bash xAI/scripts/run_preadapt_v2_all.sh              # print the DAG (default; no sbatch)
#     bash xAI/scripts/run_preadapt_v2_all.sh --dry-run    # same as default (explicit)
#     bash xAI/scripts/run_preadapt_v2_all.sh --submit     # actually submit the round-chained DAG
#
# -------------------------------------------------------------------------------------------------
# PRE-SUBMIT CHECK -- ALWAYS verify the group's GPU usage before launching:
#     module load ufrc && slurmInfo
#     squeue -A guralnick -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'
# Confirm headroom for up to 3 GPUs (2 B200 trainers + 1 L4 GPU collector per active wave).
# -------------------------------------------------------------------------------------------------

set -eo pipefail

PROJECT_ROOT=/blue/guralnick/share/r.dinnage/Projects/phenovision
cd "$PROJECT_ROOT"

WAVE="xAI/scripts/run_preadapt_v2_wave.sh"
GRID="xAI/two_noise/configs/experiment/preadapt_v2_grid.txt"

MODE="print"
case "${1:-}" in
    ""|--dry-run|--print) MODE="print" ;;
    --submit)             MODE="submit" ;;
    *) echo "ERROR: unknown arg '${1}'. Use --submit, --dry-run, or no arg (print)." >&2
       echo "Usage: bash xAI/scripts/run_preadapt_v2_all.sh [--submit|--dry-run]" >&2
       exit 1 ;;
esac

if [[ ! -f "$GRID" ]]; then
    echo "ERROR: v2 grid '$GRID' not found." >&2
    exit 1
fi

echo "================================================================================"
echo "preadapt v2 ALL-RUNS plan  (6 runs, 2 ROUNDS, hard barrier; <= 3 concurrent GPUs)"
echo "  mode: $MODE"
echo "  ROUND 1 (@s42): R1W1=wave(1,2)  ->  R1W2=wave(3 solo)"
echo "  ==== BARRIER: ROUND 2 waves wait afterany on ALL of ROUND 1's jobs ===="
echo "  ROUND 2 (@s43): R2W1=wave(4,5)  ->  R2W2=wave(6 solo)"
echo "  per wave: <=2 trainers (B200) + 1 GPU collector (L4) + N CPU collectors (0 GPU) + posthoc"
echo "  GPU invariant: per-wave <=3; intra-round wave2 afterany wave1; inter-round barrier."
echo "================================================================================"
echo

# Each wave prints a 'WAVE_JIDS <trainers> | <collectors> | <posthocs>' line; run_one_wave parses it
# (submit mode) and sets WAVE_GATE_JIDS (collectors+posthoc) + WAVE_ALL_JIDS (every id of the wave).

# run_one_wave <label> <gate_label> <gate_dep> <line1> [line2]
#   <label>      : short space-free wave label (e.g. R1W1) for readable print-mode placeholders.
#   <gate_label> : human-readable description of what this wave waits on (print mode only).
#   <gate_dep>   : SLURM dep clause for this wave's trainers/collectors (empty = start immediately).
#                  In print mode it is empty (no real ids yet); we DISPLAY <gate_label> instead.
#   prints the wave plan; sets WAVE_GATE_JIDS (collectors+posthoc, for the next intra-round wave)
#   and WAVE_ALL_JIDS (every id of this wave, for the round barrier).
WAVE_GATE_JIDS=""
WAVE_ALL_JIDS=""
run_one_wave() {
    local label="$1"; shift
    local gate_label="$1"; shift
    local gate="$1"; shift
    local -a wlines=("$@")
    local out
    if [[ "$MODE" == "print" ]]; then
        # Print path: show the wave invocation + the SYMBOLIC dependency (real ids unknown until
        # submit). Placeholders are SPACE-FREE so mk_afterany's `<*` skip-token test works on them.
        echo "  ${label} wave(${wlines[*]})${gate_label:+  [dep: $gate_label]}:"
        echo "      bash $WAVE ${wlines[*]}   # --submit threads the real afterany dep here"
        WAVE_GATE_JIDS="<${label}-collectors+posthoc>"
        WAVE_ALL_JIDS="<${label}-alljobs>"
        return 0
    fi
    # Submit path: actually run the wave with --submit and parse its WAVE_JIDS line.
    out="$(bash "$WAVE" "${wlines[@]}" ${gate:+"$gate"} --submit)"
    echo "$out"
    local line
    line="$(grep -E '^WAVE_JIDS ' <<<"$out" | tail -n1 || true)"
    if [[ -z "$line" ]]; then
        echo "[all] ERROR: wave(${wlines[*]}) did not emit a WAVE_JIDS line." >&2
        exit 1
    fi
    # WAVE_JIDS <trainers> | <collectors> | <posthocs>
    local trainers collectors posthocs
    trainers="${line#WAVE_JIDS }"; trainers="${trainers%% | *}"
    local rest="${line#*WAVE_JIDS }"
    rest="${rest#* | }"                       # drop trainers field
    collectors="${rest%% | *}"
    posthocs="${rest##* | }"
    WAVE_GATE_JIDS="${collectors} ${posthocs}"   # next intra-round wave gates on these (free GPUs)
    WAVE_ALL_JIDS="${trainers} ${collectors} ${posthocs}"
}

# Build a SLURM 'afterany:a:b:c' clause from a whitespace-separated id list (skip placeholders).
# Always returns 0 (an empty clause -- e.g. print-mode placeholders only -- echoes nothing); a
# nonzero return here would trip `set -e` inside the `$(...)` capture and abort the planner.
mk_afterany() {
    local ids="$1" clause=""
    local id
    for id in $ids; do
        [[ "$id" == \<* ]] && continue   # print-mode placeholder
        clause+="${clause:+:}${id}"
    done
    [[ -n "$clause" ]] && echo "afterany:${clause}"
    return 0
}

# ---- ROUND 1 (@s42): lines 1,2,3 -------------------------------------------------------------
echo "ROUND 1 (@s42):"
# R1W1 = wave(1,2), no dependency (starts immediately).
run_one_wave R1W1 "" "" 1 2
R1_ALL="$WAVE_ALL_JIDS"
R1W1_GATE="$WAVE_GATE_JIDS"

# R1W2 = wave(3 solo), gated on R1W1's collectors+posthoc (so the two round-1 waves are not co-resident).
echo
run_one_wave R1W2 "afterany R1W1 collectors+posthoc" "$(mk_afterany "$R1W1_GATE")" 3
R1_ALL="$R1_ALL $WAVE_ALL_JIDS"

echo
echo "  ==== BARRIER: ROUND 2 depends afterany on ALL of ROUND 1's jobs ===="
echo "       round1 jobs: $R1_ALL"
ROUND1_BARRIER="$(mk_afterany "$R1_ALL")"
echo

# ---- ROUND 2 (@s43): lines 4,5,6 -- first wave gated on the WHOLE of round 1 ------------------
echo "ROUND 2 (@s43):"
# R2W1 = wave(4,5), gated on the round-1 barrier (ALL of round 1).
run_one_wave R2W1 "afterany ALL of ROUND 1 (barrier)" "$ROUND1_BARRIER" 4 5
R2W1_GATE="$WAVE_GATE_JIDS"

# R2W2 = wave(6 solo), gated on R2W1's collectors+posthoc.
echo
run_one_wave R2W2 "afterany R2W1 collectors+posthoc" "$(mk_afterany "$R2W1_GATE")" 6

echo
echo "================================================================================"
echo "DAG:"
echo "  R1W1 wave(1,2) ──┐"
echo "                   ├─(afterany collectors+posthoc)→ R1W2 wave(3) ──┐"
echo "  (start now)  ────┘                                               │"
echo "                                                                   ▼"
echo "      ====================== ROUND BARRIER (afterany ALL of R1) ======================"
echo "                                                                   │"
echo "                                                                   ▼"
echo "  R2W1 wave(4,5) ─(afterany collectors+posthoc)→ R2W2 wave(6)"
echo
echo "  GPU concurrency: each active wave = <=2 trainers + 1 GPU collector = <=3 GPU."
echo "  Intra-round: wave2 afterany wave1's collectors+posthoc -> waves not co-resident."
echo "  Inter-round: round2 afterany ALL of round1 -> replicate s43 starts only after s42 done."
if [[ "$MODE" == "print" ]]; then
    echo
    echo "  PRINT-ONLY: nothing submitted. To submit the whole round-chained DAG:"
    echo "      bash xAI/scripts/run_preadapt_v2_all.sh --submit"
fi
echo "================================================================================"

#!/bin/bash
# =============================================================================
# launch_v2_round.sh ROUND ACCOUNT QOS QOSB
#   Submit one full preadapt-v2 round as sequential 1-trainer waves, fully
#   unattended (pure SLURM dependencies — no live session needed).
#
#   ROUND   : 1  (s42: naive -> plantclef -> mae)
#             2  (s43: naive -> plantclef -> mae)
#   ACCOUNT : SLURM account, e.g. rdinnage.fiu   (or guralnick)
#   QOS     : GPU QOS,        e.g. rdinnage.fiu   (or guralnick)
#   QOSB    : burst QOS for the CPU collectors, e.g. rdinnage.fiu-b
#
#   Example (run from repo root, after migration_setup.sh + env are ready):
#     bash xAI/scripts/launch_v2_round.sh 1 rdinnage.fiu rdinnage.fiu rdinnage.fiu-b
#
#   Each wave = 1 B200 trainer + 1 L4 GPU collector (heavy_every=8, 210 GB) +
#   5 burst CPU collectors. Trainers chain `afterany` (next starts after the
#   previous ENDS); each wave's collectors start `after` their trainer begins.
#   375 GB QOS fits a full trainer(104)+heavy-collector(210) wave.
# =============================================================================
set -eo pipefail
cd "$(dirname "$0")/../.."                      # repo root
ROUND="${1:?usage: launch_v2_round.sh ROUND ACCOUNT QOS QOSB [DEP_JOBID]}"
ACCT="${2:?account}"; QOS="${3:?qos}"; QOSB="${4:?burst qos}"
DEP="${5:-}"   # optional: seed the FIRST trainer with --dependency=afterany:<jobid>, e.g. to chain
               # round 2 behind round 1's last trainer so total GPU use never exceeds one wave.
GRID=xAI/two_noise/configs/experiment/preadapt_v2_grid.txt
STORE=xAI/output/preadapt_v2/_extract_store
D90=xAI/output/preadapt_v2/_d90_queue
case "$ROUND" in
  1) LINES="1 2 3" ;;
  2) LINES="4 5 6" ;;
  *) echo "ROUND must be 1 or 2"; exit 1 ;;
esac
[ -s "$GRID" ] || { echo "ERROR: $GRID missing (regenerate with gen_preadapt_grid_v2.py)"; exit 1; }

PREV="$DEP"
for L in $LINES; do
  args="$(sed -n "${L}p" "$GRID")"
  cond=$(echo "$args" | grep -oE -- '--condition [a-z]+' | awk '{print $2}')
  seed=$(echo "$args" | grep -oE -- '--seed [0-9]+' | awk '{print $2}')
  RUN="${cond}__stable_evo__s${seed}"
  RD="xAI/output/preadapt_v2/$RUN"
  dep=${PREV:+--dependency=afterany:$PREV}
  TJ=$(sbatch --parsable $dep --account="$ACCT" --qos="$QOS" \
        --job-name="preadapt_train_$RUN" \
        --export=ALL,PREADAPT_ARGS="$args --output_dir $RD" \
        xAI/scripts/submit_preadapt_train.sh)
  echo "trainer $RUN = $TJ ${dep:+(afterany $PREV)}"
  sbatch --parsable --dependency=after:"$TJ" --account="$ACCT" --qos="$QOS" \
    --partition=hpg-turin --gres=gpu:l4:1 --cpus-per-task=10 --mem-per-cpu=21G \
    --job-name="preadapt_cwgpu_$RUN" \
    --export=ALL,PREADAPT_RUN_DIRS=$RD,PREADAPT_WORKER_ID=0,PREADAPT_OUT_STORE=$STORE,PREADAPT_REQUIRE_PASSES=gpu,cpu,HEAVY_EVERY=8,D90_QUEUE=$D90,PREADAPT_TRAIN_JIDS=$TJ \
    xAI/scripts/submit_preadapt_collector_gpu.sh >/dev/null
  for w in 0 1 2 3 4; do
    sbatch --dependency=after:"$TJ" --account="$ACCT" --qos="$QOSB" \
      --job-name="preadapt_cwcpu${w}_$RUN" \
      --export=ALL,PREADAPT_RUN_DIRS=$RD,PREADAPT_WORKER_ID=$w,PREADAPT_OUT_STORE=$STORE,PREADAPT_REQUIRE_PASSES=gpu,cpu,PREADAPT_TRAIN_JIDS=$TJ,HEAVY_EVERY=8 \
      xAI/scripts/submit_preadapt_collector_cpu.sh >/dev/null
  done
  echo "  + 1 GPU collector (he=8, 210 GB) + 5 burst CPU collectors (dependency after:$TJ)"
  PREV=$TJ
done
echo "round $ROUND chain submitted under --account=$ACCT --qos=$QOS (CPU: $QOSB). It runs unattended."
echo "watch: squeue -u \$USER | grep preadapt   ;   results in xAI/output/preadapt_v2/<run>/ + _extract_store/"

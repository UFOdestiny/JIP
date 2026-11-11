#!/usr/bin/env bash
# End-to-end run for all backbones: data preparation, generation, main-judge annotations and unlabeled pools,
# probe training, then the evaluation of JIP and JIP + Judge on test and test2.
# With the optional baselines package present, its scores and probes are computed before the evaluation.
set -euo pipefail
source "$(dirname "$0")/common.sh"
"$PY" -m jip.pipeline.prepare_data --pools 2000,6000
serve gpt-oss-20b $JUDGE_PORT $JUDGE_GPU 0.85
for bb in $BACKBONES; do
  scripts/run_backbone.sh $bb gen &   # GPU $GEN_GPU, overlaps with judging of the previous backbone
  GEN_PID=$!
  [ -n "${PREV:-}" ] && scripts/run_backbone.sh $PREV judge
  wait $GEN_PID
  PREV=$bb
done
scripts/run_backbone.sh $PREV judge
for bb in $BACKBONES; do scripts/run_backbone.sh $bb pools; done  # pool generation needs $GEN_GPU as well
stop_server
for bb in $BACKBONES; do scripts/run_backbone.sh $bb probes; done
if [ -x baselines/scripts/run_baselines.sh ]; then
  for bb in $BACKBONES; do baselines/scripts/run_baselines.sh $bb; done
else
  echo "baselines/ not found: evaluating JIP only"
fi
for bb in $BACKBONES; do
  for split in test test2; do cpu jip.pipeline.evaluate --backbone $bb --split $split > "$JIP_LOGS/evaluate_${bb}_$split.log" 2>&1 & done
done
wait
stamp "main run done"

#!/usr/bin/env bash
# One backbone, in four stages:
#   gen    greedy generation and feature extraction                                              [GPU $GEN_GPU]
#   judge  main-judge annotations: correctness labels, answer-blind step labels and verdicts
#   pools  unlabeled train pools of the OOD domains (2,000 and 6,000 questions): generation,
#          answer-blind judge targets, features; the 6,000-question pools go to $JIP_POOL_ARTIFACTS
#   probes SAPLMA and JIP's probe (10 seeds each)                                                [GPU $GEN_GPU]
# judge and pools need the judge server on $JUDGE_PORT (scripts/run_main.sh starts it).
# Usage: scripts/run_backbone.sh <llama|qwen8b|ministral> <gen|judge|pools|probes>
set -euo pipefail
source "$(dirname "$0")/common.sh"
BB=$1; STAGE=$2
LOG="$JIP_LOGS/$BB"; mkdir -p "$LOG"
judge() { "$PY" -m jip.pipeline.judge --judge gptoss --backbone $BB --port $JUDGE_PORT "$@"; }
# GSM8K/MATH pools are separate question sets (pipeline.prepare_data --pools); the others subsample train.
pool_data() { case $1 in gsm8k|math500) echo "$JIP_DATASETS/pools/n$2" ;; *) echo "" ;; esac; }
build_pool() {  # <size> <datasets>
  local n=$1 ds
  for ds in $2; do
    run $GEN_GPU jip.pipeline.generate --backbone $BB --dataset $ds --splits train --max_samples $n \
        --gpu_memory_utilization 0.4 --data_root "$(pool_data $ds $n)"
    for task in steps final_blind; do judge --task $task --dataset $ds --splits train; done
    run $GEN_GPU jip.pipeline.features --backbone $BB --dataset $ds --splits train
  done
}

case $STAGE in
gen)
  stamp "$BB gen"
  for ds in $DATASETS; do run $GEN_GPU jip.pipeline.generate --backbone $BB --dataset $ds --splits "$(splits_of $ds)"; done > "$LOG/generate.log" 2>&1
  for ds in $DATASETS; do run $GEN_GPU jip.pipeline.features --backbone $BB --dataset $ds --splits "$(splits_of $ds)"; done > "$LOG/features.log" 2>&1
  ;;
judge)
  stamp "$BB judge"
  {
    # Reference-based correctness labels where answer matching does not apply.
    for ds in hotpotqa musique babi; do judge --task correct --dataset $ds --splits "$(splits_of $ds)"; done
    # Answer-blind step labels and verdicts: imitation targets on HotpotQA train, judge output J elsewhere.
    for ds in $DATASETS; do for task in steps final_blind; do judge --task $task --dataset $ds --splits "$(splits_of $ds)"; done; done
  } > "$LOG/judge.log" 2>&1
  ;;
pools)
  stamp "$BB pools"
  build_pool 2000 "$OOD" > "$LOG/pools.log" 2>&1  # StrategyQA: all 1,443 train questions
  ( export JIP_ARTIFACTS="$JIP_POOL_ARTIFACTS"; build_pool 6000 "musique babi gsm8k math500" ) >> "$LOG/pools.log" 2>&1
  ;;
probes)
  stamp "$BB probes"
  run $GEN_GPU jip.pipeline.train --backbone $BB --methods saplma > "$LOG/train_saplma.log" 2>&1
  run $GEN_GPU jip.pipeline.train --backbone $BB --methods jip > "$LOG/train_jip.log" 2>&1
  ;;
*) echo "unknown stage $STAGE" >&2; exit 1 ;;
esac
stamp "$BB $STAGE done"

#!/usr/bin/env bash
# Runs the silva24b reproduction grid (2 backbones x 4 heads) on CPU, 4 jobs in parallel.
# Reduced budget for CPU: max EPOCHS epochs, early stopping patience 2, seeds given in SEEDS.
cd "$(dirname "$0")/.."
EPOCHS=${EPOCHS:-5}; SEEDS=${SEEDS:-0}; PAR=${PAR:-4}
mkdir -p results/nip/logs
for s in $SEEDS; do for h in none sqn eval hist; do for b in sasrec gru; do
  echo "$b $h $s"; done; done; done | \
xargs -P "$PAR" -L 1 bash -c 'python3 scripts/train_nip.py --backbone $0 --head $1 --seed $2 --epochs '"$EPOCHS"' --patience 2 --threads 1 > results/nip/logs/$0-$1-s$2.log 2>&1'
echo done > results/nip/logs/ALL_DONE

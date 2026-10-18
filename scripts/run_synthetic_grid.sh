#!/usr/bin/env bash
# Step S5 grid: N in {10, 20, 40}; LSVI-UCB only where enumerating all assortments is feasible.
cd "$(dirname "$0")/.."
mkdir -p results/synthetic/logs
python3 scripts/run_synthetic.py --N 10 --agents myopic mnlvql lsvi > results/synthetic/logs/N10.log 2>&1
python3 scripts/run_synthetic.py --N 20 --agents myopic mnlvql lsvi > results/synthetic/logs/N20.log 2>&1
python3 scripts/run_synthetic.py --N 40 --agents myopic mnlvql > results/synthetic/logs/N40.log 2>&1
echo done > results/synthetic/logs/ALL_DONE

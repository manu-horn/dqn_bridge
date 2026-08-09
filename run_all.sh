#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-42 43 44}"
EPISODES="${EPISODES:-500}"

for agent in dqn.py double_dqn.py reinforce.py; do
    for seed in $SEEDS; do
        echo "==> Training $agent (seed $seed, ${EPISODES} episodes)"
        "$PYTHON" "$agent" --seed "$seed" --episodes "$EPISODES"
    done
done

echo "==> Comparing runs"
"$PYTHON" utils.py
#!/usr/bin/env bash
for i in $(seq 3 6); do
    echo "LEVEL $i"
    [[ $i -le 4 ]] && bs=32 || bs=16
    python unitransform.py --level $i --ckpt-level $(($i - 1)) --resume --batch-size $bs --epochs 10
done
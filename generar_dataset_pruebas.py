#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synthetic dataset generator for software test cases.

Usage:
    python generate_test_dataset.py --n 200 --out ./output --seed 42

Output:
    - synthetic_test_dataset.csv
    - synthetic_test_dataset.json
in the folder specified by --out (created if it does not exist).
"""

import argparse
import json
import os
from pathlib import Path
from random import Random
from csv import DictWriter

# ---------------------------
# Configuration and catalogs
# ---------------------------
TEST_TYPE   = ["Functional", "Non-functional"]
METHODOLOGY = ["Agile", "Waterfall"]
PHASE       = ["Static", "Dynamic"]
ORIGIN      = ["Manual", "Automated"]
COMPLEXITY  = ["Low", "Medium", "High", "Critical"]
PRIORITY    = ["Low", "Medium", "High", "Critical"]

# Optional distributions (you can adjust them)
# They should sum ~1.0 (not strictly required, they are normalized).
WEIGHTS_COMPLEXITY = [0.25, 0.35, 0.25, 0.15]   # Low..Critical
WEIGHTS_PRIORITY   = [0.30, 0.35, 0.25, 0.10]   # Low..Critical

# ---------------------------
# Probability logic
# ---------------------------
def failure_probability(origin_val: str, compl_val: str, prior_val: str, base: float = 0.20) -> float:
    """
    Calculate the failure probability according to pseudocode rules.
    Adjust the increments as needed.
    """
    p = base
    if origin_val == "Manual":
        p += 0.15
    if compl_val == "High":
        p += 0.20
    elif compl_val == "Critical":
        p += 0.30
    if prior_val == "High":
        p += 0.10
    elif prior_val == "Critical":
        p += 0.15

    # Clamp to avoid exact 0/1 extremes
    if p < 0.05:
        p = 0.05
    if p > 0.95:
        p = 0.95
    return float(round(p, 6))

# ---------------------------
# Utilities
# ---------------------------
def sample_choice(rng: Random, options, weights=None):
    """
    Select one element from 'options'.
    If 'weights' are provided, use weighted choice.
    """
    if not weights:
        return rng.choice(options)
    # Lightweight implementation of random.choices for 1 element (Python 3.6+ compatible)
    # Normalize weights:
    total = float(sum(weights))
    threshold = rng.random()
    accum = 0.0
    for opt, w in zip(options, weights):
        accum += (w / total)
        if threshold <= accum:
            return opt
    return options[-1]

def generate_records(n: int, seed: int = 42):
    rng = Random(seed)
    records = []
    for i in range(1, n + 1):
        test_id = f"TC{i:06d}"
        ttype = sample_choice(rng, TEST_TYPE)
        meth  = sample_choice(rng, METHODOLOGY)
        phas  = sample_choice(rng, PHASE)
        orig  = sample_choice(rng, ORIGIN)
        comp  = sample_choice(rng, COMPLEXITY, WEIGHTS_COMPLEXITY)
        prio  = sample_choice(rng, PRIORITY, WEIGHTS_PRIORITY)

        p_fail = failure_probability(orig, comp, prio)
        failed = rng.random() < p_fail
        result = "Failed" if failed else "Passed"

        records.append({
            "testId": test_id,
            "type": ttype,
            "methodology": meth,
            "phase": phas,
            "origin": orig,
            "complexity": comp,
            "priority": prio,
            "result": result,
            "p_fail_model": round(p_fail, 3)  # useful for synthetic dataset audit
        })
    return records

def save_csv(records, csv_path: Path):
    fields = ["testId","type","methodology","phase","origin","complexity","priority","result","p_fail_model"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

def save_json(records, json_path: Path):
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Synthetic dataset generator for software test cases.")
    parser.add_argument("--n", type=int, default=200, help="Number of records to generate (default: 200)")
    parser.add_argument("--out", type=str, default="./output", help="Output folder (default: ./output)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = generate_records(args.n, seed=args.seed)

    csv_path  = out_dir / "synthetic_test_dataset.csv"
    json_path = out_dir / "synthetic_test_dataset.json"

    save_csv(records, csv_path)
    save_json(records, json_path)

    print(f"[OK] Generated {args.n} records.")
    print(f"[OK] CSV : {csv_path.resolve()}")
    print(f"[OK] JSON: {json_path.resolve()}")

if __name__ == "__main__":
    main()

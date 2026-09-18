#!/usr/bin/env python3
"""Aggregate Gate A v8 summaries without selecting any pose post hoc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gate_a_io import write_csv, write_json
from gate_a_protocol import PROTOCOL_VERSION


METRICS = (
    "sr_reach",
    "sr_ref",
    "delta_reach",
    "sampled_pose_success_std",
    "sampled_pose_success_range",
)


def discover_summaries(result_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(result_root.rglob("summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("protocol_version") != PROTOCOL_VERSION:
            continue
        if "sr_reach" not in payload:
            continue
        rows.append({**payload, "summary_path": str(path.resolve())})
    if not rows:
        raise RuntimeError(f"no Gate A v8 scene summaries under {result_root}")
    return rows


def _hierarchical_bootstrap_mean(
    rows: Sequence[dict[str, Any]],
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """Resample policy seeds, then scenes within each sampled policy seed."""
    rng = np.random.default_rng(seed)
    by_policy: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_policy.setdefault(int(row["policy_seed"]), []).append(row)
    policies = sorted(by_policy)
    values = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled_policies = rng.choice(policies, size=len(policies), replace=True)
        policy_means = []
        for policy in sampled_policies:
            scene_rows = by_policy[int(policy)]
            sampled_indices = rng.integers(0, len(scene_rows), size=len(scene_rows))
            policy_means.append(
                float(np.mean([float(scene_rows[i][metric]) for i in sampled_indices]))
            )
        values[index] = float(np.mean(policy_means))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def aggregate_group(
    rows: Sequence[dict[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "task_name": rows[0]["task_name"],
        "policy_name": rows[0]["policy_name"],
        "scene_policy_count": len(rows),
        "policy_seed_count": len({int(row["policy_seed"]) for row in rows}),
        "scene_count": len({str(row["scene_id"]) for row in rows}),
        "fraction_delta_reach_gt_zero": float(
            np.mean([float(row["delta_reach"]) > 0 for row in rows])
        ),
    }
    for metric_index, metric in enumerate(METRICS):
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        low, high = _hierarchical_bootstrap_mean(
            rows, metric, iterations=iterations, seed=seed + metric_index
        )
        result[f"mean_{metric}"] = float(np.mean(values))
        result[f"{metric}_ci95_low"] = low
        result[f"{metric}_ci95_high"] = high
    return result


def aggregate(rows: Sequence[dict[str, Any]], *, iterations: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task_name"]), str(row["policy_name"])), []).append(row)
    return [
        aggregate_group(group, iterations=iterations, seed=seed + index * 100)
        for index, (_, group) in enumerate(sorted(grouped.items()))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    args = parser.parse_args()
    if args.bootstrap_iterations < 100:
        parser.error("bootstrap iterations must be at least 100")
    rows = discover_summaries(args.result_root)
    aggregated = aggregate(
        rows, iterations=args.bootstrap_iterations, seed=args.bootstrap_seed
    )
    output_dir = args.output_dir or args.result_root / "analysis"
    write_csv(rows, output_dir / "scene_policy_results.csv")
    write_csv(aggregated, output_dir / "task_results.csv")
    write_json(
        {
            "protocol_version": PROTOCOL_VERSION,
            "bootstrap": "policy seeds, then scenes within policy seed",
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
            "groups": aggregated,
        },
        output_dir / "aggregate_summary.json",
    )
    print(output_dir.resolve())


if __name__ == "__main__":
    main()

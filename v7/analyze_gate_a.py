#!/usr/bin/env python3
"""Aggregate Gate A v7 geometry-feasibility versus policy-success-basin results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from gate_a_protocol import (
    ACCEPTED_VALIDATION_LABELS,
    DIAGNOSTIC_LABELS,
    PROTOCOL_VERSION,
    VALIDATION_LABELS,
)


HEADROOMS = ("H_geom", "H_train")
DERIVED_METRICS = HEADROOMS


def read_rows(paths):
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["_source_file"] = str(path)
                rows.append(row)
    return rows


def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def interval(values):
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def effect_values(rates: Mapping[str, float]) -> dict[str, float]:
    return {
        "H_geom": rates["p_oracle"] - rates["p_geom"],
        "H_train": rates["p_oracle"] - rates["p_train"],
    }


def build_clusters(rows, required_split="test"):
    clusters = {}
    for row in rows:
        split = row.get("experiment_split") or "test"
        if split != required_split:
            continue
        if int(row.get("protocol_version", -1)) != PROTOCOL_VERSION:
            raise ValueError("Gate A v7 analysis rejects historical or unversioned results")
        if str(row.get("failure_type", "")).startswith("rollout_error"):
            raise RuntimeError("rollout_error is not a policy failure; rerun this shard")
        labels = [label for label in row["evaluation_labels"].split("+") if label]
        unknown = sorted(set(labels) - set(ACCEPTED_VALIDATION_LABELS))
        if unknown:
            raise ValueError(f"unknown baseline labels {unknown} in {row['_source_file']}")
        key = (row["task"], int(row.get("policy_seed") or 1), int(row["scene_id"]))
        cluster = clusters.setdefault(
            key,
            {
                "labels": {name: {} for name in ACCEPTED_VALIDATION_LABELS},
                "candidate_ids": {},
            },
        )
        comparison_seed = int(row["comparison_seed"] if row.get("comparison_seed") is not None else row["seed"])
        candidate_id = int(row["candidate_id"])
        outcome = int(as_bool(row["success"]))
        for label in labels:
            previous = cluster["labels"][label].get(comparison_seed)
            if previous is not None and previous != outcome:
                raise ValueError(f"conflicting outcome for {key}/{label}/seed {comparison_seed}")
            cluster["labels"][label][comparison_seed] = outcome
            if label in cluster["candidate_ids"] and cluster["candidate_ids"][label] != candidate_id:
                raise ValueError(f"selector changed candidate within {key}/{label}")
            cluster["candidate_ids"][label] = candidate_id

    if not clusters:
        raise RuntimeError(f"no validation rows for experiment_split={required_split!r}")
    incomplete = {}
    unpaired = {}
    for key, cluster in clusters.items():
        missing = [name for name in VALIDATION_LABELS if not cluster["labels"][name]]
        if missing:
            incomplete[str(key)] = missing
            continue
        present_labels = [
            name for name in ACCEPTED_VALIDATION_LABELS if cluster["labels"][name]
        ]
        seed_sets = [set(cluster["labels"][name]) for name in present_labels]
        if any(value != seed_sets[0] for value in seed_sets[1:]):
            unpaired[str(key)] = {
                name: sorted(cluster["labels"][name]) for name in present_labels
            }
    if incomplete:
        raise RuntimeError(f"incomplete validation baselines: {incomplete}")
    if unpaired:
        raise RuntimeError(f"validation comparison seeds are not paired: {unpaired}")
    return clusters


def cluster_rates(cluster):
    return {
        name: float(np.mean(list(cluster["labels"][name].values())))
        for name in ACCEPTED_VALIDATION_LABELS
        if cluster["labels"][name]
    }


def point_summary(clusters):
    per_scene = []
    for (task, policy_seed, scene_id), cluster in sorted(clusters.items()):
        rates = cluster_rates(cluster)
        per_scene.append(
            {
                "task": task,
                "policy_seed": policy_seed,
                "scene_id": scene_id,
                **rates,
                **effect_values(rates),
                "validation_rollouts": len(next(iter(cluster["labels"].values()))),
                "selector_candidate_ids": cluster["candidate_ids"],
                "selector_pose_overlaps": len(cluster["candidate_ids"])
                - len(set(cluster["candidate_ids"].values())),
            }
        )
    return per_scene


def _draw_cluster_rates(cluster, rng):
    seeds = sorted(next(iter(cluster["labels"].values())))
    sampled = rng.choice(seeds, size=len(seeds), replace=True)
    return {
        name: float(np.mean([cluster["labels"][name][int(seed)] for seed in sampled]))
        for name in VALIDATION_LABELS
    }


def hierarchical_bootstrap(clusters, samples, seed):
    """Resample task-policy units, scenes within unit, then paired rollout seeds."""
    rng = np.random.default_rng(seed)
    task_policy_units = sorted({key[:2] for key in clusters})
    scenes_by_unit = {
        unit: sorted([key for key in clusters if key[:2] == unit])
        for unit in task_policy_units
    }
    draws = {name: [] for name in (*VALIDATION_LABELS, *DERIVED_METRICS)}
    for _ in range(samples):
        sampled_unit_indices = rng.choice(
            len(task_policy_units), size=len(task_policy_units), replace=True
        )
        sample_rates = []
        for unit_index in sampled_unit_indices:
            keys = scenes_by_unit[task_policy_units[int(unit_index)]]
            sampled_indices = rng.choice(len(keys), size=len(keys), replace=True)
            for index in sampled_indices:
                sample_rates.append(_draw_cluster_rates(clusters[keys[int(index)]], rng))
        macro = {
            name: float(np.mean([rates[name] for rates in sample_rates]))
            for name in VALIDATION_LABELS
        }
        cluster_effects = [effect_values(rates) for rates in sample_rates]
        effects = {
            name: float(np.mean([values[name] for values in cluster_effects]))
            for name in DERIVED_METRICS
        }
        for name, value in {**macro, **effects}.items():
            draws[name].append(value)
    return {name: interval(values) for name, values in draws.items()}


def macro_summary(items):
    result = {
        name: float(np.mean([item[name] for item in items]))
        for name in (*VALIDATION_LABELS, *DERIVED_METRICS)
    }
    for name in DIAGNOSTIC_LABELS:
        available = [item[name] for item in items if name in item]
        result[name] = float(np.mean(available)) if available else None
        result[f"{name}_coverage"] = len(available)
    return result


def basin_evidence(rows, required_split):
    """Summarize the sampled P_geo versus empirical basin relationship."""
    groups = {}
    for row in rows:
        if (row.get("experiment_split") or "test") != required_split:
            continue
        if int(row.get("protocol_version", -1)) != PROTOCOL_VERSION:
            raise ValueError("Gate A v7 analysis rejects historical basin rows")
        if not as_bool(row.get("geometry_feasible")):
            raise ValueError("GT-basin rows must come from the geometry-feasible set")
        key = (row["task"], int(row.get("policy_seed") or 1), int(row["scene_id"]))
        groups.setdefault(key, []).append(row)
    per_scene = []
    for (task, policy_seed, scene_id), values in sorted(groups.items()):
        rates = np.asarray([float(row["success_rate"]) for row in values], dtype=float)
        members = np.asarray([as_bool(row["basin_member"]) for row in values], dtype=bool)
        per_scene.append(
            {
                "task": task,
                "policy_seed": policy_seed,
                "scene_id": scene_id,
                "sampled_geometry_feasible_candidates": len(values),
                "basin_members": int(members.sum()),
                "geometry_feasible_outside_basin": int((~members).sum()),
                "mixed_basin_membership": bool(members.any() and not members.all()),
                "success_rate_min": float(rates.min()),
                "success_rate_max": float(rates.max()),
                "success_rate_range": float(np.ptp(rates)),
                "success_rate_std": float(np.std(rates)),
            }
        )
    outside_clusters = sum(
        item["geometry_feasible_outside_basin"] > 0 for item in per_scene
    )
    mixed_clusters = sum(item["mixed_basin_membership"] for item in per_scene)
    return {
        "scope": "rollout-labelled sample of the geometry-feasible pool",
        "scene_policy_clusters": len(per_scene),
        "clusters_with_geometry_feasible_pose_outside_basin": outside_clusters,
        "clusters_with_mixed_basin_membership": mixed_clusters,
        "observed_geometry_basin_mismatch": bool(outside_clusters),
        "observed_within_geometry_policy_variation": bool(mixed_clusters),
        "per_scene": per_scene,
    }


def summarize(
    rows,
    candidate_rows,
    bootstrap_samples,
    seed,
    split,
    minimum_effect,
    required_positive_tasks,
    minimum_oracle_success,
    basin_rows=(),
):
    clusters = build_clusters(rows, required_split=split)
    per_scene = point_summary(clusters)
    tasks = sorted({item["task"] for item in per_scene})
    per_task = {}
    for task in tasks:
        items = [item for item in per_scene if item["task"] == task]
        task_clusters = {key: value for key, value in clusters.items() if key[0] == task}
        per_task[task] = {
            **macro_summary(items),
            "ci95": hierarchical_bootstrap(task_clusters, bootstrap_samples, seed),
            "policy_seed_count": len({item["policy_seed"] for item in items}),
            "scene_policy_clusters": len(items),
        }

    overall = macro_summary(per_scene)
    overall["ci95"] = hierarchical_bootstrap(clusters, bootstrap_samples, seed)
    qualifying_tasks = [
        task
        for task, values in per_task.items()
        if values["ci95"]["H_geom"][0] >= minimum_effect
        and values["p_oracle"] >= minimum_oracle_success
        
    ]
    gate_passed = split == "test" and len(qualifying_tasks) >= required_positive_tasks

    pool_audit = {
        "candidate_rows": len(candidate_rows),
        "geometry_eligible_rows": sum(
            as_bool(row.get("geometry_eligible")) for row in candidate_rows
        ),
        "trajectory_unreachable_rows": sum(
            not as_bool(row.get("trajectory_reachable")) for row in candidate_rows
        ),
        "default_training_pose_proxy_rows": sum(
            as_bool(row.get("default_training_pose_proxy")) for row in candidate_rows
        ),
    }
    rollout_errors = sum(
        str(row.get("failure_type", "")).startswith("rollout_error") for row in rows
    )
    basin_audit = basin_evidence(basin_rows, split)
    audit = {
        "experiment_split": split,
        "task_count": len(tasks),
        "policy_seed_count": len({key[1] for key in clusters}),
        "scene_policy_clusters": len(clusters),
        "validation_rows": len(rows),
        "validation_rollout_errors": rollout_errors,
        "paired_seed_check": "passed",
        "pool": pool_audit,
        "basin_evidence": basin_audit,
    }
    decision = {
        "status": ("calibration_only" if split != "test" else ("pass" if gate_passed else "no_go")),
        "primary_headroom": "H_geom",
        "minimum_effect": minimum_effect,
        "minimum_oracle_success": minimum_oracle_success,
        "required_positive_tasks": required_positive_tasks,
        "qualifying_tasks": qualifying_tasks,
        "qualifying_task_count": len(qualifying_tasks),
        "observed_geometry_basin_mismatch": basin_audit[
            "observed_geometry_basin_mismatch"
        ],
        "reason": (
            "calibration split; no formal Gate-A decision"
            if split != "test"
            else (
                "preregistered Gate-A conditions met"
                if gate_passed
                else "preregistered Gate-A conditions not met"
            )
        ),
    }
    main_table = [
        {
            "task": task,
            "geometry_success_rate": values["p_geom"],
            "training_success_rate": values["p_train"],
            "oracle_success_rate": values["p_oracle"],
            "mobipi_success_rate": values.get("p_mobipi"),
            "H_geom": values["H_geom"],
            "H_train": values["H_train"],
            "H_geom_ci95": values["ci95"]["H_geom"],
        }
        for task, values in per_task.items()
    ]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "audit": audit,
        "decision": decision,
        "per_scene": per_scene,
        "per_task": per_task,
        "main_table": main_table,
        "overall": overall,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--split", choices=("calibration", "test"), default="test")
    parser.add_argument("--minimum-effect", type=float, default=0.10)
    parser.add_argument("--required-positive-tasks", type=int, default=2)
    parser.add_argument("--minimum-oracle-success", type=float, default=0.20)
    parser.add_argument("--expected-validation-rollouts", type=int, default=20)
    args = parser.parse_args()
    validation_paths = sorted(args.result_root.rglob("validation_rollouts.csv"))
    if not validation_paths:
        raise FileNotFoundError(
            f"no validation_rollouts.csv below {args.result_root}"
        )
    candidate_paths = sorted(args.result_root.rglob("candidate_pool_all.csv"))
    basin_paths = sorted(args.result_root.rglob("gt_basin_all.csv"))
    if not basin_paths:
        raise FileNotFoundError(
            f"no gt_basin_all.csv below {args.result_root}"
        )
    validation_rows = read_rows(validation_paths)
    for key, cluster in build_clusters(validation_rows, required_split=args.split).items():
        if any(
            outcomes and len(outcomes) != args.expected_validation_rollouts
            for outcomes in cluster["labels"].values()
        ):
            raise RuntimeError(f"incomplete validation count for {key}")
    result = summarize(
        validation_rows,
        read_rows(candidate_paths),
        args.bootstrap_samples,
        args.seed,
        args.split,
        args.minimum_effect,
        args.required_positive_tasks,
        args.minimum_oracle_success,
        read_rows(basin_paths),
    )
    result["source_files"] = [str(path) for path in validation_paths]
    result["candidate_source_files"] = [str(path) for path in candidate_paths]
    result["basin_source_files"] = [str(path) for path in basin_paths]
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

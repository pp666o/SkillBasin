#!/usr/bin/env python3
"""Dependency-light synthetic smoke test for Gate A v8 artifacts.

The generated numbers are synthetic implementation checks, not experimental
evidence and must not be copied into a paper result table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gate_a_io import write_json, write_scene_artifacts
from gate_a_protocol import (
    DEFAULT_CONFIG,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    evaluate_reachability_and_reference,
    paired_rollout_seeds,
    sample_reachability_poses,
    summarize_gate_a,
)


def run_smoke(output_root: Path) -> dict:
    task_name = "SyntheticTurnOnFixture"
    scene_id = 0
    policy_seed = 1
    reference_pose = np.asarray([0.0, 0.0, 0.0])
    target_xy = np.asarray([0.0, 0.0])
    sampled, attempts = sample_reachability_poses(
        reference_pose,
        target_xy,
        num_poses=4,
        seed=8008,
        square_side_m=1.0,
        target_radius_m=1.0,
        yaw_jitter_deg=30.0,
        max_attempts=100,
    )
    seeds = paired_rollout_seeds(8008, policy_seed, scene_id, 2)

    def synthetic_rollout_runner(pose, seed, pose_id, pose_type):
        jitter = 0.12 if int(seed) % 2 else -0.12
        success = bool(float(pose[0]) + 0.35 * float(pose[1]) + jitter >= 0.0)
        return {
            "success": success,
            "termination_reason": "synthetic_success" if success else "synthetic_failure",
            "episode_length": 2,
            "video_path": "",
            "error": "",
            "synthetic_smoke_test": True,
        }

    evaluator_kwargs = {
        "task_name": task_name,
        "scene_id": scene_id,
        "policy_name": "synthetic_frozen_policy",
        "policy_checkpoint": "SYNTHETIC_NO_CHECKPOINT",
        "policy_version": "synthetic-smoke-v1",
        "policy_seed": policy_seed,
    }
    sampled_results, reference_result = evaluate_reachability_and_reference(
        sampled_poses=sampled,
        reference_pose=reference_pose,
        rollout_seeds=seeds,
        evaluator_kwargs=evaluator_kwargs,
        rollout_runner=synthetic_rollout_runner,
    )
    summary = summarize_gate_a(sampled_results, reference_result)
    scene_dir = output_root / task_name / "policy_seed_1" / "scene_0"
    config = {
        **DEFAULT_CONFIG,
        "num_sampled_poses": 4,
        "rollouts_per_pose": 2,
        "max_sampling_attempts": 100,
        "protocol_seed": 8008,
        "synthetic_smoke_test": True,
    }
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "synthetic_smoke_test": True,
        "not_experimental_evidence": True,
        "task_name": task_name,
        "scene_id": scene_id,
        "policy_seed": policy_seed,
        "reference_pose_source": "synthetic_smoke_fixture",
        "reference_pose": reference_pose.tolist(),
        "target_definition": "synthetic_origin",
        "target_xy": target_xy.tolist(),
        "rollout_seed_contract": "same exact K seeds for samples and reference",
    }
    write_scene_artifacts(
        scene_dir=scene_dir,
        config=config,
        metadata=metadata,
        sampling_attempts=attempts,
        sampled_results=sampled_results,
        reference_result=reference_result,
        summary=summary,
        target_xy=target_xy,
    )
    write_json(
        {
            **summary.to_dict(),
            "synthetic_smoke_test": True,
            "not_experimental_evidence": True,
        },
        scene_dir / "summary.json",
    )
    required = [
        scene_dir / "config.json",
        scene_dir / "metadata.json",
        scene_dir / "poses.csv",
        scene_dir / "sampling_attempts.csv",
        scene_dir / "rollouts.csv",
        scene_dir / "summary.json",
        scene_dir / "figures" / "pose_success_map.png",
        scene_dir / "figures" / "pose_success_distribution.png",
        scene_dir / "figures" / "reach_vs_reference.png",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"smoke artifacts missing or empty: {missing}")
    print(scene_dir.resolve())
    print(summary.to_dict())
    return summary.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "smoke_output",
    )
    args = parser.parse_args()
    run_smoke(args.output_root)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pure protocol logic for Gate A v8.

This module intentionally has no RoboCasa, MuJoCo, or policy dependency.  It
defines the preregistered reachability-region sampler, the one common pose
evaluation path, and the only primary Gate A statistics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


PROTOCOL_VERSION = 8
PROTOCOL_NAME = "n2m_reachability_region"
POSE_TYPE_REACH = "reachability_sample"
POSE_TYPE_REFERENCE = "reference"

DEFAULT_CONFIG = {
    "protocol_version": PROTOCOL_VERSION,
    "protocol_name": PROTOCOL_NAME,
    "num_sampled_poses": 48,
    "rollouts_per_pose": 5,
    "square_side_m": 1.0,
    "target_radius_m": 1.0,
    "yaw_jitter_deg": 30.0,
    "max_sampling_attempts": 5000,
    "protocol_seed": 20260915,
    "basin_analysis_enabled": False,
    "basin_success_threshold": None,
    "legacy_geometry_enabled": False,
    "legacy_oracle_enabled": False,
    "legacy_waypoint_ik_enabled": False,
}


def wrap_angle(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass(frozen=True)
class ReachabilityPose:
    pose_id: str
    sample_index: int
    x: float
    y: float
    yaw: float
    sampling_seed: int

    @property
    def array(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.yaw], dtype=float)


@dataclass
class PoseEvaluation:
    protocol_version: int
    task_name: str
    scene_id: str
    policy_name: str
    policy_checkpoint: str
    policy_version: str
    policy_seed: int
    pose_id: str
    pose_type: str
    x: float
    y: float
    yaw: float
    num_rollouts: int
    success_count: int
    success_rate: float
    rollout_seeds: list[int]
    sampling_seed: int | None = None
    is_collision_free: bool = True
    rollout_records: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def pose_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("rollout_records", None)
        row["rollout_seeds"] = list(self.rollout_seeds)
        return row


@dataclass
class GateASummary:
    protocol_version: int
    protocol_name: str
    task_name: str
    scene_id: str
    policy_name: str
    policy_checkpoint: str
    policy_version: str
    policy_seed: int
    num_sampled_poses: int
    rollouts_per_pose: int
    reference_rollouts: int
    sr_reach: float
    sr_ref: float
    delta_reach: float
    sampled_pose_success_min: float
    sampled_pose_success_max: float
    sampled_pose_success_mean: float
    sampled_pose_success_std: float
    sampled_pose_success_median: float
    sampled_pose_success_q25: float
    sampled_pose_success_q75: float
    sampled_pose_success_range: float
    zero_success_pose_count: int
    full_success_pose_count: int
    basin_analysis_enabled: bool
    basin_success_threshold: float | None
    basin_pose_count: int | None
    basin_pose_fraction: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_default_config(config: Mapping[str, Any]) -> None:
    """Reject attempts to silently re-enable the retired Gate A v7 path."""
    if int(config.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise ValueError(f"Gate A v8 requires protocol_version={PROTOCOL_VERSION}")
    if int(config.get("num_sampled_poses", 0)) <= 0:
        raise ValueError("num_sampled_poses must be positive")
    if int(config.get("rollouts_per_pose", 0)) <= 0:
        raise ValueError("rollouts_per_pose must be positive")
    for key in (
        "square_side_m",
        "target_radius_m",
        "yaw_jitter_deg",
    ):
        if float(config.get(key, 0.0)) <= 0.0:
            raise ValueError(f"{key} must be positive")
    if int(config.get("max_sampling_attempts", 0)) < int(
        config["num_sampled_poses"]
    ):
        raise ValueError("max_sampling_attempts must cover all requested poses")
    forbidden = {
        "legacy_geometry_enabled": config.get("legacy_geometry_enabled", False),
        "legacy_oracle_enabled": config.get("legacy_oracle_enabled", False),
        "legacy_waypoint_ik_enabled": config.get(
            "legacy_waypoint_ik_enabled", False
        ),
    }
    enabled = [key for key, value in forbidden.items() if bool(value)]
    if enabled:
        raise ValueError(
            "Gate A v8 default path forbids legacy mechanisms: " + ", ".join(enabled)
        )


def derive_seed(base_seed: int, *coordinates: int) -> int:
    entropy = [int(base_seed), *[int(value) for value in coordinates]]
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def paired_rollout_seeds(
    base_seed: int, policy_seed: int, scene_id: int, count: int
) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    return [derive_seed(base_seed, policy_seed, scene_id, 80, index) for index in range(count)]


def _validator_reason(
    validator: Callable[[np.ndarray], Any] | None, pose: np.ndarray
) -> str | None:
    if validator is None:
        return None
    result = validator(pose.copy())
    if result is None or result is True:
        return None
    if result is False:
        return "invalid_base_pose"
    if isinstance(result, str):
        return result or None
    if isinstance(result, Sequence) and len(result) == 2:
        accepted, reason = result
        return None if bool(accepted) else str(reason or "invalid_base_pose")
    raise TypeError("pose validator must return None/True, False, reason, or (accepted, reason)")


def sample_reachability_poses(
    reference_pose: Sequence[float],
    target_xy: Sequence[float],
    *,
    num_poses: int = 48,
    seed: int = 20260915,
    square_side_m: float = 1.0,
    target_radius_m: float = 1.0,
    yaw_jitter_deg: float = 30.0,
    max_attempts: int = 5000,
    pose_validator: Callable[[np.ndarray], Any] | None = None,
) -> tuple[list[ReachabilityPose], list[dict[str, Any]]]:
    """Sample the square/circle intersection and log every attempt.

    The proposal distribution is uniform in the 1 m square around the
    reference.  Circle membership and only clear base reset/collision failures
    are rejection conditions.  No arm, image, geometry-score, or policy signal
    is consulted.
    """
    reference = np.asarray(reference_pose, dtype=float)
    target = np.asarray(target_xy, dtype=float)
    if reference.shape != (3,) or target.shape != (2,):
        raise ValueError("reference_pose must be SE(2) and target_xy must be length 2")
    if not np.isfinite(reference).all() or not np.isfinite(target).all():
        raise ValueError("reference and target must be finite")
    if num_poses <= 0 or max_attempts < num_poses:
        raise ValueError("invalid pose/attempt count")
    if square_side_m <= 0 or target_radius_m <= 0 or yaw_jitter_deg <= 0:
        raise ValueError("sampling extents must be positive")

    rng = np.random.default_rng(int(seed))
    half_side = float(square_side_m) / 2.0
    yaw_limit = math.radians(float(yaw_jitter_deg))
    accepted: list[ReachabilityPose] = []
    attempts: list[dict[str, Any]] = []

    for attempt_index in range(int(max_attempts)):
        pose = np.asarray(
            [
                rng.uniform(reference[0] - half_side, reference[0] + half_side),
                rng.uniform(reference[1] - half_side, reference[1] + half_side),
                wrap_angle(reference[2] + rng.uniform(-yaw_limit, yaw_limit)),
            ],
            dtype=float,
        )
        radius = float(np.linalg.norm(pose[:2] - target))
        reason = None
        if radius > float(target_radius_m) + 1e-12:
            reason = "outside_target_radius"
        else:
            try:
                reason = _validator_reason(pose_validator, pose)
            except Exception as exc:
                reason = f"spawn_or_validation_error:{type(exc).__name__}:{exc}"

        accepted_pose_id = ""
        if reason is None:
            accepted_pose_id = f"reach_{len(accepted):03d}"
            accepted.append(
                ReachabilityPose(
                    pose_id=accepted_pose_id,
                    sample_index=len(accepted),
                    x=float(pose[0]),
                    y=float(pose[1]),
                    yaw=float(pose[2]),
                    sampling_seed=int(seed),
                )
            )
        attempts.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "sampling_seed": int(seed),
                "attempt_index": attempt_index,
                "requested_x": float(pose[0]),
                "requested_y": float(pose[1]),
                "requested_yaw": float(pose[2]),
                "target_distance_m": radius,
                "accepted": reason is None,
                "accepted_pose_id": accepted_pose_id,
                "rejection_reason": "" if reason is None else reason,
            }
        )
        if len(accepted) == num_poses:
            break

    if len(accepted) != num_poses:
        reasons: dict[str, int] = {}
        for row in attempts:
            if row["rejection_reason"]:
                reasons[row["rejection_reason"]] = reasons.get(row["rejection_reason"], 0) + 1
        raise RuntimeError(
            f"sampled only {len(accepted)}/{num_poses} poses after {max_attempts} "
            f"attempts; rejection counts={reasons}"
        )
    return accepted, attempts


def evaluate_pose_success_rate(
    *,
    task_name: str,
    scene_id: str | int,
    policy_name: str,
    policy_checkpoint: str,
    policy_version: str,
    policy_seed: int,
    pose_id: str,
    pose_type: str,
    pose: Sequence[float],
    rollout_seeds: Sequence[int],
    rollout_runner: Callable[[np.ndarray, int, str, str], bool | Mapping[str, Any]],
    sampling_seed: int | None = None,
    is_collision_free: bool = True,
) -> PoseEvaluation:
    """Evaluate any pose through the one shared Gate A rollout function."""
    value = np.asarray(pose, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("pose must be a finite SE(2) vector")
    if pose_type not in (POSE_TYPE_REACH, POSE_TYPE_REFERENCE):
        raise ValueError(f"unsupported pose_type={pose_type!r}")
    if not rollout_seeds:
        raise ValueError("rollout_seeds cannot be empty")

    records: list[dict[str, Any]] = []
    for rollout_index, rollout_seed in enumerate(rollout_seeds):
        result = rollout_runner(value.copy(), int(rollout_seed), pose_id, pose_type)
        if isinstance(result, Mapping):
            if "success" not in result:
                raise RuntimeError("rollout result mapping has no success field")
            record = dict(result)
            success = bool(record["success"])
        else:
            success = bool(result)
            record = {"success": success}
        record.update(
            {
                "protocol_version": PROTOCOL_VERSION,
                "task_name": task_name,
                "scene_id": str(scene_id),
                "policy_name": policy_name,
                "policy_checkpoint": policy_checkpoint,
                "policy_version": policy_version,
                "policy_seed": int(policy_seed),
                "pose_id": pose_id,
                "pose_type": pose_type,
                "x": float(value[0]),
                "y": float(value[1]),
                "yaw": float(value[2]),
                "rollout_index": rollout_index,
                "rollout_seed": int(rollout_seed),
                "success": success,
            }
        )
        records.append(record)

    success_count = sum(bool(row["success"]) for row in records)
    return PoseEvaluation(
        protocol_version=PROTOCOL_VERSION,
        task_name=task_name,
        scene_id=str(scene_id),
        policy_name=policy_name,
        policy_checkpoint=policy_checkpoint,
        policy_version=policy_version,
        policy_seed=int(policy_seed),
        pose_id=pose_id,
        pose_type=pose_type,
        x=float(value[0]),
        y=float(value[1]),
        yaw=float(value[2]),
        num_rollouts=len(records),
        success_count=int(success_count),
        success_rate=float(success_count / len(records)),
        rollout_seeds=[int(seed_value) for seed_value in rollout_seeds],
        sampling_seed=None if sampling_seed is None else int(sampling_seed),
        is_collision_free=bool(is_collision_free),
        rollout_records=records,
    )


def evaluate_reachability_and_reference(
    *,
    sampled_poses: Sequence[ReachabilityPose],
    reference_pose: Sequence[float],
    rollout_seeds: Sequence[int],
    evaluator: Callable[..., PoseEvaluation] = evaluate_pose_success_rate,
    evaluator_kwargs: Mapping[str, Any],
    rollout_runner: Callable[[np.ndarray, int, str, str], bool | Mapping[str, Any]],
) -> tuple[list[PoseEvaluation], PoseEvaluation]:
    """Run samples and reference through exactly the same evaluator callable."""
    sampled_results = [
        evaluator(
            **evaluator_kwargs,
            pose_id=item.pose_id,
            pose_type=POSE_TYPE_REACH,
            pose=item.array,
            rollout_seeds=rollout_seeds,
            rollout_runner=rollout_runner,
            sampling_seed=item.sampling_seed,
        )
        for item in sampled_poses
    ]
    reference_result = evaluator(
        **evaluator_kwargs,
        pose_id="reference",
        pose_type=POSE_TYPE_REFERENCE,
        pose=reference_pose,
        rollout_seeds=rollout_seeds,
        rollout_runner=rollout_runner,
        sampling_seed=None,
    )
    return sampled_results, reference_result


def summarize_gate_a(
    sampled_results: Sequence[PoseEvaluation],
    reference_result: PoseEvaluation,
    *,
    basin_analysis_enabled: bool = False,
    basin_success_threshold: float | None = None,
) -> GateASummary:
    """Compute SR_reach, SR_ref, Delta_reach and descriptive pose variation."""
    if not sampled_results:
        raise ValueError("sampled_results cannot be empty")
    if any(row.pose_type != POSE_TYPE_REACH for row in sampled_results):
        raise ValueError("SR_reach accepts reachability samples only")
    if reference_result.pose_type != POSE_TYPE_REFERENCE:
        raise ValueError("reference_result must have pose_type=reference")
    first = sampled_results[0]
    identity = (
        first.task_name,
        first.scene_id,
        first.policy_name,
        first.policy_checkpoint,
        first.policy_version,
        first.policy_seed,
    )
    for row in [*sampled_results, reference_result]:
        current = (
            row.task_name,
            row.scene_id,
            row.policy_name,
            row.policy_checkpoint,
            row.policy_version,
            row.policy_seed,
        )
        if current != identity:
            raise ValueError("all evaluations must share task/scene/policy identity")
    expected_rollouts = first.num_rollouts
    expected_seeds = first.rollout_seeds
    for row in [*sampled_results, reference_result]:
        if row.num_rollouts != expected_rollouts or row.rollout_seeds != expected_seeds:
            raise ValueError(
                "all sampled poses and reference must use the same K rollout seeds"
            )

    rates = np.asarray([row.success_rate for row in sampled_results], dtype=float)
    sr_reach = float(np.mean(rates))
    sr_ref = float(reference_result.success_rate)
    threshold = basin_success_threshold if basin_analysis_enabled else None
    if basin_analysis_enabled and (threshold is None or not 0.0 <= threshold <= 1.0):
        raise ValueError("enabled basin analysis requires a threshold in [0, 1]")
    basin_count = int(np.sum(rates >= float(threshold))) if threshold is not None else None

    return GateASummary(
        protocol_version=PROTOCOL_VERSION,
        protocol_name=PROTOCOL_NAME,
        task_name=first.task_name,
        scene_id=first.scene_id,
        policy_name=first.policy_name,
        policy_checkpoint=first.policy_checkpoint,
        policy_version=first.policy_version,
        policy_seed=first.policy_seed,
        num_sampled_poses=len(sampled_results),
        rollouts_per_pose=first.num_rollouts,
        reference_rollouts=reference_result.num_rollouts,
        sr_reach=sr_reach,
        sr_ref=sr_ref,
        delta_reach=float(sr_ref - sr_reach),
        sampled_pose_success_min=float(np.min(rates)),
        sampled_pose_success_max=float(np.max(rates)),
        sampled_pose_success_mean=sr_reach,
        sampled_pose_success_std=float(np.std(rates)),
        sampled_pose_success_median=float(np.median(rates)),
        sampled_pose_success_q25=float(np.quantile(rates, 0.25)),
        sampled_pose_success_q75=float(np.quantile(rates, 0.75)),
        sampled_pose_success_range=float(np.max(rates) - np.min(rates)),
        zero_success_pose_count=int(np.sum(rates == 0.0)),
        full_success_pose_count=int(np.sum(rates == 1.0)),
        basin_analysis_enabled=bool(basin_analysis_enabled),
        basin_success_threshold=threshold,
        basin_pose_count=basin_count,
        basin_pose_fraction=(
            None if basin_count is None else float(basin_count / len(sampled_results))
        ),
    )

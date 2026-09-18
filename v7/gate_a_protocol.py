#!/usr/bin/env python3
"""Pure Gate-A protocol utilities.

This module deliberately has no RoboCasa / MuJoCo dependency.  Candidate
generation, geometric selection, seed pairing, and
pose-registry validation can therefore be regression-tested on any machine.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


ANCHOR_LABELS = ("p_train", "p_mobipi")

PROTOCOL_VERSION = 7

MAIN_LABELS = (
    "p_geom",
    "p_train",
    "p_oracle",
)

DIAGNOSTIC_LABELS = ("p_mobipi",)
GEOMETRY_ABLATION_LABELS = ()

VALIDATION_LABELS = MAIN_LABELS
ACCEPTED_VALIDATION_LABELS = MAIN_LABELS + DIAGNOSTIC_LABELS


def wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def pose_key(pose: Sequence[float], decimals: int = 6) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(pose, dtype=float), decimals=decimals))


def target_frame_pose(target: Sequence[float], relative_pose: Sequence[float]) -> np.ndarray:
    """Convert [radius, bearing_offset, heading_offset] to world SE(2)."""
    radius, bearing_offset, heading_offset = map(float, relative_pose)
    target_heading = float(target[3]) if len(target) > 3 else 0.0
    bearing = target_heading + bearing_offset
    x = float(target[0]) + radius * math.cos(bearing)
    y = float(target[1]) + radius * math.sin(bearing)
    face_target = math.atan2(float(target[1]) - y, float(target[0]) - x)
    return np.asarray([x, y, wrap_angle(face_target + heading_offset)], dtype=float)


def pose_to_target_frame(target: Sequence[float], pose: Sequence[float]) -> list[float]:
    dx = float(pose[0]) - float(target[0])
    dy = float(pose[1]) - float(target[1])
    target_heading = float(target[3]) if len(target) > 3 else 0.0
    bearing = math.atan2(dy, dx)
    face_target = math.atan2(-dy, -dx)
    return [
        float(math.hypot(dx, dy)),
        wrap_angle(bearing - target_heading),
        wrap_angle(float(pose[2]) - face_target),
    ]


def load_pose_registry(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("pose registry must use schema_version=2")
    if not isinstance(payload.get("tasks"), dict):
        raise ValueError("pose registry must contain a tasks object")
    return payload


def _registry_entry(registry: Mapping, task: str, scene_id: int, policy_seed: int) -> Mapping:
    task_entry = registry.get("tasks", {}).get(task)
    if not isinstance(task_entry, Mapping):
        raise KeyError(f"pose registry has no task {task!r}")
    scene_entry = task_entry.get("scenes", {}).get(str(scene_id), {})
    seed_entry = scene_entry.get("policy_seeds", {}).get(str(policy_seed), {})
    merged = dict(task_entry.get("default", {}))
    merged.update(task_entry.get("policy_seeds", {}).get(str(policy_seed), {}))
    merged.update(scene_entry.get("default", {}))
    merged.update(seed_entry)
    return merged


def resolve_registered_pose(
    registry: Mapping,
    task: str,
    scene_id: int,
    policy_seed: int,
    target: Sequence[float],
    key: str,
) -> np.ndarray:
    entry = _registry_entry(
        registry, task, scene_id, policy_seed
    )
    relative = entry.get(key)

    if relative is None:
        raise KeyError(
            f"missing {key} for "
            f"{task}/scene_{scene_id}/seed_{policy_seed}"
        )

    if len(relative) != 3:
        raise ValueError(
            f"{key} must contain exactly three numbers"
        )

    value = np.asarray(relative, dtype=float)
    if (
        not np.isfinite(value).all()
        or float(value[0]) < 0
    ):
        raise ValueError(
            f"{key} must be finite with nonnegative radius"
        )

    return target_frame_pose(target, value)


def resolve_training_pose(
    registry: Mapping,
    task: str,
    scene_id: int,
    policy_seed: int,
    target: Sequence[float],
) -> np.ndarray:
    return resolve_registered_pose(
        registry,
        task,
        scene_id,
        policy_seed,
        target,
        "p_train_target_relative",
    )


def resolve_optional_mobipi_pose(
    registry: Mapping,
    task: str,
    scene_id: int,
    policy_seed: int,
    target: Sequence[float],
):
    try:
        return resolve_registered_pose(
            registry,
            task,
            scene_id,
            policy_seed,
            target,
            "p_mobipi_target_relative",
        )
    except KeyError:
        return None


def generate_candidate_pool(
    target: Sequence[float],
    anchors: Mapping[str, Sequence[float]],
    floor_bounds: Sequence[Sequence[float]],
    global_grid_step: float = 0.10,
    heading_offsets_deg: Sequence[float] = (-45, -30, -15, 0, 15, 30, 45),
    local_translation_scales: Sequence[float] = (0, 0.025, 0.05, 0.10, 0.20, 0.30),
    local_heading_offsets_deg: Sequence[float] = (-30, -20, -10, -5, 0, 5, 10, 20, 30),
) -> list[dict]:
    """Generate a room-wide pool plus task-frame perturbations around p_train.

    The global component is a cheap floor grid, not a fixed target-centered
    annulus.  Task feasibility is established later by collision, rendered
    visibility, and key-trajectory IK checks.
    """
    records: list[dict] = []
    seen: set[tuple[float, ...]] = set()

    def append(pose: Sequence[float], source: str, anchor_labels: Iterable[str] = ()) -> None:
        value = np.asarray(pose, dtype=float).copy()
        value[2] = wrap_angle(value[2])
        key = pose_key(value)
        labels = sorted(set(anchor_labels))
        if key in seen:
            for record in records:
                if pose_key(record["pose"]) == key:
                    record["anchor_labels"] = sorted(set(record["anchor_labels"]) | set(labels))
                    record["sources"] = sorted(set(record["sources"]) | {source})
                    return
        seen.add(key)
        relative = pose_to_target_frame(target, value)
        records.append(
            {
                "pool_id": len(records),
                "pose": value,
                "sources": [source],
                "anchor_labels": labels,
                "radius": relative[0],
                "bearing_offset": relative[1],
                "heading_offset": relative[2],
                "radial_bin": int(relative[0] / 0.10),
                "bearing_bin": int((relative[1] + math.pi) / math.radians(30.0)),
            }
        )

    for label in ANCHOR_LABELS:
        if label in anchors:
            append(anchors[label], f"anchor:{label}", [label])

    train_pose = np.asarray(anchors["p_train"], dtype=float)
    frame_yaw = float(target[3]) if len(target) > 3 else 0.0
    normal = np.asarray([math.cos(frame_yaw), math.sin(frame_yaw)], dtype=float)
    tangent = np.asarray([-normal[1], normal[0]], dtype=float)
    signed_offsets = sorted(
        {0.0, *[sign * float(scale) for scale in local_translation_scales for sign in (-1, 1)]}
    )
    for normal_offset in signed_offsets:
        for tangent_offset in signed_offsets:
            if math.hypot(normal_offset, tangent_offset) > max(local_translation_scales) + 1e-9:
                continue
            xy = train_pose[:2] + normal * normal_offset + tangent * tangent_offset
            for dtheta in local_heading_offsets_deg:
                append(
                    [xy[0], xy[1], train_pose[2] + math.radians(dtheta)],
                    "policy_local",
                )

    floor = np.asarray(floor_bounds, dtype=float)
    min_x, min_y = floor[:, :2].min(axis=0)
    max_x, max_y = floor[:, :2].max(axis=0)
    xs = np.arange(min_x, max_x + global_grid_step * 0.5, global_grid_step)
    ys = np.arange(min_y, max_y + global_grid_step * 0.5, global_grid_step)
    for x in xs:
        for y in ys:
            face_target = math.atan2(float(target[1]) - y, float(target[0]) - x)
            for heading_offset in heading_offsets_deg:
                append(
                    [x, y, face_target + math.radians(heading_offset)],
                    "room_global",
                )
    return records


def stratified_select(
    records: Sequence[Mapping], count: int, seed: int, include_anchor_rows: bool = True
) -> list[dict]:
    """Select ``count`` eligible poses plus any ineligible diagnostic anchors."""
    if count <= 0:
        raise ValueError("count must be positive")
    eligible = [dict(record) for record in records if bool(record.get("geometry_eligible"))]
    anchors = [dict(record) for record in records if record.get("anchor_labels")]
    selected: list[dict] = []
    used: set[int] = set()
    if include_anchor_rows:
        for record in anchors:
            pool_id = int(record["pool_id"])
            if pool_id not in used:
                selected.append(record)
                used.add(pool_id)

    eligible_selected = sum(bool(record.get("geometry_eligible")) for record in selected)

    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in eligible:
        if int(record["pool_id"]) not in used:
            groups[(int(record["radial_bin"]), int(record["bearing_bin"]))].append(record)
    rng = np.random.default_rng(seed)
    for values in groups.values():
        rng.shuffle(values)
    group_keys = sorted(groups)
    rng.shuffle(group_keys)
    while eligible_selected < count and group_keys:
        next_keys = []
        for key in group_keys:
            if groups[key] and eligible_selected < count:
                record = groups[key].pop()
                selected.append(record)
                used.add(int(record["pool_id"]))
                eligible_selected += 1
            if groups[key]:
                next_keys.append(key)
        group_keys = next_keys

    eligible_count = sum(bool(record.get("geometry_eligible")) for record in selected)
    if eligible_count < count:
        raise RuntimeError(
            f"only {len(eligible)} geometry-eligible poses after full-pool filtering; "
            f"need {count} plus diagnostic anchors"
        )
    for candidate_id, record in enumerate(selected):
        record["candidate_id"] = candidate_id
    return selected


def paired_seed(base_seed: int, policy_seed: int, scene_id: int, phase: int, rollout_id: int) -> int:
    """A comparison seed shared by every candidate in the same scene/phase."""
    # NumPy legacy RNG requires a uint32 seed.
    return int(np.random.SeedSequence(
        [int(base_seed), int(policy_seed), int(scene_id), int(phase), int(rollout_id)]
    ).generate_state(1, dtype=np.uint32)[0])


def minmax_score(rows: Sequence[Mapping], fields: Mapping[str, tuple[float, bool]]) -> dict[int, float]:
    """Compute a preregistered equal-weight geometry score.

    fields maps column -> (weight, higher_is_better). Missing/non-finite values
    contribute zero and are visible in the saved audit table. Normalization is
    performed only over the geometry-feasible set supplied by choose_geometry.
    """
    scores = {int(row["candidate_id"]): 0.0 for row in rows}
    total_weight = sum(abs(weight) for weight, _ in fields.values())
    if total_weight <= 0:
        raise ValueError("geometry score needs at least one positive weight")
    for field, (weight, higher_is_better) in fields.items():
        values = np.asarray([float(row.get(field, np.nan)) for row in rows], dtype=float)
        finite = np.isfinite(values)
        normalized = np.zeros(len(rows), dtype=float)
        if finite.any():
            lo, hi = float(values[finite].min()), float(values[finite].max())
            normalized[finite] = 0.5 if hi == lo else (values[finite] - lo) / (hi - lo)
            if not higher_is_better:
                normalized[finite] = 1.0 - normalized[finite]
        for row, value in zip(rows, normalized):
            scores[int(row["candidate_id"])] += abs(weight) * float(value) / total_weight
    return scores

def choose_geometry(rows, fields, tie_seed=0):
    """Select from geometry-feasible rows before any execution; ignore outcomes."""
    eligible = [dict(row) for row in rows if row.get("geometry_eligible")]
    if not eligible:
        raise RuntimeError("no geometry-feasible candidate; cannot compare headroom")
    scores = minmax_score(eligible, fields)
    best = max(scores.values())
    tied = sorted(
        [row for row in eligible if abs(scores[row["candidate_id"]] - best) <= 1e-12],
        key=lambda row: int(row["pool_id"]),
    )
    winner = tied[int(np.random.default_rng(tie_seed).integers(len(tied)))]
    return winner, scores

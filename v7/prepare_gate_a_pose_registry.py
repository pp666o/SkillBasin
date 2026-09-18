#!/usr/bin/env python3
"""Build an audited Gate-A training-pose registry from demonstrations.

Demonstration CSV columns:
  task,target_x,target_y,target_yaw,base_x,base_y,base_yaw[,policy_seed,source]
Optional Mobi-pi selector CSV adds scene_id and otherwise uses the same
columns. Its audited SE(2) medoid is stored per scene and policy seed as
p_mobipi_target_relative.
The training pose is stored in the target coordinate frame and is used only
as a candidate-generation prior plus a sanity-check evaluation pose.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from gate_a_protocol import pose_to_target_frame, wrap_angle


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def circular_medoid(values):
    values = np.asarray(values, dtype=float)
    distances = np.abs(
        (values[:, None] - values[None, :] + math.pi) % (2 * math.pi) - math.pi
    )
    return wrap_angle(float(values[int(np.argmin(distances.sum(axis=1)))]))


def robust_pose(rows, fields):
    return [
        circular_medoid([float(row[field]) for row in rows])
        if "yaw" in field or "bearing" in field or "heading" in field
        else float(np.median([float(row[field]) for row in rows]))
        for field in fields
    ]


def _relative_groups(rows, by_scene=False):
    grouped = {}
    for row in rows:
        target = [
            float(row["target_x"]),
            float(row["target_y"]),
            0.0,
            float(row["target_yaw"]),
        ]
        relative = pose_to_target_frame(
            target, [float(row["base_x"]), float(row["base_y"]), float(row["base_yaw"])]
        )
        key = (row["task"], str(row.get("policy_seed") or "default"))
        if by_scene:
            if row.get("scene_id") in (None, ""):
                raise ValueError("Mobi-pi selector rows require scene_id")
            key = (row["task"], str(row["scene_id"]), key[1])
        grouped.setdefault(key, []).append(
            {
                "radius": relative[0],
                "bearing": relative[1],
                "heading": relative[2],
                "source": row.get("source", ""),
            }
        )
    return grouped


def _se2_medoid(values):
    samples = np.asarray(
        [
            [
                row["radius"] * np.cos(row["bearing"]),
                row["radius"] * np.sin(row["bearing"]),
                wrap_angle(row["bearing"] + np.pi + row["heading"]),
            ]
            for row in values
        ]
    )
    translation = np.linalg.norm(
        samples[:, None, :2] - samples[None, :, :2], axis=-1
    )
    angles = np.abs(
        (samples[:, None, 2] - samples[None, :, 2] + np.pi) % (2 * np.pi) - np.pi
    )
    return values[int(np.argmin((translation + 0.2 * angles).sum(axis=1)))]


def build_registry(demo_rows, mobipi_rows=()):
    demo_by_task = _relative_groups(demo_rows)

    tasks = {}
    for (task, policy_seed), values in sorted(demo_by_task.items()):
        # Choose one actual SE(2) sample, rather than mixing coordinates across modes.
        representative = _se2_medoid(values)
        entry = {
            "p_train_target_relative": [
                representative[k]
                for k in ("radius", "bearing", "heading")
            ],
            "p_train_sample_count": len(values),
            "p_train_sources": sorted(
                {
                    r["source"]
                    for r in values
                    if r["source"]
                }
            ),
            "p_train_representative_source": representative["source"],
            "representative":
                "SE2 medoid; translation_m + 0.2 * yaw_rad",
        }
        if not representative["source"]:
            raise ValueError(
                f"representative demonstration source required "
                f"for {task}/{policy_seed}"
            )
        if not entry["p_train_sources"]:
            raise ValueError(f"demonstration source required for {task}/{policy_seed}")
        task_entry = tasks.setdefault(task, {})
        if policy_seed == "default":
            task_entry["default"] = entry
        else:
            task_entry.setdefault("policy_seeds", {})[policy_seed] = entry

    for (task, scene_id, policy_seed), values in sorted(
        _relative_groups(mobipi_rows, by_scene=True).items()
    ):
        representative = _se2_medoid(values)
        if not representative["source"]:
            raise ValueError(f"Mobi-pi selector source required for {task}/{policy_seed}")
        task_entry = tasks.setdefault(task, {})
        scene_entry = task_entry.setdefault("scenes", {}).setdefault(scene_id, {})
        if policy_seed == "default":
            entry = scene_entry.setdefault("default", {})
        else:
            entry = scene_entry.setdefault("policy_seeds", {}).setdefault(policy_seed, {})
        entry.update(
            {
                "p_mobipi_target_relative": [
                    representative[key] for key in ("radius", "bearing", "heading")
                ],
                "p_mobipi_sample_count": len(values),
                "p_mobipi_sources": sorted(
                    {row["source"] for row in values if row["source"]}
                ),
                "p_mobipi_representative_source": representative["source"],
            }
        )
    return {
        "schema_version": 2,
        "coordinate_contract": {
            "p_train_target_relative": "[radius_m, bearing_offset_rad, heading_offset_from_face_target_rad]",
            "p_mobipi_target_relative": "[radius_m, bearing_offset_rad, heading_offset_from_face_target_rad]",
        },
        "tasks": tasks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-csv", required=True, type=Path)
    parser.add_argument("--mobipi-csv", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    registry = build_registry(
        read_csv(args.demo_csv),
        read_csv(args.mobipi_csv) if args.mobipi_csv else (),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()

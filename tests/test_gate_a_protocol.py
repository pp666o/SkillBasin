import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_gate_a import basin_evidence, build_clusters, summarize
from gate_a_protocol import (
    ACCEPTED_VALIDATION_LABELS,
    MAIN_LABELS,
    PROTOCOL_VERSION,
    choose_geometry,
    VALIDATION_LABELS,
    generate_candidate_pool,
    paired_seed,
    resolve_training_pose,
    stratified_select,
)


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.target = [0.0, 0.0, 0.8, 0.0]
        self.anchors = {
            "p_train": np.array([0.4, 0.0, math.pi]),
        }
        self.floor_bounds = [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]]

    def test_gate_a_main_naming_is_minimal(self):
        self.assertEqual(MAIN_LABELS, ("p_geom", "p_train", "p_oracle"))
        self.assertIn("p_mobipi", ACCEPTED_VALIDATION_LABELS)
        for removed in ("p_default", "p_demo", "p_nav", "p_near", "p_IK", "p_star"):
            self.assertNotIn(removed, VALIDATION_LABELS)

    def test_pool_has_room_wide_and_multiscale_policy_local_coverage(self):
        pool = generate_candidate_pool(self.target, self.anchors, self.floor_bounds)
        global_radii = [row["radius"] for row in pool if "room_global" in row["sources"]]
        self.assertGreater(max(global_radii), 1.35)
        local = [row for row in pool if "policy_local" in row["sources"]]
        local_distances = {
            round(float(np.linalg.norm(row["pose"][:2] - self.anchors["p_train"][:2])), 3)
            for row in local
        }
        self.assertIn(0.025, local_distances)
        self.assertIn(0.3, local_distances)

    def test_optional_mobipi_anchor_is_kept(self):
        anchors = dict(self.anchors, p_mobipi=np.array([0.5, 0.1, math.pi]))
        pool = generate_candidate_pool(self.target, anchors, self.floor_bounds)
        labels = {label for row in pool for label in row["anchor_labels"]}
        self.assertEqual(labels, {"p_train", "p_mobipi"})

    def test_filter_then_select_keeps_anchors_and_requested_count(self):
        pool = generate_candidate_pool(self.target, self.anchors, self.floor_bounds)[:100]
        for index, row in enumerate(pool):
            row["geometry_eligible"] = index % 2 == 0
        selected = stratified_select(pool, 24, seed=7)
        self.assertEqual(sum(row["geometry_eligible"] for row in selected), 24)
        labels = {label for row in selected for label in row["anchor_labels"]}
        self.assertEqual(labels, {"p_train"})

    def test_missing_training_pose_is_fatal(self):
        registry = {
            "schema_version": 2,
            "tasks": {"Task": {"default": {}}},
        }
        with self.assertRaisesRegex(KeyError, "p_train_target_relative"):
            resolve_training_pose(registry, "Task", 0, 1, self.target)

    def test_comparison_seed_is_candidate_independent(self):
        self.assertEqual(paired_seed(3, 2, 4, 3, 9), paired_seed(3, 2, 4, 3, 9))
        self.assertNotEqual(paired_seed(3, 2, 4, 2, 9), paired_seed(3, 2, 4, 3, 9))


class AnalysisTest(unittest.TestCase):
    def synthetic_rows(self):
        rows = []
        for task in ("TaskA", "TaskB"):
            for policy_seed in (1, 2):
                for scene_id in (0, 1):
                    for comparison_seed in (10, 11, 12, 13):
                        for candidate_id, label in enumerate(VALIDATION_LABELS):
                            success = label == "p_oracle"
                            rows.append(
                                {
                                    "protocol_version": PROTOCOL_VERSION,
                                    "task": task,
                                    "policy_seed": str(policy_seed),
                                    "scene_id": str(scene_id),
                                    "candidate_id": str(candidate_id),
                                    "evaluation_labels": label,
                                    "comparison_seed": str(comparison_seed),
                                    "success": str(success),
                                    "experiment_split": "test",
                                    "failure_type": "success" if success else "policy_failure",
                                    "_source_file": "synthetic.csv",
                                }
                            )
        return rows

    def test_paired_hierarchy_and_preregistered_decision(self):
        rows = self.synthetic_rows()
        clusters = build_clusters(rows)
        self.assertEqual(len(clusters), 8)
        result = summarize(
            rows,
            [],
            bootstrap_samples=100,
            seed=1,
            split="test",
            minimum_effect=0.10,
            required_positive_tasks=2,
            minimum_oracle_success=0.20,
        )
        self.assertEqual(result["decision"]["status"], "pass")
        self.assertEqual(result["overall"]["H_geom"], 1.0)
        self.assertEqual(len(result["main_table"]), 2)
        self.assertEqual(result["audit"]["paired_seed_check"], "passed")

    def test_basin_evidence_reports_geometry_policy_mismatch(self):
        rows = [
            {
                "protocol_version": PROTOCOL_VERSION,
                "experiment_split": "test",
                "task": "TaskA",
                "policy_seed": 1,
                "scene_id": 0,
                "geometry_feasible": True,
                "success_rate": rate,
                "basin_member": rate >= 0.8,
            }
            for rate in (0.1, 0.4, 0.9)
        ]
        evidence = basin_evidence(rows, "test")
        self.assertEqual(evidence["clusters_with_mixed_basin_membership"], 1)
        self.assertTrue(evidence["observed_geometry_basin_mismatch"])
        self.assertEqual(
            evidence["per_scene"][0]["geometry_feasible_outside_basin"], 2
        )


if __name__ == "__main__":
    unittest.main()

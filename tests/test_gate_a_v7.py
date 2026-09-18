import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

from analyze_gate_a import build_clusters, effect_values
from gate_a_protocol import (
    choose_geometry,
    paired_seed,
    resolve_optional_mobipi_pose,
    resolve_training_pose,
)
from prepare_gate_a_pose_registry import build_registry
from gate_a_geometry import approach_ik, robot_collision, validate_contact_spec


class FakeSim:
    """Six independent coordinates; sufficient to exercise pose constraints/restoration."""
    def __init__(self, rotation_enabled=True, obstacle=False):
        self.rotation_enabled, self.obstacle = rotation_enabled, obstacle
        self.model = SimpleNamespace(
            site_id2name=lambda _: "eef", dof_jntid=np.arange(6),
            jnt_qposadr=np.arange(6), jnt_range=np.tile([-3., 3.], (6, 1)),
            jnt_limited=np.ones(6),
            geom_id2name=lambda i: ("robot0_link", "cabinet")[i],
        )
        self.data = SimpleNamespace(qpos=np.zeros(6), qvel=np.zeros(6),
            site_xpos=np.zeros((1, 3)), site_xmat=np.eye(3).reshape(1, 9),
            ncon=0, contact=[],
            get_site_jacp=lambda _: np.hstack([np.eye(3), np.zeros((3, 3))]),
            get_site_jacr=lambda _: np.hstack([np.zeros((3, 3)), np.eye(3) if rotation_enabled else np.zeros((3, 3))]))
        self.forward()

    def forward(self):
        self.data.site_xpos[0] = self.data.qpos[:3]
        self.data.site_xmat[0] = Rotation.from_rotvec(
            self.data.qpos[3:] if self.rotation_enabled else np.zeros(3)).as_matrix().ravel()
        hit = self.obstacle and .035 < self.data.qpos[0] < .085
        self.data.contact = [SimpleNamespace(dist=-.01, geom1=0, geom2=1)] if hit else []
        self.data.ncon = len(self.data.contact)


class V7Test(unittest.TestCase):
    def target(self):
        return dict(point=np.array([.15, 0., 0.]), normal=np.array([1., 0., 0.]),
                    eef_rotation=Rotation.from_euler("z", 45, degrees=True).as_matrix(),
                    approach_distance=.10, clearance=.015)

    def run_ik(self, sim):
        env = SimpleNamespace(sim=sim, robots=[SimpleNamespace(
            eef_site_id={"right": 0}, _ref_joint_vel_indexes=np.arange(6))])
        return approach_ik(env, self.target())

    def test_orientation_and_state_restoration(self):
        sim = FakeSim()
        result = self.run_ik(sim)
        self.assertTrue(result["trajectory_reachable"])
        self.assertIn("precontact_position_residual", result)
        self.assertLess(result["trajectory_orientation_residual"], np.deg2rad(10))
        np.testing.assert_array_equal(sim.data.qpos, np.zeros(6))
        np.testing.assert_array_equal(sim.data.qvel, np.zeros(6))

    def test_position_only_solution_is_rejected(self):
        result = self.run_ik(FakeSim(rotation_enabled=False))
        self.assertFalse(result["trajectory_reachable"])
        self.assertGreater(result["trajectory_orientation_residual"], .5)

    def test_collision_between_endpoints_is_rejected(self):
        self.assertFalse(self.run_ik(FakeSim(obstacle=True))["trajectory_reachable"])

    def test_floor_exemption_does_not_hide_arm_collision(self):
        sim = FakeSim()
        sim.model.geom_id2name = lambda i: ("robot0_arm", "floor")[i]
        sim.data.contact = [SimpleNamespace(dist=-.01, geom1=0, geom2=1)]
        sim.data.ncon = 1
        self.assertTrue(robot_collision(sim))
        sim.model.geom_id2name = lambda i: ("mobilebase0_wheel", "floor")[i]
        self.assertFalse(robot_collision(sim))

    def test_geometry_selection_ignores_rollout_and_excludes_invalid(self):
        rows = [
            dict(candidate_id=0, pool_id=0, geometry_eligible=True, quality=.1, success=1),
            dict(candidate_id=1, pool_id=1, geometry_eligible=True, quality=.8, success=0),
            dict(candidate_id=2, pool_id=2, geometry_eligible=False, quality=1., success=1)]
        winner, _ = choose_geometry(rows, {"quality": (1., True)})
        self.assertEqual(winner["candidate_id"], 1)
        for row in rows:
            row["success"] = 1 - row["success"]
        self.assertEqual(choose_geometry(rows, {"quality": (1., True)})[0]["candidate_id"], 1)

    def test_geometry_ties_are_reproducible_and_not_always_first(self):
        rows = [dict(candidate_id=i, pool_id=i, geometry_eligible=True, q=1.) for i in range(2)]
        self.assertEqual(choose_geometry(rows, {"q": (1., True)}, 8)[0],
                         choose_geometry(rows[::-1], {"q": (1., True)}, 8)[0])
        self.assertEqual({choose_geometry(rows, {"q": (1., True)}, i)[0]["candidate_id"]
                          for i in range(20)}, {0, 1})

    def test_headroom_is_signed_oracle_minus_geometry(self):
        values = effect_values({"p_oracle": .9, "p_train": .2, "p_geom": .8})
        self.assertAlmostEqual(values["H_geom"], .1)
        self.assertAlmostEqual(values["H_train"], .7)

    def test_shared_pose_yields_identical_paired_results(self):
        rows = [dict(protocol_version=7, experiment_split="test", task="T", policy_seed=1,
                     scene_id=0, candidate_id=0, comparison_seed=i, success=i % 2,
                     evaluation_labels="p_train+p_geom+p_oracle") for i in range(20)]
        cluster = next(iter(build_clusters(rows).values()))
        self.assertEqual(cluster["labels"]["p_train"], cluster["labels"]["p_geom"])
        rows[0]["failure_type"] = "rollout_error:ValueError"
        with self.assertRaisesRegex(RuntimeError, "rollout_error"):
            build_clusters(rows)
        rows[0]["failure_type"] = "policy_failure"
        rows[0]["protocol_version"] = 5
        with self.assertRaisesRegex(ValueError, "historical"):
            build_clusters(rows)

    def test_seeds_fit_numpy_and_are_distinct(self):
        seeds = [paired_seed(20260913, 1, 0, phase, i)
                 for phase in (1, 2, 3) for i in range(20)]
        self.assertEqual(len(set(seeds)), len(seeds))
        for seed in seeds:
            np.random.RandomState(seed)

    def test_contact_calibration_cannot_be_silently_assumed(self):
        with self.assertRaisesRegex(ValueError, "reviewed"):
            validate_contact_spec({"reviewed": False})

    def test_training_representative_preserves_policy_seed_and_actual_pose(self):
        rows = [dict(task="T", target_x=0, target_y=0, target_yaw=0,
                     base_x=x, base_y=0, base_yaw=0,
                     policy_seed=seed, source="train/demos.hdf5")
                for seed, x in [(1, .2), (1, .3), (1, .3), (2, 1.0)]]
        registry = build_registry(rows)
        np.testing.assert_allclose(resolve_training_pose(registry, "T", 0, 1, [0,0,0,0]),
                                   [.3, 0, 0], atol=1e-9)
        np.testing.assert_allclose(resolve_training_pose(registry, "T", 0, 2, [0,0,0,0]),
                                   [1., 0, 0], atol=1e-9)

    def test_registry_can_include_audited_mobipi_selector_pose(self):
        demo = [dict(task="T", target_x=0, target_y=0, target_yaw=0,
                     base_x=.3, base_y=0, base_yaw=0, policy_seed=1,
                     source="train/demos.hdf5")]
        selector = [dict(task="T", scene_id=0, target_x=0, target_y=0, target_yaw=0,
                         base_x=.5, base_y=.1, base_yaw=0, policy_seed=1,
                         source="mobipi/selector.json")]
        registry = build_registry(demo, selector)
        pose = resolve_optional_mobipi_pose(registry, "T", 0, 1, [0, 0, 0, 0])
        np.testing.assert_allclose(pose, [.5, .1, 0], atol=1e-9)
        self.assertIsNone(
            resolve_optional_mobipi_pose(registry, "T", 1, 1, [0, 0, 0, 0])
        )


if __name__ == "__main__":
    unittest.main()

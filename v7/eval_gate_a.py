#!/usr/bin/env python3
"""Gate A v7: geometry-feasibility versus policy-success-basin evaluation.

Candidate poses are first filtered and scored using geometry-only signals.
Policy rollouts over a sampled geometry-feasible set construct an empirical
success basin and select a policy-specific oracle pose. Final p_geom, p_train,
p_oracle, and (when registered) p_mobipi success rates are evaluated with
independent rollout seeds.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import importlib
import json
import math
import os
from pathlib import Path
import re

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from gate_a_geometry import resolve_contact, approach_ik, robot_collision, validate_contact_spec
from robomimic.envs.wrappers import EnvWrapper

from gate_a_protocol import (
    choose_geometry,
    PROTOCOL_VERSION,
    generate_candidate_pool,
    load_pose_registry,
    paired_seed,
    resolve_optional_mobipi_pose,
    resolve_training_pose,
    stratified_select,
    wrap_angle,
)


TASK_TO_DATASET = {
    "TurnOnStove": "TurnOnStove",
    "TurnOnSinkFaucet": "TurnOnSinkFaucet",
    "TurnOnMicrowave": "TurnOnMicrowave",
}

GEOMETRY_SCORE_FIELDS = {
    # Conventional geometry / kinematics only.
    "target_distance": (1.0, False),
    "pixel_visibility": (1.0, True),
    "trajectory_ik_margin": (1.0, True),
    "trajectory_max_residual": (1.0, False),
    "trajectory_orientation_residual": (1.0, False),
    "trajectory_contact_alignment": (1.0, True),
    "trajectory_manipulability": (1.0, True),
}


def parse_csv(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def configure_robocasa_assets(asset_root):
    """Point RoboCasa at local assets without modifying its checkout."""
    import robocasa
    import robocasa.models
    import robocasa.models.objects.kitchen_objects as kitchen_objects
    import robocasa.models.objects.kitchen_object_utils as object_utils

    asset_root = str(Path(asset_root).expanduser().resolve())
    robocasa.models.assets_root = asset_root
    object_utils.BASE_ASSET_ZOO_PATH = str(Path(asset_root) / "objects")
    raw_categories = importlib.reload(kitchen_objects).OBJ_CATEGORIES
    object_utils.OBJ_CATEGORIES.clear()
    for name, kwargs in raw_categories.items():
        common = deepcopy(kwargs)
        objaverse_kwargs = common.pop("objaverse", None)
        aigen_kwargs = common.pop("aigen", None)
        registries = {}
        if objaverse_kwargs is not None:
            objaverse_kwargs.update(common)
            registries["objaverse"] = object_utils.ObjCat(name=name, **objaverse_kwargs)
        if aigen_kwargs is not None:
            aigen_kwargs.update(common)
            registries["aigen"] = object_utils.ObjCat(
                name=name, aigen_cat=True, **aigen_kwargs
            )
        object_utils.OBJ_CATEGORIES[name] = registries


def _csv_value(value):
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def save_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(value) for key, value in row.items()} for row in rows
        )


def set_pose(raw_env, pose):
    from robocasa.utils.env_utils import set_robot_base_pose

    set_robot_base_pose(raw_env, np.asarray(pose, dtype=np.float64))
    for joint_name in (
        "mobilebase0_joint_mobile_forward",
        "mobilebase0_joint_mobile_side",
        "mobilebase0_joint_mobile_yaw",
    ):
        joint_id = raw_env.sim.model.joint_name2id(joint_name)
        raw_env.sim.data.qvel[raw_env.sim.model.jnt_dofadr[joint_id]] = 0.0
    raw_env.sim.forward()


def refresh_observation(env):
    if hasattr(env, "obs_history"):
        raw_obs = env.env.get_observation()
        env.update_obs(raw_obs, reset=True)
        env.obs_history = env._get_initial_obs_history(raw_obs)
        return env._get_stacked_obs_from_history()
    return env.get_observation()


def restore_snapshot(env, snapshot):
    if isinstance(snapshot, dict) and "states" in snapshot:
        env.reset_to({"states": snapshot["states"]})
    else:
        env.reset_to(snapshot)

    return refresh_observation(env)


def calibrated_base_collision(
    raw_env, penetration_tolerance=0.002, ignore_regex=r"floor|ground"
):
    """Ignore expected support-floor contact and report penetrations separately."""
    blocking, ignored = [], []
    pattern = re.compile(ignore_regex, flags=re.IGNORECASE) if ignore_regex else None
    for index in range(raw_env.sim.data.ncon):
        contact = raw_env.sim.data.contact[index]
        if float(contact.dist) >= -abs(float(penetration_tolerance)):
            continue
        name1 = raw_env.sim.model.geom_id2name(contact.geom1) or ""
        name2 = raw_env.sim.model.geom_id2name(contact.geom2) or ""
        if "mobilebase" not in name1 and "mobilebase" not in name2:
            continue
        pair = [name1, name2, float(contact.dist)]
        other = name2 if "mobilebase" in name1 else name1
        (ignored if pattern and pattern.search(other) else blocking).append(pair)
    return bool(blocking), blocking, ignored


def _camera_projection(raw_env, point, camera_name, width, height):
    cam_id = raw_env.sim.model.camera_name2id(camera_name)
    cam_pos = np.asarray(raw_env.sim.data.cam_xpos[cam_id], dtype=float)
    cam_mat = np.asarray(raw_env.sim.data.cam_xmat[cam_id], dtype=float).reshape(3, 3)
    local = cam_mat.T @ (np.asarray(point, dtype=float) - cam_pos)
    if local[2] >= 0:
        return None
    depth = -float(local[2])
    fovy = math.radians(float(raw_env.sim.model.cam_fovy[cam_id]))
    fy = 0.5 * height / math.tan(fovy / 2.0)
    fx = fy * width / height
    u = width / 2.0 + fx * float(local[0]) / depth
    v = height / 2.0 - fy * float(local[1]) / depth
    margin = min(u, width - 1 - u, v, height - 1 - v) / max(width, height)
    return u, v, depth, float(margin)


def _metric_depth(raw_env, depth_buffer):
    near = float(raw_env.sim.model.vis.map.znear * raw_env.sim.model.stat.extent)
    far = float(raw_env.sim.model.vis.map.zfar * raw_env.sim.model.stat.extent)
    return near * far / np.maximum(far - depth_buffer * (far - near), 1e-9)


def pixel_visibility(
    raw_env, point, camera_names, width=128, height=128, patch_radius=4
):
    """Depth-test a patch around the projected interaction point."""
    best = {
        "visible": False,
        "pixel_visibility": 0.0,
        "frustum_margin": -1.0,
        "visible_cameras": [],
    }
    for camera_name in camera_names:
        try:
            projection = _camera_projection(raw_env, point, camera_name, width, height)
            if projection is None:
                continue
            u, v, target_depth, margin = projection
            if not (0 <= u < width and 0 <= v < height):
                continue
            rendered = raw_env.sim.render(
                width=width, height=height, camera_name=camera_name, depth=True
            )
            depth_buffer = rendered[1] if isinstance(rendered, tuple) else rendered
            metric = _metric_depth(raw_env, np.asarray(depth_buffer, dtype=float)[::-1])
            ui, vi = int(round(u)), int(round(v))
            x0, x1 = max(0, ui - patch_radius), min(width, ui + patch_radius + 1)
            y0, y1 = max(0, vi - patch_radius), min(height, vi + patch_radius + 1)
            patch = metric[y0:y1, x0:x1]
            ratio = float(np.mean(patch >= target_depth - 0.03)) if patch.size else 0.0
            if ratio > best["pixel_visibility"]:
                best.update(
                    visible=ratio > 0,
                    pixel_visibility=ratio,
                    frustum_margin=margin,
                    visible_cameras=[camera_name],
                )
        except Exception:
            continue
    return best


def coarse_geometry_ok(pose, base_bounds, floor_bounds, robot_size):
    from shapely.geometry import Point, Polygon

    footprint = Point(float(pose[0]), float(pose[1])).buffer(float(robot_size))
    return Polygon(floor_bounds).contains(footprint) and not any(
        Polygon(bounds).intersects(footprint) for bounds in base_bounds
    )


class LockedBaseWrapper(EnvWrapper):
    def __init__(self, env, pose, episode_lang):
        super().__init__(env)
        self.pose = np.asarray(pose, dtype=float)
        self._ep_lang_str = episode_lang

    def step(self, action):
        action = np.asarray(action).copy()
        if action.ndim != 1 or action.shape[0] < 10:
            raise ValueError(
                f"expected a flat Mobi-pi action with at least 10 values, got {action.shape}"
            )
        # Mobi-pi / RoboCasa uses action[7:10] for the three mobile-base
        # commands. Index 10 belongs to the following controller part and must
        # remain under policy control.
        action[7:10] = 0.0
        obs, reward, done, info = self.env.step(action)
        set_pose(self.unwrapped.env, self.pose)
        return obs, reward, done, info


def load_policy_from_checkpoint(config, ckpt_path):
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.torch_utils as TorchUtils
    from robomimic.algo import RolloutPolicy, algo_factory
    from robomimic.utils.file_utils import maybe_dict_from_checkpoint
    import robomimic.utils.lang_utils as LangUtils

    ObsUtils.initialize_obs_utils_with_config(config)
    checkpoint = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
    shape_meta, env_meta = checkpoint["shape_metadata"], checkpoint["env_metadata"]
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    model.deserialize(checkpoint["model"])
    action_stats = deepcopy(checkpoint.get("action_normalization_stats"))
    if action_stats is not None:
        for stats in action_stats.values():
            for key, value in stats.items():
                if isinstance(value, list):
                    stats[key] = np.asarray(value)
    rollout_model = RolloutPolicy(
        model,
        obs_normalization_stats=checkpoint.get("obs_normalization_stats"),
        action_normalization_stats=action_stats,
        lang_encoder=LangUtils.LangEncoder(device=device),
    )
    return rollout_model, env_meta, shape_meta


def load_eval_env(config, env_meta, shape_meta, override):
    import torch
    import robomimic.utils.env_utils as EnvUtils

    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    metadata = deepcopy(env_meta)
    metadata["env_kwargs"].update(override)
    env = EnvUtils.create_env_from_metadata(
        env_meta=metadata,
        env_name=metadata["env_name"],
        render=False,
        render_offscreen=True,
        use_image_obs=shape_meta["use_images"],
    )
    return EnvUtils.wrap_env_from_config(env, config=config)


def run_rollout(
    env, raw_env, snapshot, pose, episode_lang, rollout_model, config, args, video_dir, seed
):
    import robomimic.utils.train_utils as TrainUtils

    restore_snapshot(env, snapshot)
    set_pose(raw_env, pose)
    refresh_observation(env)
    env._ep_lang_str = episode_lang
    locked_env = LockedBaseWrapper(env, pose, episode_lang)
    video_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
    try:
        logs, paths = TrainUtils.rollout_with_stats(
            policy=rollout_model,
            envs=[locked_env],
            horizon=[config.experiment.rollout.horizon if args.horizon is None else args.horizon],
            use_goals=config.use_goals,
            num_episodes=1,
            render=False,
            video_dir=str(video_dir),
            epoch=video_dir.name,
            video_skip=config.experiment.get("video_skip", 5),
            terminate_on_success=config.experiment.rollout.terminate_on_success,
            reset_before_rollout=False,
            del_envs_after_rollouts=False,
            data_logger=None,
        )
        rate = float(logs.get(locked_env.name, {}).get("Success_Rate", -1))
        if rate < 0:
            raise RuntimeError("rollout log has no Success_Rate")
        videos = [str(path) for path in (paths or {}).values() if path]
        success = rate >= 0.5
        return success, "success" if success else "policy_failure", "", videos
    except Exception as exc:
        raise RuntimeError(f"rollout execution failed at {video_dir}: {exc}") from exc


def append_rollout(
    sink, row, phase, rollout_id, seed, result, evaluation_labels=""
):
    success, failure_type, error, videos = result
    sink.append(
        {
            "protocol_version": PROTOCOL_VERSION,
            "task": row["task"],
            "scene_id": row["scene_id"],
            "policy_seed": row["policy_seed"],
            "experiment_split": row["experiment_split"],
            "candidate_id": row["candidate_id"],
            "pool_id": row["pool_id"],
            "evaluation_labels": evaluation_labels,
            "phase": phase,
            "rollout_id": rollout_id,
            "comparison_seed": seed,
            "x": row["x"],
            "y": row["y"],
            "theta": row["theta"],
            "success": bool(success),
            "failure_type": failure_type,
            "video_path": videos[-1] if videos else "",
            "error": error,
        }
    )


def save_geometry_success_plot(rows, path, success_threshold):
    """Plot all rollout-labelled geometry-feasible samples for one scene."""
    if not rows:
        return
    import matplotlib.pyplot as plt

    scores = np.asarray([float(row["geometry_score"]) for row in rows], dtype=float)
    rates = np.asarray([float(row["success_rate"]) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(scores, rates, c=rates, vmin=0.0, vmax=1.0, cmap="viridis", s=48)
    ax.axhline(success_threshold, color="tab:red", linestyle="--", linewidth=1.2,
               label=f"basin threshold = {success_threshold:.2f}")
    ax.set(xlabel="geometry score G(p)", ylabel="policy success Sπ(p)",
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def evaluate_task(
    args, task_name, policy_seed, config, rollout_model, env_meta, shape_meta, registry
):
    from mobipi.utils.env_utils import get_env_map_and_default_robot_init_pose

    task_root = Path(args.output_root) / task_name / f"policy_seed_{policy_seed}"
    task_root.mkdir(parents=True, exist_ok=True)
    camera_names = list(env_meta["env_kwargs"].get("camera_names", []))
    agent_cameras = [name for name in camera_names if "agentview" in name]
    if not agent_cameras:
        raise RuntimeError(
            "No agentview camera found. "
            f"Available cameras: {camera_names}"
        )
    all_pool, gt_basin_rows, all_selected, basin_rollout_rows, validation_rows, summaries = [], [], [], [], [], []

    for scene_id in args.scenes:
        override = {
            "layout_and_style_ids": [[scene_id, scene_id]],
            "camera_names": camera_names,
            "place_robot_for_nav": False,
            "seed": args.scene_seed_base + scene_id,
        }
        env = load_eval_env(config, env_meta, shape_meta, override)
        env.reset()
        raw_env = env.unwrapped.env
        episode_lang = raw_env.get_ep_meta().get("lang") or env_meta.get("env_lang") or ""
        env._ep_lang_str = episode_lang
        snapshot = env.get_state()
        base_bounds, floor_bounds, default_pos, default_ori, robot_size = (
            get_env_map_and_default_robot_init_pose(env=raw_env)
        )
        default_pose = np.asarray([default_pos[0], default_pos[1], default_ori[-1]])
        target = resolve_contact(raw_env, task_name, args.geometry_config, scene_id)
        target_frame = [*target["point"], target["frame_yaw"]]
        if registry is None:
            if not args.allow_default_training_pose_proxy:
                raise RuntimeError(
                    "--pose-registry is required unless the explicit default-pose proxy is enabled"
                )
            training_pose = default_pose.copy()
        else:
            training_pose = resolve_training_pose(
                registry, task_name, scene_id, policy_seed, target_frame
            )
        anchors = {"p_train": training_pose}
        mobipi_pose = None if registry is None else resolve_optional_mobipi_pose(
            registry, task_name, scene_id, policy_seed, target_frame
        )
        if mobipi_pose is not None:
            anchors["p_mobipi"] = mobipi_pose

        generated = generate_candidate_pool(
            target_frame,
            anchors,
            floor_bounds,
            global_grid_step=args.global_grid_step,
        )
        coarse = []
        for record in generated:
            if record["anchor_labels"] or coarse_geometry_ok(
                record["pose"], base_bounds, floor_bounds, robot_size
            ):
                value = dict(record)
                value["geometry_eligible"] = True
                coarse.append(value)
        prefilter = stratified_select(
            coarse,
            min(args.max_exact_filter_candidates, len(coarse)),
            args.protocol_seed + scene_id,
        )

        pool_rows = []
        for record in prefilter:
            base_footprint_feasible = coarse_geometry_ok(
                record["pose"], base_bounds, floor_bounds, robot_size
            )
            restore_snapshot(env, snapshot)
            set_pose(raw_env, record["pose"])
            collision, contacts, ignored = calibrated_base_collision(
                raw_env,
                penetration_tolerance=args.collision_penetration_tolerance,
                ignore_regex=args.collision_ignore_regex,
            )
            arm_contacts = robot_collision(raw_env.sim, args.collision_penetration_tolerance)
            collision = collision or bool(arm_contacts)
            contacts.extend(arm_contacts)
            visibility = pixel_visibility(raw_env, target["point"], agent_cameras)
            trajectory = approach_ik(
                raw_env, target, penetration_tolerance=args.collision_penetration_tolerance
            )
            pose = np.asarray(record["pose"], dtype=float)
            train_pose = np.asarray(anchors["p_train"], dtype=float)
            distance_to_train = float(
                np.linalg.norm(pose[:2] - train_pose[:2])
                + 0.20 * abs(wrap_angle(pose[2] - train_pose[2]))
            )
            visibility_margin = float(
                visibility["pixel_visibility"] - args.min_pixel_visibility
            )
            coarse_arm_position_residual = float(
                trajectory["precontact_position_residual"]
            )
            coarse_arm_reachable = bool(
                np.isfinite(coarse_arm_position_residual)
                and coarse_arm_position_residual
                <= args.max_coarse_ik_position_residual
            )
            coarse_reachability_margin = float(
                args.max_coarse_ik_position_residual
                - coarse_arm_position_residual
            )
            geometry_feasible = bool(
                base_footprint_feasible
                and (not collision)
                and visibility["pixel_visibility"]
                >= args.min_pixel_visibility
                and coarse_arm_reachable
            )
            pool_rows.append(
                {
                    **record,
                    "task": task_name,
                    "scene_id": scene_id,
                    "policy_seed": policy_seed,
                    "experiment_split": args.split,
                    "default_training_pose_proxy": registry is None,

                    "x": float(pose[0]),
                    "y": float(pose[1]),
                    "theta": float(pose[2]),
                    "target_x": float(target["point"][0]),
                    "target_y": float(target["point"][1]),
                    "target_z": float(target["point"][2]),
                    "target_distance": float(
                        np.linalg.norm(pose[:2] - target["point"][:2])
                    ),
                    "distance_to_train": distance_to_train,
                    "simulator_default_x": float(default_pose[0]),
                    "simulator_default_y": float(default_pose[1]),
                    "simulator_default_theta": float(default_pose[2]),
                    "collision": collision,
                    "base_footprint_feasible": base_footprint_feasible,
                    "blocking_contacts": contacts,
                    "ignored_contacts": ignored,
                    **visibility,
                    **trajectory,
                    "coarse_arm_reachable": coarse_arm_reachable,
                    "coarse_arm_position_residual": coarse_arm_position_residual,
                    "visibility_margin": visibility_margin,
                    "coarse_reachability_margin": coarse_reachability_margin,
                    "domain_margin": min(
                        visibility_margin,
                        coarse_reachability_margin,
                    ),

                    # Keep old name for protocol compatibility.
                    "geometry_eligible": geometry_feasible,
                    "geometry_feasible": geometry_feasible,
                }
            )
        # Freeze both choices before running either policy.
        selected = pool_rows
        scene_dir = task_root / f"scene_{scene_id}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        for row in selected:
            row["protocol_version"] = PROTOCOL_VERSION
        save_csv(
            pool_rows,
            scene_dir / "candidate_pool_preselection.csv",
        )
        geom_row, geometry_scores = choose_geometry(
            selected,
            GEOMETRY_SCORE_FIELDS,
            tie_seed=(
                args.protocol_seed
                + policy_seed * 100
                + scene_id
            ),
        )

        train_row = next(
            row
            for row in selected
            if "p_train" in row["anchor_labels"]
        )

        if train_row["collision"]:
            raise RuntimeError(
                "training pose collides; "
                "check demonstration coordinate mapping"
            )

        mobipi_row = next(
            (
                row
                for row in selected
                if "p_mobipi" in row["anchor_labels"]
            ),
            None,
        )
        if mobipi_row is not None and not mobipi_row["geometry_eligible"]:
            raise RuntimeError(
                "registered p_mobipi is not geometry feasible; audit the selector "
                "output or its target-frame mapping"
            )

        for row in selected:
            row["geometry_score"] = geometry_scores.get(
                row["candidate_id"],
                "",
            )

        save_csv(
            pool_rows,
            scene_dir / "candidate_pool.csv",
        )

        # ---------------------------------------------------------
        # GT basin discovery
        # ---------------------------------------------------------

        eligible_count = sum(
            bool(row["geometry_eligible"])
            for row in selected
        )

        basin_count = min(
            args.basin_candidates,
            eligible_count,
        )

        rows_by_pool_id = {
            int(row["pool_id"]): row
            for row in selected
        }

        if basin_count == 1:
            basin_candidates = [geom_row]
        else:
            # 先抽 basin_count - 1 个，再显式加入 p_geom
            sampled_basin_rows = stratified_select(
                [
                    row
                    for row in selected
                    if int(row["candidate_id"])
                    != int(geom_row["candidate_id"])
                ],
                basin_count - 1,
                seed=(
                    args.protocol_seed
                    + policy_seed * 10000
                    + scene_id * 100
                    + 17
                ),
                include_anchor_rows=False,
            )

            basin_candidates = [
                rows_by_pool_id[int(row["pool_id"])]
                for row in sampled_basin_rows
            ]

            basin_candidates.append(geom_row)

        basin_success_rates = {}
        scene_basin_rows = []

        for row in basin_candidates:
            candidate_id = int(row["candidate_id"])
            pose = np.asarray(
                [row["x"], row["y"], row["theta"]],
                dtype=float,
            )

            outcomes = []

            for rollout_id in range(args.basin_rollouts):
                seed = paired_seed(
                    args.protocol_seed,
                    policy_seed,
                    scene_id,
                    2,  # phase 2 = GT basin discovery
                    rollout_id,
                )

                video_dir = (
                    scene_dir
                    / f"basin_candidate_{candidate_id:03d}"
                    / f"rollout_{rollout_id:02d}"
                )

                result = run_rollout(
                    env,
                    raw_env,
                    snapshot,
                    pose,
                    episode_lang,
                    rollout_model,
                    config,
                    args,
                    video_dir,
                    seed,
                )

                outcomes.append(bool(result[0]))

                append_rollout(
                    basin_rollout_rows,
                    row,
                    "basin_discovery",
                    rollout_id,
                    seed,
                    result,
                )

            success_rate = float(np.mean(outcomes))

            basin_success_rates[candidate_id] = success_rate

            scene_basin_rows.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "task": task_name,
                    "scene_id": scene_id,
                    "policy_seed": policy_seed,
                    "experiment_split": args.split,

                    "candidate_id": candidate_id,
                    "pool_id": row["pool_id"],

                    "x": row["x"],
                    "y": row["y"],
                    "theta": row["theta"],

                    "geometry_feasible": bool(
                        row["geometry_eligible"]
                    ),
                    "geometry_score": row["geometry_score"],

                    "target_distance": row["target_distance"],
                    "pixel_visibility": row["pixel_visibility"],
                    "coarse_arm_reachable": row[
                        "coarse_arm_reachable"
                    ],
                    "trajectory_ik_margin": row[
                        "trajectory_ik_margin"
                    ],
                    "trajectory_max_residual": row[
                        "trajectory_max_residual"
                    ],
                    "trajectory_orientation_residual": row[
                        "trajectory_orientation_residual"
                    ],
                    "trajectory_manipulability": row[
                        "trajectory_manipulability"
                    ],
                    "trajectory_contact_alignment": row[
                        "trajectory_contact_alignment"
                    ],
                    "distance_to_train": row[
                        "distance_to_train"
                    ],

                    "success_rate": success_rate,
                    "basin_member": bool(
                        success_rate
                        >= args.basin_success_threshold
                    ),
                }
            )

        # Empirical oracle within the rollout-labelled candidate sample.
        best_oracle_rate = max(
            basin_success_rates.values()
        )

        oracle_row = min(
            (
                row
                for row in basin_candidates
                if abs(
                    basin_success_rates[int(row["candidate_id"])]
                    - best_oracle_rate
                ) <= 1e-12
            ),
            key=lambda row: int(row["candidate_id"]),
        )

        gt_basin_rows.extend(scene_basin_rows)

        save_csv(
            scene_basin_rows,
            scene_dir / "gt_basin.csv",
        )
        save_geometry_success_plot(
            scene_basin_rows,
            scene_dir / "geometry_vs_policy_success.png",
            args.basin_success_threshold,
        )

        label_to_row = {
            "p_geom": geom_row,
            "p_train": train_row,
            "p_oracle": oracle_row,
        }
        if mobipi_row is not None:
            label_to_row["p_mobipi"] = mobipi_row

        (scene_dir / "selection.json").write_text(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "selector_candidate_ids": {
                        k: v["candidate_id"]
                        for k, v in label_to_row.items()
                    },
                    "target": {
                        k: _csv_value(v)
                        for k, v in target.items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        labels_by_candidate = {}
        for label, row in label_to_row.items():
            labels_by_candidate.setdefault(row["candidate_id"], []).append(label)
        for row in selected:
            row["evaluation_labels"] = labels_by_candidate.get(row["candidate_id"], [])

        validation_rates = {}
        for candidate_id, labels in sorted(labels_by_candidate.items()):
            row = next(item for item in selected if item["candidate_id"] == candidate_id)
            pose = np.asarray([row["x"], row["y"], row["theta"]], dtype=float)
            outcomes = []
            for rollout_id in range(args.validation_rollouts):
                seed = paired_seed(args.protocol_seed, policy_seed, scene_id, 3, rollout_id)
                video_dir = scene_dir / f"validation_candidate_{candidate_id:03d}" / f"rollout_{rollout_id:02d}"
                result = run_rollout(
                    env, raw_env, snapshot, pose, episode_lang, rollout_model, config, args, video_dir, seed
                )
                outcomes.append(result[0])
                append_rollout(
                    validation_rows,
                    row,
                    "validation",
                    rollout_id,
                    seed,
                    result,
                    evaluation_labels="+".join(labels),
                )
            rate = float(np.mean(outcomes))
            for label in labels:
                validation_rates[label] = rate

        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "reference_kind": "demonstration_training_pose",
            "task": task_name,
            "scene_id": scene_id,
            "policy_seed": policy_seed,
            "experiment_split": args.split,
            "generated_pool": len(generated),
            "coarse_valid_pool": len(coarse),
            "exact_filtered_pool": len(pool_rows),
            "geometry_eligible_pool": int(sum(row["geometry_eligible"] for row in pool_rows)),
            "selected_candidates": len(selected),
            "selected_eligible_candidates": int(
                sum(row["geometry_eligible"] for row in selected)
            ),
            "selector_candidate_ids": {
                label: row["candidate_id"] for label, row in label_to_row.items()
            },
            "validation_success_rates": validation_rates,

            "geom_success_rate": validation_rates["p_geom"],
            "train_pose_success_rate": validation_rates["p_train"],
            "oracle_success_rate": validation_rates["p_oracle"],
            "mobipi_success_rate": validation_rates.get("p_mobipi"),

            "H_geom": (
                validation_rates["p_oracle"]
                - validation_rates["p_geom"]
            ),

            "H_train": (
                validation_rates["p_oracle"]
                - validation_rates["p_train"]
            ),

            "empirical_basin_candidates": len(scene_basin_rows),

            "empirical_basin_size": int(
                sum(
                    bool(row["basin_member"])
                    for row in scene_basin_rows
                )
            ),

            "empirical_basin_fraction": float(
                np.mean(
                    [
                        bool(row["basin_member"])
                        for row in scene_basin_rows
                    ]
                )
            ),

            "sampled_success_rate_min": float(
                min(row["success_rate"] for row in scene_basin_rows)
            ),
            "sampled_success_rate_max": float(
                max(row["success_rate"] for row in scene_basin_rows)
            ),
            "sampled_success_rate_range": float(
                max(row["success_rate"] for row in scene_basin_rows)
                - min(row["success_rate"] for row in scene_basin_rows)
            ),
            "sampled_success_rate_std": float(
                np.std([row["success_rate"] for row in scene_basin_rows])
            ),
            "geometry_feasible_outside_basin": int(
                sum(not row["basin_member"] for row in scene_basin_rows)
            ),
            "mixed_basin_membership": bool(
                0
                < sum(bool(row["basin_member"]) for row in scene_basin_rows)
                < len(scene_basin_rows)
            ),
            "oracle_scope": "sampled_geometry_feasible_candidates",

            "default_training_pose_proxy": registry is None,
        }
        summaries.append(summary)
        all_pool.extend(pool_rows)
        all_selected.extend(selected)
        save_csv(pool_rows, scene_dir / "candidate_pool.csv")
        save_csv(selected, scene_dir / "candidates.csv")
        (scene_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        close = getattr(env, "close", None)
        if callable(close):
            close()

    save_csv(
        all_pool,
        task_root / "candidate_pool_all.csv",
    )

    save_csv(
        all_selected,
        task_root / "candidates_all.csv",
    )

    save_csv(
        gt_basin_rows,
        task_root / "gt_basin_all.csv",
    )

    save_csv(
        basin_rollout_rows,
        task_root / "basin_rollouts.csv",
    )

    save_csv(
        validation_rows,
        task_root / "validation_rollouts.csv",
    )
    (task_root / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summaries


def ensure_checkpoint_alias(ckpt_root, task_name, seed):
    policy_root = Path(ckpt_root) / "robocasa" / "bc_xfmr"
    expected = f"*-{task_name}/seed_{seed}_*_mg-300/*/models/model_epoch_*.pth"
    if list(policy_root.glob(expected)):
        return
    for run_dir in policy_root.iterdir():
        if not run_dir.is_dir() or run_dir.name.endswith(f"-{task_name}"):
            continue
        pattern = f"seed_{seed}_*{task_name}_mg-300/*/models/model_epoch_*.pth"
        if list(run_dir.glob(pattern)):
            alias = policy_root / f"{run_dir.name}-{task_name}"
            if not alias.exists():
                alias.symlink_to(run_dir.name, target_is_directory=True)
            return
    raise FileNotFoundError(f"no seed {seed} checkpoint for {task_name} under {policy_root}")


def configure_language_cache(cache_root, use_clip, clip_cache_root=None):
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "huggingface")
    import torch
    import robomimic.utils.lang_utils as lang_utils

    if not use_clip:
        class WorkspaceLangEncoder:
            def __init__(self, device):
                self.device = device

            def get_lang_emb(self, lang):
                if isinstance(lang, (list, tuple)):
                    return torch.zeros((len(lang), 768), device=self.device)
                return None

        lang_utils.LangEncoder = WorkspaceLangEncoder
        return
    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    class WorkspaceLangEncoder:
        def __init__(self, device):
            self.device = device
            variant = "openai/clip-vit-large-patch14"
            cache = Path(clip_cache_root) if clip_cache_root else cache_root / "clip"
            snapshots = sorted(
                cache.glob("models--openai--clip-vit-large-patch14/snapshots/*/model.safetensors")
            )
            source = str(snapshots[-1].parent) if snapshots else variant
            local_only = source != variant
            self.model = CLIPTextModelWithProjection.from_pretrained(
                source, cache_dir=cache, local_files_only=local_only
            ).to(device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                source, cache_dir=cache, local_files_only=local_only
            )

        def get_lang_emb(self, lang):
            if lang is None:
                return None
            with torch.no_grad():
                tokens = self.tokenizer(
                    text=lang,
                    add_special_tokens=True,
                    padding="max_length",
                    return_attention_mask=True,
                    return_tensors="pt",
                ).to(self.device)
                embeddings = self.model(**tokens)["text_embeds"].detach()
            return embeddings[0] if isinstance(lang, str) else embeddings

    lang_utils.LangEncoder = WorkspaceLangEncoder


def require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Gate A refuses silent CPU fallback")
    print(f"Using CUDA device 0: {torch.cuda.get_device_name(0)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", default="assets/robocasa")
    parser.add_argument("--ckpt-root", default="artifacts/mobipi/ckpts")
    parser.add_argument("--data-root", default="artifacts/mobipi/data")
    parser.add_argument("--output-root", default="results/gate_a_v7")
    parser.add_argument("--pose-registry", default=None)
    parser.add_argument("--geometry-config", required=True, help="Reviewed contact frames, normals and EEF orientations")
    parser.add_argument("--allow-default-training-pose-proxy", action="store_true")
    parser.add_argument("--tasks", default="TurnOnStove,TurnOnSinkFaucet,TurnOnMicrowave")
    parser.add_argument("--scenes", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--policy-seeds", default="1,2,3")
    parser.add_argument("--split", choices=("calibration", "test"), default="test")
    parser.add_argument("--scene-seed-base", type=int, default=20260913)
    parser.add_argument("--protocol-seed", type=int, default=20260913)
    parser.add_argument("--max-exact-filter-candidates", type=int, default=2500)
    parser.add_argument("--global-grid-step", type=float, default=0.10)
    parser.add_argument("--min-pixel-visibility", type=float, default=0.50)
    parser.add_argument("--collision-penetration-tolerance", type=float, default=0.002)
    parser.add_argument("--collision-ignore-regex", default=r"floor|ground")
    parser.add_argument("--validation-rollouts", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--clip-cache-root", default=None)
    parser.add_argument(
        "--max-coarse-ik-position-residual",
        type=float,
        default=0.08,
        help=(
            "Relaxed positional-residual threshold at the far pre-contact "
            "waypoint. The residual comes from the weighted 6D IK diagnostic; "
            "strict orientation/path success is not required."
            "Strict orientation/path IK remains a score diagnostic."
        ),
    )

    parser.add_argument(
        "--basin-candidates",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--basin-rollouts",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--basin-success-threshold",
        type=float,
        default=0.80,
    )   
    args = parser.parse_args()
    for name in ("asset_root", "ckpt_root", "data_root", "output_root"):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    args.tasks = parse_csv(args.tasks)
    args.scenes = parse_csv(args.scenes, int)
    args.policy_seeds = parse_csv(args.policy_seeds, int)
    unknown = sorted(set(args.tasks) - set(TASK_TO_DATASET))
    if unknown:
        parser.error(f"unsupported tasks: {unknown}")
    if args.max_exact_filter_candidates < 2 or args.global_grid_step <= 0:
        parser.error("need at least two geometric candidates and a positive grid step")
    if not 1 <= args.validation_rollouts < 100:
        parser.error("validation rollouts must be in [1, 99]")
    if args.max_coarse_ik_position_residual <= 0:
        parser.error(
            "max coarse IK position residual must be positive"
        )

    if args.basin_candidates < 1:
        parser.error("basin candidates must be positive")

    if not 1 <= args.basin_rollouts < 100:
        parser.error("basin rollouts must be in [1, 99]")

    if not 0.0 < args.basin_success_threshold <= 1.0:
        parser.error(
            "basin success threshold must be in (0, 1]"
        )
    if args.protocol_seed < 0:
        parser.error("protocol seed must be nonnegative")
    if args.allow_default_training_pose_proxy and args.split == "test":
        parser.error("test requires an audited demonstration pose registry")
    registry = load_pose_registry(args.pose_registry) if args.pose_registry else None
    if registry is None and not args.allow_default_training_pose_proxy:
        parser.error("--pose-registry is required")
    args.geometry_config = json.loads(Path(args.geometry_config).read_text(encoding="utf-8"))
    if args.geometry_config.get("schema_version") != 1:
        parser.error("geometry config requires schema_version=1")
    for task in args.tasks:
        task_config = args.geometry_config.get("tasks", {}).get(task, {})
        for scene in args.scenes:
            spec = task_config.get("scenes", {}).get(str(scene), task_config.get("default"))
            if spec is None:
                parser.error(f"missing contact calibration for {task}/scene {scene}")
            validate_contact_spec(spec)
    output = Path(args.output_root)
    if output.exists() and any(output.rglob("validation_rollouts.csv")):
        parser.error("output contains earlier results; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(
        json.dumps(
            {"protocol_version": PROTOCOL_VERSION, **vars(args)}, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    require_cuda()
    configure_robocasa_assets(args.asset_root)
    from mobipi.utils.policy_utils import get_config_for_policy

    final = {}
    for task_name in args.tasks:
        final[task_name] = {}
        for policy_seed in args.policy_seeds:
            ensure_checkpoint_alias(args.ckpt_root, task_name, policy_seed)
            config, ckpt_path = get_config_for_policy(
                args.ckpt_root,
                args.data_root,
                task_name,
                "bc_xfmr",
                seed=policy_seed,
                dataset_name="mg-300",
            )
            rgb_config = config.observation.encoder.get("rgb", {})
            use_clip = bool(
                config.train.get("language_conditioned", False)
                or rgb_config.get("core_class", "") == "VisualCoreLanguageConditioned"
            )
            configure_language_cache(
                Path(args.output_root) / ".cache",
                use_clip=use_clip,
                clip_cache_root=args.clip_cache_root,
            )
            rollout_model, env_meta, shape_meta = load_policy_from_checkpoint(
                config, ckpt_path
            )
            final[task_name][str(policy_seed)] = evaluate_task(
                args,
                task_name,
                policy_seed,
                config,
                rollout_model,
                env_meta,
                shape_meta,
                registry,
            )
    (output / "summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

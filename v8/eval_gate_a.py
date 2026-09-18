#!/usr/bin/env python3
"""Gate A v8: N2M-style reachability-region policy evaluation.

For every task/scene/policy, this entry point samples N valid base poses from
the preregistered 1 m square / 1 m target-radius intersection, evaluates K
frozen-policy rollouts at every sampled pose, and evaluates the independent
RoboCasa reference pose with the same K seeds and the same evaluator path.

The default path has no geometry ranking, empirical best-pose selection,
waypoint IK, visibility score, or policy-informed candidate selection.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from gate_a_io import write_json, write_scene_artifacts
from gate_a_protocol import (
    DEFAULT_CONFIG,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    derive_seed,
    evaluate_reachability_and_reference,
    paired_rollout_seeds,
    sample_reachability_poses,
    summarize_gate_a,
    validate_default_config,
)


TASK_TO_DATASET = {
    "TurnOnStove": "TurnOnStove",
    "TurnOnSinkFaucet": "TurnOnSinkFaucet",
    "TurnOnMicrowave": "TurnOnMicrowave",
}

TASK_TO_TARGET_FIXTURE = {
    "TurnOnStove": "stove",
    "TurnOnSinkFaucet": "sink",
    "TurnOnMicrowave": "microwave",
}


def parse_csv(value: str, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def configure_robocasa_assets(asset_root: str | Path) -> None:
    """Point RoboCasa to local assets without modifying its checkout."""
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


def set_pose(raw_env: Any, pose: Sequence[float]) -> None:
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


def refresh_observation(env: Any) -> Any:
    if hasattr(env, "obs_history"):
        raw_obs = env.env.get_observation()
        env.update_obs(raw_obs, reset=True)
        env.obs_history = env._get_initial_obs_history(raw_obs)
        return env._get_stacked_obs_from_history()
    return env.get_observation()


def restore_snapshot(env: Any, snapshot: Any) -> Any:
    if isinstance(snapshot, dict) and "states" in snapshot:
        env.reset_to({"states": snapshot["states"]})
    else:
        env.reset_to(snapshot)
    return refresh_observation(env)


def calibrated_base_collision(
    raw_env: Any,
    penetration_tolerance: float = 0.002,
    ignore_regex: str = r"floor|ground",
) -> tuple[bool, list[list[Any]], list[list[Any]]]:
    """Reject only clear mobile-base penetration; ignore normal floor support."""
    blocking: list[list[Any]] = []
    ignored: list[list[Any]] = []
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


def coarse_base_pose_ok(
    pose: Sequence[float],
    base_bounds: Sequence[Sequence[Sequence[float]]],
    floor_bounds: Sequence[Sequence[float]],
    robot_radius: float,
) -> bool:
    """Cheap footprint/spawn legality test; it is not a ranking signal."""
    from shapely.geometry import Point, Polygon

    footprint = Point(float(pose[0]), float(pose[1])).buffer(float(robot_radius))
    return Polygon(floor_bounds).contains(footprint) and not any(
        Polygon(bounds).intersects(footprint) for bounds in base_bounds
    )


def task_target_xy(raw_env: Any, task_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Use the task fixture origin as the fixed N2M sampling target."""
    attribute = TASK_TO_TARGET_FIXTURE[task_name]
    fixture = getattr(raw_env, attribute)
    value = np.asarray(fixture.pos, dtype=float)
    if value.size < 2 or not np.isfinite(value[:2]).all():
        raise RuntimeError(f"invalid {attribute} fixture position for {task_name}")
    return value[:2].copy(), {
        "target_definition": "robocasa_task_fixture_origin_xy",
        "target_fixture_attribute": attribute,
        "target_xy": [float(value[0]), float(value[1])],
    }


def load_reference_registry(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 8 or not isinstance(payload.get("tasks"), dict):
        raise ValueError("reference registry must use schema_version=8 and contain tasks")
    return payload


def resolve_reference_pose(
    *,
    registry: Mapping[str, Any] | None,
    task_name: str,
    scene_id: int,
    policy_seed: int,
    environment_reference_pose: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve a rollout-independent RoboCasa reference pose and its provenance."""
    if registry is None:
        pose = np.asarray(environment_reference_pose, dtype=float)
        return pose, {
            "reference_pose_source": "robocasa_task_environment_initial_pose",
            "reference_registry": None,
            "reference_source_record": "raw_env._init_robot_pos/_init_robot_ori",
        }

    task = registry.get("tasks", {}).get(task_name, {})
    scene = task.get("scenes", {}).get(str(scene_id), {})
    entry = dict(task.get("default", {}))
    entry.update(task.get("policy_seeds", {}).get(str(policy_seed), {}))
    entry.update(scene.get("default", {}))
    entry.update(scene.get("policy_seeds", {}).get(str(policy_seed), {}))
    pose_value = entry.get("reference_pose_world")
    source = entry.get("source")
    if pose_value is None or not source:
        raise KeyError(
            f"missing audited reference_pose_world/source for "
            f"{task_name}/scene_{scene_id}/seed_{policy_seed}"
        )
    pose = np.asarray(pose_value, dtype=float)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError("reference_pose_world must be a finite [x, y, yaw]")
    return pose, {
        "reference_pose_source": "robocasa_demonstration_registry",
        "reference_registry": registry.get("registry_name", "unnamed"),
        "reference_source_record": source,
    }


class LockedBaseWrapper:
    """Thin proxy that zeros base actions while preserving the wrapped env API."""

    def __init__(self, env: Any, pose: Sequence[float], episode_lang: str):
        self.env = env
        self.pose = np.asarray(pose, dtype=float)
        self._ep_lang_str = episode_lang

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def step(self, action: Sequence[float]):
        value = np.asarray(action).copy()
        if value.ndim != 1 or value.shape[0] < 10:
            raise ValueError(f"expected a flat mobile policy action, got {value.shape}")
        value[7:10] = 0.0
        obs, reward, done, info = self.env.step(value)
        set_pose(self.unwrapped.env, self.pose)
        return obs, reward, done, info


def load_policy_from_checkpoint(config: Any, checkpoint_path: str):
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.torch_utils as TorchUtils
    from robomimic.algo import RolloutPolicy, algo_factory
    from robomimic.utils.file_utils import maybe_dict_from_checkpoint
    import robomimic.utils.lang_utils as LangUtils

    ObsUtils.initialize_obs_utils_with_config(config)
    checkpoint = maybe_dict_from_checkpoint(ckpt_path=checkpoint_path)
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


def load_eval_env(config: Any, env_meta: Mapping[str, Any], shape_meta: Mapping[str, Any], override: Mapping[str, Any]):
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
    *,
    env: Any,
    raw_env: Any,
    snapshot: Any,
    pose: Sequence[float],
    episode_lang: str,
    rollout_model: Any,
    config: Any,
    horizon: int | None,
    video_dir: Path,
    seed: int,
) -> dict[str, Any]:
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
            horizon=[config.experiment.rollout.horizon if horizon is None else horizon],
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
    except Exception as exc:
        raise RuntimeError(f"rollout execution failed at {video_dir}: {exc}") from exc
    task_logs = logs.get(locked_env.name, {})
    rate = float(task_logs.get("Success_Rate", -1))
    if rate < 0:
        raise RuntimeError("rollout log has no Success_Rate")
    videos = [str(path) for path in (paths or {}).values() if path]
    success = rate >= 0.5
    episode_length = task_logs.get("Horizon", "")
    return {
        "success": success,
        "termination_reason": "success" if success else "policy_failure",
        "episode_length": episode_length,
        "video_path": videos[-1] if videos else "",
        "error": "",
    }


def ensure_checkpoint_alias(checkpoint_root: str | Path, task_name: str, seed: int) -> None:
    policy_root = Path(checkpoint_root) / "robocasa" / "bc_xfmr"
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


def configure_language_cache(cache_root: str | Path, use_clip: bool, clip_cache_root: str | None = None) -> None:
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
            snapshots = sorted(cache.glob("models--openai--clip-vit-large-patch14/snapshots/*/model.safetensors"))
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


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Gate A refuses silent CPU fallback")
    print(f"Using CUDA device 0: {torch.cuda.get_device_name(0)}")


def evaluate_scene(
    *,
    args: argparse.Namespace,
    task_name: str,
    scene_id: int,
    policy_seed: int,
    config: Any,
    rollout_model: Any,
    env_meta: Mapping[str, Any],
    shape_meta: Mapping[str, Any],
    checkpoint_path: str,
    reference_registry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from mobipi.utils.env_utils import get_env_map_and_default_robot_init_pose

    camera_names = list(env_meta["env_kwargs"].get("camera_names", []))
    override = {
        "layout_and_style_ids": [[scene_id, scene_id]],
        "camera_names": camera_names,
        "place_robot_for_nav": False,
        "seed": args.scene_seed_base + scene_id,
    }
    env = load_eval_env(config, env_meta, shape_meta, override)
    try:
        env.reset()
        raw_env = env.unwrapped.env
        episode_lang = raw_env.get_ep_meta().get("lang") or env_meta.get("env_lang") or ""
        env._ep_lang_str = episode_lang
        snapshot = env.get_state()
        base_bounds, floor_bounds, default_pos, default_ori, robot_radius = (
            get_env_map_and_default_robot_init_pose(env=raw_env)
        )
        environment_reference = np.asarray(
            [default_pos[0], default_pos[1], default_ori[-1]], dtype=float
        )
        reference_pose, reference_meta = resolve_reference_pose(
            registry=reference_registry,
            task_name=task_name,
            scene_id=scene_id,
            policy_seed=policy_seed,
            environment_reference_pose=environment_reference,
        )
        target_xy, target_meta = task_target_xy(raw_env, task_name)

        def validate_base_pose(pose: np.ndarray):
            if not coarse_base_pose_ok(pose, base_bounds, floor_bounds, robot_radius):
                return False, "base_footprint_overlap_or_outside_floor"
            try:
                restore_snapshot(env, snapshot)
                set_pose(raw_env, pose)
            except Exception as exc:
                return False, f"illegal_reset_or_spawn:{type(exc).__name__}:{exc}"
            collision, _, _ = calibrated_base_collision(
                raw_env,
                penetration_tolerance=args.collision_penetration_tolerance,
                ignore_regex=args.collision_ignore_regex,
            )
            return (not collision), ("" if not collision else "base_collision")

        reference_valid, reference_reason = validate_base_pose(reference_pose)
        if not reference_valid:
            raise RuntimeError(f"reference pose is not a legal base spawn: {reference_reason}")

        sampling_seed = derive_seed(args.protocol_seed, policy_seed, scene_id, 81)
        sampled_poses, sampling_attempts = sample_reachability_poses(
            reference_pose,
            target_xy,
            num_poses=args.num_sampled_poses,
            seed=sampling_seed,
            square_side_m=args.square_side_m,
            target_radius_m=args.target_radius_m,
            yaw_jitter_deg=args.yaw_jitter_deg,
            max_attempts=args.max_sampling_attempts,
            pose_validator=validate_base_pose,
        )
        for attempt in sampling_attempts:
            attempt.update(
                task_name=task_name,
                scene_id=scene_id,
                policy_name="bc_xfmr",
                policy_seed=policy_seed,
            )
        rollout_seeds = paired_rollout_seeds(
            args.protocol_seed, policy_seed, scene_id, args.rollouts_per_pose
        )
        scene_dir = (
            Path(args.output_root)
            / task_name
            / f"policy_seed_{policy_seed}"
            / f"scene_{scene_id}"
        )

        def rollout_runner(pose: np.ndarray, seed: int, pose_id: str, pose_type: str):
            return run_rollout(
                env=env,
                raw_env=raw_env,
                snapshot=snapshot,
                pose=pose,
                episode_lang=episode_lang,
                rollout_model=rollout_model,
                config=config,
                horizon=args.horizon,
                video_dir=scene_dir / "videos" / pose_type / pose_id / f"rollout_{seed}",
                seed=seed,
            )

        evaluator_kwargs = {
            "task_name": task_name,
            "scene_id": scene_id,
            "policy_name": "bc_xfmr",
            "policy_checkpoint": str(Path(checkpoint_path).resolve()),
            "policy_version": args.policy_version,
            "policy_seed": policy_seed,
        }
        sampled_results, reference_result = evaluate_reachability_and_reference(
            sampled_poses=sampled_poses,
            reference_pose=reference_pose,
            rollout_seeds=rollout_seeds,
            evaluator_kwargs=evaluator_kwargs,
            rollout_runner=rollout_runner,
        )
        summary = summarize_gate_a(
            sampled_results,
            reference_result,
            basin_analysis_enabled=args.basin_analysis_enabled,
            basin_success_threshold=args.basin_success_threshold,
        )
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_name": PROTOCOL_NAME,
            "task_name": task_name,
            "scene_id": scene_id,
            "policy_name": "bc_xfmr",
            "policy_checkpoint": str(Path(checkpoint_path).resolve()),
            "policy_version": args.policy_version,
            "policy_git_commit": args.policy_git_commit,
            "policy_seed": policy_seed,
            "episode_language": episode_lang,
            "reference_pose": [float(value) for value in reference_pose],
            **reference_meta,
            **target_meta,
            "sampling_attempt_count": len(sampling_attempts),
            "sampling_rejection_count": sum(not row["accepted"] for row in sampling_attempts),
            "rollout_seed_contract": "same exact K seeds for every sampled pose and reference",
        }
        artifact_config = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_name": PROTOCOL_NAME,
            "num_sampled_poses": args.num_sampled_poses,
            "rollouts_per_pose": args.rollouts_per_pose,
            "square_side_m": args.square_side_m,
            "target_radius_m": args.target_radius_m,
            "yaw_jitter_deg": args.yaw_jitter_deg,
            "max_sampling_attempts": args.max_sampling_attempts,
            "protocol_seed": args.protocol_seed,
            "sampling_seed": sampling_seed,
            "basin_analysis_enabled": args.basin_analysis_enabled,
            "basin_success_threshold": args.basin_success_threshold,
            "legacy_geometry_enabled": False,
            "legacy_oracle_enabled": False,
            "legacy_waypoint_ik_enabled": False,
        }
        write_scene_artifacts(
            scene_dir=scene_dir,
            config=artifact_config,
            metadata=metadata,
            sampling_attempts=sampling_attempts,
            sampled_results=sampled_results,
            reference_result=reference_result,
            summary=summary,
            target_xy=target_xy,
        )
        return summary.to_dict()
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", default="assets/robocasa")
    parser.add_argument("--ckpt-root", default="artifacts/mobipi/ckpts")
    parser.add_argument("--data-root", default="artifacts/mobipi/data")
    parser.add_argument("--output-root", default="results/gate_a_v8")
    parser.add_argument("--reference-registry", default=None)
    parser.add_argument("--tasks", default=",".join(TASK_TO_DATASET))
    parser.add_argument("--scenes", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--policy-seeds", default="1,2,3")
    parser.add_argument("--policy-version", default="checkpoint-metadata-unavailable")
    parser.add_argument("--policy-git-commit", default="unknown")
    parser.add_argument("--scene-seed-base", type=int, default=20260915)
    parser.add_argument("--protocol-seed", type=int, default=DEFAULT_CONFIG["protocol_seed"])
    parser.add_argument("--num-sampled-poses", type=int, default=DEFAULT_CONFIG["num_sampled_poses"])
    parser.add_argument("--rollouts-per-pose", type=int, default=DEFAULT_CONFIG["rollouts_per_pose"])
    parser.add_argument("--square-side-m", type=float, default=DEFAULT_CONFIG["square_side_m"])
    parser.add_argument("--target-radius-m", type=float, default=DEFAULT_CONFIG["target_radius_m"])
    parser.add_argument("--yaw-jitter-deg", type=float, default=DEFAULT_CONFIG["yaw_jitter_deg"])
    parser.add_argument("--max-sampling-attempts", type=int, default=DEFAULT_CONFIG["max_sampling_attempts"])
    parser.add_argument("--collision-penetration-tolerance", type=float, default=0.002)
    parser.add_argument("--collision-ignore-regex", default=r"floor|ground")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--clip-cache-root", default=None)
    parser.add_argument("--basin-analysis-enabled", action="store_true")
    parser.add_argument("--basin-success-threshold", type=float, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    for name in ("asset_root", "ckpt_root", "data_root", "output_root"):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    args.tasks = parse_csv(args.tasks)
    args.scenes = parse_csv(args.scenes, int)
    args.policy_seeds = parse_csv(args.policy_seeds, int)
    unknown = sorted(set(args.tasks) - set(TASK_TO_DATASET))
    if unknown:
        parser.error(f"unsupported tasks: {unknown}")
    config_for_validation = {
        **DEFAULT_CONFIG,
        "num_sampled_poses": args.num_sampled_poses,
        "rollouts_per_pose": args.rollouts_per_pose,
        "square_side_m": args.square_side_m,
        "target_radius_m": args.target_radius_m,
        "yaw_jitter_deg": args.yaw_jitter_deg,
        "max_sampling_attempts": args.max_sampling_attempts,
        "protocol_seed": args.protocol_seed,
        "basin_analysis_enabled": args.basin_analysis_enabled,
        "basin_success_threshold": args.basin_success_threshold,
    }
    try:
        validate_default_config(config_for_validation)
    except ValueError as exc:
        parser.error(str(exc))
    if args.basin_analysis_enabled and args.basin_success_threshold is None:
        parser.error("--basin-analysis-enabled requires --basin-success-threshold")
    if args.basin_success_threshold is not None and not 0 <= args.basin_success_threshold <= 1:
        parser.error("basin success threshold must be in [0, 1]")
    if args.protocol_seed < 0:
        parser.error("protocol seed must be nonnegative")

    reference_registry = (
        load_reference_registry(args.reference_registry) if args.reference_registry else None
    )
    output = Path(args.output_root)
    if output.exists() and any(output.rglob("rollouts.csv")):
        parser.error("output contains earlier results; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json({**config_for_validation, **vars(args)}, output / "run_config.json")

    require_cuda()
    configure_robocasa_assets(args.asset_root)
    from mobipi.utils.policy_utils import get_config_for_policy

    final: dict[str, Any] = {}
    for task_name in args.tasks:
        final[task_name] = {}
        for policy_seed in args.policy_seeds:
            ensure_checkpoint_alias(args.ckpt_root, task_name, policy_seed)
            config, checkpoint_path = get_config_for_policy(
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
                output / ".cache", use_clip=use_clip, clip_cache_root=args.clip_cache_root
            )
            rollout_model, env_meta, shape_meta = load_policy_from_checkpoint(
                config, checkpoint_path
            )
            summaries = []
            for scene_id in args.scenes:
                summaries.append(
                    evaluate_scene(
                        args=args,
                        task_name=task_name,
                        scene_id=scene_id,
                        policy_seed=policy_seed,
                        config=config,
                        rollout_model=rollout_model,
                        env_meta=env_meta,
                        shape_meta=shape_meta,
                        checkpoint_path=checkpoint_path,
                        reference_registry=reference_registry,
                    )
                )
            final[task_name][str(policy_seed)] = summaries
    write_json(final, output / "summary.json")


if __name__ == "__main__":
    main()

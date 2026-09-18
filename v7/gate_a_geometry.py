"""Audited contact geometry and diagnostic pre-contact IK for Gate A v7.

The contact frame is calibrated on simulator assets, never on policy outcomes.
Strict 6D trajectory IK is retained as a geometry-quality diagnostic rather
than the definition of the geometry-feasible set.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def validate_contact_spec(spec):
    if spec.get("reviewed") is not True or not spec.get("source"):
        raise ValueError("contact geometry needs reviewed=true and a calibration source")
    if spec.get("frame_type") not in ("geom", "site"):
        raise ValueError("contact frame_type must be geom or site")
    for name in ("point_local", "normal_local"):
        value = np.asarray(spec[name], dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"invalid {name}")
    normal = np.asarray(spec["normal_local"], dtype=float)
    if np.linalg.norm(normal) < 1e-8:
        raise ValueError("contact normal cannot be zero")
    rotation = np.asarray(spec["eef_rotation_local"], dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError("eef_rotation_local must be a proper rotation matrix")
    if not 0.01 <= float(spec.get("approach_distance_m", 0.10)) <= 0.30:
        raise ValueError("approach distance must be in [0.01, 0.30] m")
    if not 0.005 <= float(spec.get("precontact_clearance_m", 0.015)) < float(spec.get("approach_distance_m", 0.10)):
        raise ValueError("precontact clearance must be positive and below approach distance")


def resolve_contact(raw_env, task, config, scene_id):
    task_config = config["tasks"][task]
    spec = task_config.get("scenes", {}).get(str(scene_id), task_config.get("default"))
    if spec is None:
        raise ValueError(f"missing calibrated contact for {task}/scene {scene_id}")
    validate_contact_spec(spec)
    fixture = getattr(raw_env, {
        "TurnOnStove": "stove", "TurnOnSinkFaucet": "sink",
        "TurnOnMicrowave": "microwave",
    }[task])
    frame_name = spec["frame_name"].format(
        prefix=fixture.naming_prefix, knob=getattr(raw_env, "knob", "")
    )
    kind = spec["frame_type"]
    model, data = raw_env.sim.model, raw_env.sim.data
    index = getattr(model, f"{kind}_name2id")(frame_name)
    origin = np.asarray(getattr(data, f"{kind}_xpos")[index], dtype=float)
    frame = np.asarray(getattr(data, f"{kind}_xmat")[index], dtype=float).reshape(3, 3)
    normal = frame @ np.asarray(spec["normal_local"], dtype=float)
    normal /= np.linalg.norm(normal)
    return {
        "point": origin + frame @ np.asarray(spec["point_local"], dtype=float),
        "normal": normal,
        "eef_rotation": frame @ np.asarray(spec["eef_rotation_local"], dtype=float),
        "frame_yaw": float(fixture.rot),
        "approach_distance": float(spec.get("approach_distance_m", 0.10)),
        "clearance": float(spec.get("precontact_clearance_m", 0.015)),
        "source": spec["source"], "frame_name": frame_name,
    }


def robot_collision(sim, tolerance=0.002):
    """Check active MuJoCo robot contacts, including self and arm/environment.

    Only mobile-base/floor support is exempt; arm/floor contacts still block.
    """
    blocked = []
    for index in range(sim.data.ncon):
        contact = sim.data.contact[index]
        if float(contact.dist) >= -tolerance:
            continue
        names = [sim.model.geom_id2name(i) or "" for i in (contact.geom1, contact.geom2)]
        robot = [any(token in n for token in ("robot0", "gripper0", "mobilebase0")) for n in names]
        if not any(robot):
            continue
        support = any(
            "mobilebase0" in names[i] and not robot[1-i]
            and any(token in names[1-i].lower() for token in ("floor", "ground"))
            for i in (0, 1)
        )
        if not support:
            blocked.append([*names, float(contact.dist)])
    return blocked


def pose_error(position, rotation, target_position, target_rotation):
    return np.concatenate([
        np.asarray(target_position) - position,
        Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec(),
    ])


def approach_ik(raw_env, target, max_iters=160, position_tolerance=0.01,
                angle_tolerance=np.deg2rad(10), penetration_tolerance=0.002):
    """Solve sequential 6D pre-contact poses and check interpolation collisions.

    A geometric approach template only: it does not certify successful actuation.
    Failed convergence is conservative rejection, not proof of unreachability.
    """
    sim, robot = raw_env.sim, raw_env.robots[0]
    site = robot.eef_site_id
    if isinstance(site, dict):
        site = site["right"] if "right" in site else next(iter(site.values()))
    elif isinstance(site, (list, tuple, np.ndarray)):
        site = site[-1]
    site = int(site)
    site_name = sim.model.site_id2name(site)
    dofs = np.asarray(robot._ref_joint_vel_indexes, dtype=int)
    joints = [int(sim.model.dof_jntid[d]) for d in dofs]
    qidx = np.asarray([sim.model.jnt_qposadr[j] for j in joints], dtype=int)
    ranges = np.asarray([sim.model.jnt_range[j] for j in joints], dtype=float)
    limited = np.asarray([sim.model.jnt_limited[j] for j in joints], dtype=bool)
    saved_pos, saved_vel = sim.data.qpos.copy(), sim.data.qvel.copy()
    results = []
    initial_contacts = robot_collision(sim, penetration_tolerance)
    path_ok = not initial_contacts
    normal, goal_rotation = target["normal"], target["eef_rotation"]
    distances = np.linspace(target["approach_distance"], target["clearance"], 4)
    try:
        for distance in distances:
            goal = target["point"] + distance * normal
            previous = sim.data.qpos[qidx].copy()
            q = previous.copy()
            for _ in range(max_iters):
                sim.forward()
                rotation = np.asarray(sim.data.site_xmat[site]).reshape(3, 3)
                error = pose_error(np.asarray(sim.data.site_xpos[site]), rotation, goal, goal_rotation)
                if np.linalg.norm(error[:3]) <= position_tolerance and np.linalg.norm(error[3:]) <= angle_tolerance:
                    break
                jac = np.vstack([sim.data.get_site_jacp(site_name)[:, dofs],
                                 sim.data.get_site_jacr(site_name)[:, dofs]])
                weights = np.diag([1, 1, 1, 0.2, 0.2, 0.2])
                weighted = weights @ jac
                delta = weighted.T @ np.linalg.solve(
                    weighted @ weighted.T + 0.02**2 * np.eye(6), weights @ error)
                q += np.clip(delta, -0.10, 0.10)
                q[limited] = np.clip(q[limited], ranges[limited, 0] + 1e-4, ranges[limited, 1] - 1e-4)
                sim.data.qpos[qidx] = q
            sim.forward()
            rotation = np.asarray(sim.data.site_xmat[site]).reshape(3, 3)
            error = pose_error(np.asarray(sim.data.site_xpos[site]), rotation, goal, goal_rotation)
            pos_res, ang_res = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
            jac = np.vstack([sim.data.get_site_jacp(site_name)[:, dofs],
                             sim.data.get_site_jacr(site_name)[:, dofs]])
            margins = np.minimum(
                (q[limited] - ranges[limited, 0]) / np.maximum(np.ptp(ranges[limited], axis=1), 1e-8),
                (ranges[limited, 1] - q[limited]) / np.maximum(np.ptp(ranges[limited], axis=1), 1e-8))
            contacts = []
            steps = max(2, int(np.ceil(np.max(np.abs(q - previous)) / 0.025)) + 1)
            for alpha in np.linspace(0, 1, steps):
                sim.data.qpos[qidx] = previous + alpha * (q - previous)
                sim.forward()
                contacts.extend(robot_collision(sim, penetration_tolerance))
                if contacts:
                    break
            sim.data.qpos[qidx] = q
            sim.forward()
            position_ok = pos_res <= position_tolerance
            orientation_ok = ang_res <= angle_tolerance
            collision_free = not contacts
            ok = position_ok and orientation_ok and collision_free

            path_ok = path_ok and ok

            results.append({
                "position": goal.tolist(),
                "position_residual": pos_res,
                "orientation_residual_rad": ang_res,
                "joint_margin": max(0.0, float(margins.min())) if margins.size else 0.0,
                "manipulability": float(np.prod(np.linalg.svd(jac, compute_uv=False))),
                "alignment": float(np.cos(ang_res)),
                "position_ok": bool(position_ok),
                "orientation_ok": bool(orientation_ok),
                "collision_free": bool(collision_free),
                "pass": bool(ok),
                "blocking_contacts": contacts,
            })

        # Do not break here. Full 4-waypoint IK is now a diagnostic / score,
        # not the definition of geometric reachability.
    finally:
        sim.data.qpos[:] = saved_pos
        sim.data.qvel[:] = saved_vel
        sim.forward()
    return {
        # Strict trajectory diagnostic. Do NOT use as the hard P_geo gate.
        "trajectory_reachable": bool(path_ok and len(results) == len(distances)),
        # The first waypoint is the far pre-contact waypoint and is retained
        # for auditability, not as the target-near reachability test.
        "approach_position_residual": float(results[0]["position_residual"]),
        "approach_orientation_residual": float(
            results[0]["orientation_residual_rad"]
        ),
        # The last waypoint is closest to the interaction point and therefore
        # supplies the relaxed R_arm diagnostic used by the hard gate.
        "precontact_position_residual": float(results[-1]["position_residual"]),
        # Continuous geometry-quality features.
        "trajectory_ik_margin": min(r["joint_margin"] for r in results),
        "trajectory_max_residual": max(r["position_residual"] for r in results),
        "trajectory_orientation_residual": max(
            r["orientation_residual_rad"] for r in results
        ),
        "trajectory_manipulability": min(r["manipulability"] for r in results),
        "trajectory_contact_alignment": min(r["alignment"] for r in results),
        "trajectory_waypoint_success_fraction": float(
            np.mean([r["pass"] for r in results])
        ),
        "trajectory_collision_free_fraction": float(
            np.mean([r["collision_free"] for r in results])
        ),
        "trajectory_waypoints": results,
        "initial_robot_contacts": initial_contacts,
    }

#!/usr/bin/env python3
"""Artifact writers and the three preregistered Gate A v8 figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gate_a_protocol import GateASummary, PoseEvaluation


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def make_scene_figures(
    sampled_results: Sequence[PoseEvaluation],
    reference_result: PoseEvaluation,
    target_xy: Sequence[float],
    figure_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    rates = np.asarray([row.success_rate for row in sampled_results], dtype=float)
    xs = np.asarray([row.x for row in sampled_results], dtype=float)
    ys = np.asarray([row.y for row in sampled_results], dtype=float)
    target = np.asarray(target_xy, dtype=float)

    map_path = figure_dir / "pose_success_map.png"
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    scatter = ax.scatter(xs, ys, c=rates, cmap="viridis", vmin=0, vmax=1, s=58)
    ax.scatter(
        [reference_result.x], [reference_result.y], marker="*", s=190,
        color="tab:red", edgecolor="black", linewidth=0.7, label="reference"
    )
    ax.scatter([target[0]], [target[1]], marker="X", s=95, color="black", label="target")
    ax.set(xlabel="base x (m)", ylabel="base y (m)", title="Sampled-pose success rates")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="success rate")
    fig.tight_layout()
    fig.savefig(map_path, dpi=180)
    plt.close(fig)

    distribution_path = figure_dir / "pose_success_distribution.png"
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    bins = np.linspace(-0.025, 1.025, 12)
    ax.hist(rates, bins=bins, color="tab:blue", alpha=0.78, edgecolor="white")
    ax.axvline(
        reference_result.success_rate, color="tab:red", linestyle="--", linewidth=1.6,
        label=f"reference = {reference_result.success_rate:.3f}"
    )
    ax.set(xlabel="per-pose success rate", ylabel="number of sampled poses",
           title="Reachability-region pose variation", xlim=(-0.03, 1.03))
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(distribution_path, dpi=180)
    plt.close(fig)

    comparison_path = figure_dir / "reach_vs_reference.png"
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    values = [float(np.mean(rates)), float(reference_result.success_rate)]
    bars = ax.bar(["SR_reach", "SR_ref"], values, color=["tab:blue", "tab:red"], width=0.62)
    ax.set(ylabel="success rate", ylim=(0, 1.05), title="Naive reachability vs reference")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)
    return [map_path, distribution_path, comparison_path]


def write_scene_artifacts(
    *,
    scene_dir: Path,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    sampling_attempts: Sequence[Mapping[str, Any]],
    sampled_results: Sequence[PoseEvaluation],
    reference_result: PoseEvaluation,
    summary: GateASummary,
    target_xy: Sequence[float],
) -> list[Path]:
    scene_dir.mkdir(parents=True, exist_ok=True)
    write_json(dict(config), scene_dir / "config.json")
    write_json(dict(metadata), scene_dir / "metadata.json")
    write_csv(list(sampling_attempts), scene_dir / "sampling_attempts.csv")
    evaluations = [*sampled_results, reference_result]
    write_csv([item.pose_row() for item in evaluations], scene_dir / "poses.csv")
    write_csv(
        [record for item in evaluations for record in item.rollout_records],
        scene_dir / "rollouts.csv",
    )
    write_json(summary.to_dict(), scene_dir / "summary.json")
    return make_scene_figures(
        sampled_results, reference_result, target_xy, scene_dir / "figures"
    )

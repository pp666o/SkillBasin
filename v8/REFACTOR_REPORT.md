# Gate A v8 重构交付报告

## 要求与实现逐项对应

| 审查要求 | v8 实现与产物 |
| --- | --- |
| Gate A 只研究 N2M 可达区域中的站位依赖 | `gate_a_protocol.py::sample_reachability_poses` 固定为目标 1 m 圆与 reference 周围 1 m 方形的交集 |
| 航向围绕 reference 采样 | `yaw = reference yaw + Uniform(-30°, +30°)`，角度归一化后保存 |
| 只过滤显然非法的底盘位姿 | `eval_gate_a.py::validate_base_pose` 仅检查 footprint/floor、reset/spawn 和 mobile-base penetration |
| 固定随机性并保留所有采样尝试 | 每个 scene 由 `derive_seed` 固定 sampling seed；`sampling_attempts.csv` 保存接受与拒绝记录及原因 |
| 不做几何评分、复杂 IK 或事后最优位姿选择 | v8 不导入父目录 `gate_a_geometry.py`；不存在几何 selector、关键轨迹 IK 或按成功率选点的执行分支 |
| N 个采样位姿，各 K 次 rollout | 默认 N=48、K=5；每个 `PoseEvaluation` 保存成功次数、成功率和全部 rollout seeds |
| Reference 独立于当前 rollout | 默认从 RoboCasa task/environment 固定初始位姿取得；也支持 schema v8 的审计 demonstration world-pose 注册表 |
| Reference 与采样位姿必须走同一路径 | `evaluate_reachability_and_reference` 对两类位姿调用同一个 evaluator，使用完全相同的 K 个 seeds |
| 主指标只保留 SR_reach、SR_ref、Delta_reach | `summarize_gate_a` 直接计算三项指标；reference 不进入 reachability 平均 |
| 报告区域内变化而不把 max 当 selector | 保存 min/max/mean/std/median/q25/q75/range、0/1 成功率位姿数；没有 max 位姿复评逻辑 |
| Basin threshold 降为可选诊断 | `basin_analysis_enabled=false`；只有显式开启并提供阈值时才计算比例 |
| 保存配置、元数据、逐位姿和逐 rollout 数据 | 每个 scene 输出 `config.json`、`metadata.json`、`poses.csv`、`sampling_attempts.csv`、`rollouts.csv`、`summary.json` |
| 固定三张核心图 | 输出 `pose_success_map.png`、`pose_success_distribution.png`、`reach_vs_reference.png` |
| 跨 scene/policy 汇总和置信区间 | `analyze_gate_a.py` 按 task/policy 汇总，层级 bootstrap 先抽 policy seed、再抽其 scenes，同时报告 `fraction_delta_reach_gt_zero` |
| 六项单元测试 | `tests/test_gate_a_v8.py` 覆盖均值、差值、reference 排除、采样复现、同一路径、legacy 禁用 |
| v7 可复现且 v8 默认不可进入旧路径 | 父目录 v7 文件保留；v8 是独立目录，三个 legacy flags 固定为 false，开启即报错 |

## 已执行验证

```text
python3 -m py_compile SkillBasin/v8/*.py               PASS
bash -n SkillBasin/v8/run_gate_a_parallel.sh          PASS
python3 SkillBasin/v8/eval_gate_a.py --help           PASS
python3 -m pytest -q SkillBasin/v8/tests/test_gate_a_v8.py
...... [100%]
6 passed
```

合成 smoke test 使用 1 个合成 task、1 个 scene、N=4、K=2，并成功生成所有表格和三张图。实际结构检查摘要为：

```json
{
  "synthetic_smoke_test": true,
  "not_experimental_evidence": true,
  "num_sampled_poses": 4,
  "rollouts_per_pose": 2,
  "reference_rollouts": 2,
  "sr_reach": 0.75,
  "sr_ref": 1.0,
  "delta_reach": 0.25,
  "sampled_pose_success_min": 0.0,
  "sampled_pose_success_max": 1.0,
  "sampled_pose_success_std": 0.4330127018922193,
  "sampled_pose_success_range": 1.0
}
```

上述数字来自人工定义的合成 runner，只验证代码结构，不能作为实验结果。

## 待正式环境完成

当前工作区没有 `Mobiπ_code/ckpts` 和 `Mobiπ_code/data`，因此没有把合成 smoke 冒充为真实策略实验。补齐 checkpoint 与 data 后，先按 README 的正式命令改成 `--num-sampled-poses 4 --rollouts-per-pose 2` 做单 task/scene 真机仿真 smoke；通过后再恢复 N=48、K=5。

如果 RoboCasa environment initial pose 不是该 checkpoint 对应 demonstration collection 的固定站位，应先按 `reference_registry.template.json` 生成审计过的 world-frame reference registry，并在正式命令中显式传入。

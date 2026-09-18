# Gate A v8：N2M 可达区域实验

本目录是从 v7 独立重构出的 Gate A 实现；`SkillBasin/` 根目录中的 v7 文件未被修改。

## 实验问题

在固定的 N2M 风格可达区域内，冻结操作策略是否表现出明显的站位依赖？相对于独立于当前 rollout 的 RoboCasa reference pose，朴素地从可达区域初始化会损失多少成功率？

默认候选域严格定义为：

```text
目标中心 1 m 半径圆 ∩ reference pose 周围 1 m × 1 m 方形
yaw = reference yaw + Uniform(-30°, +30°)
```

只拒绝底盘越界、底盘碰撞以及 simulator reset/spawn 失败。默认路径不使用手臂 IK、可见性、manipulability、几何评分或策略 rollout 结果筛选候选。

## 唯一主指标

- `SR_reach`：N 个采样位姿各自成功率的算术平均。
- `SR_ref`：reference pose 使用同一组 K 个 seeds 的成功率。
- `Delta_reach = SR_ref - SR_reach`。

每个位姿的 `min/max/mean/std/median/q25/q75/range` 仅描述区域内变化；其中 `max` 不参与选点或复评。basin threshold 分析默认关闭。

## 文件对应关系

| 文件 | v8 职责 |
| --- | --- |
| `gate_a_protocol.py` | 纯函数采样、统一位姿评估、统计 dataclass 和固定 seeds |
| `eval_gate_a.py` | RoboCasa 环境、冻结策略 rollout、合法底盘检查和逐场景落盘 |
| `gate_a_io.py` | CSV/JSON 与三张预注册图 |
| `analyze_gate_a.py` | 跨 policy seed / scene 的层级 bootstrap 汇总 |
| `gate_a_v8_config.json` | 默认参数快照 |
| `gate_a_v8_preregistered_manifest.json` | 实验问题、允许的过滤和指标预注册 |
| `reference_registry.template.json` | 可选的 demonstration reference world-pose 注册表格式 |
| `run_smoke_test.py` | N=4、K=2 的纯合成结构测试，不是论文结果 |
| `tests/test_gate_a_v8.py` | 六项协议回归测试 |

## Reference pose

默认使用当前 RoboCasa task/environment 的固定初始底盘位姿，来源会记录为 `robocasa_task_environment_initial_pose`。若已有逐 task/scene/policy 审计过的 demonstration pose，使用 `--reference-registry` 指向 schema v8 注册表；注册表必须提供 world-frame `[x, y, yaw]` 和可追溯 `source`，不能由当前 Gate A rollout 成败推导。

## 正式运行

在 `SkillBasin` 目录下运行（示例先跑一个 task/scene/policy）：

```bash
bash v8/run_gate_a_parallel.sh \
  --asset-root assets/robocasa \
  --ckpt-root artifacts/mobipi/ckpts \
  --data-root artifacts/mobipi/data \
  --output-root results/gate_a_v8_stove_scene0 \
  --tasks TurnOnStove \
  --scenes 0 \
  --policy-seeds 1 \
  --num-sampled-poses 48 \
  --rollouts-per-pose 5 \
  --policy-version YOUR_POLICY_VERSION \
  --policy-git-commit YOUR_POLICY_GIT_COMMIT
```

若使用审计注册表，再加：

```bash
--reference-registry /absolute/path/to/reference_registry.json
```

每个 scene 输出：`config.json`、`metadata.json`、`poses.csv`、`sampling_attempts.csv`、`rollouts.csv`、`summary.json`，以及：

- `figures/pose_success_map.png`
- `figures/pose_success_distribution.png`
- `figures/reach_vs_reference.png`

跨场景汇总：

```bash
python3 v8/analyze_gate_a.py results/gate_a_v8_stove_scene0
```

## 验证

```bash
python3 -m pytest -q v8/tests/test_gate_a_v8.py
python3 v8/run_smoke_test.py --output-root v8/smoke_output
python3 v8/analyze_gate_a.py v8/smoke_output --bootstrap-iterations 500
```

合成 smoke 输出带有 `synthetic_smoke_test=true` 和 `not_experimental_evidence=true`，只能验证数据结构、统计与绘图链路。

## v7 处理

v7 保留在父目录供历史结果复现。v8 不导入 `gate_a_geometry.py`，也没有几何选点、经验最优位姿选择或关键轨迹 IK 的兼容入口；相关 legacy flags 固定为 `false`，若试图开启会直接报错。

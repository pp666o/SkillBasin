# Gate A v7：几何可达域与策略成功域

核心问题：在已满足底盘、可见性和机械臂近交互点可达的站位集合内，同一 frozen policy 的成功率是否仍然显著变化？

H_geom = 样本内 empirical oracle 成功率 − 几何站位成功率。允许为零或负值。

四个摘要点为 p_geom、p_train、p_oracle，以及 pose registry 提供时的 p_mobipi。p_oracle 仅表示已 rollout 的几何可达样本内最优，不是全局 oracle。

## 执行流程

1. 读取 demonstrations 的训练位置，映射到当前场景。模拟器 spawn 仅可在 calibration 调试中显式充当代理。
2. 生成几何候选池，最多检查2500个候选。保留近训练位置样本和地板范围采样，没有固定环带。
3. hard gate 只要求底盘 footprint 合法、初始无 blocking collision、目标渲染可见、靠近交互点的位置 IK residual 不超阈值。四个 6D waypoint、朝向、插值碰撞、joint margin 和 manipulability 仅作连续评分与诊断。
4. 在几何可达集内分层抽取候选，每点多次 rollout 建立 sampled GT basin，并从中选择 p_oracle。p_geom 必须包含在该样本中。
5. 用与 basin discovery 独立的配对种子验证 p_geom、p_train、p_oracle 和可选 p_mobipi；同一 pose 合并执行。
6. 每个场景输出 G(p) 对 Sπ(p) 散点图、GT basin CSV，以及 success range/std 和 basin 内外混合统计。
7. 用配对分层 bootstrap 计算 H_geom 与95%区间。异常终止该分片，不能当作策略失败。校准集不产生正式通过结论。

默认判定：至少两个任务的 H_geom 区间下界达到10个百分点，且 p_oracle 平均成功率至少20%。这是实用门槛，不是理论定理。

## 几何数据必须核对

gate_a_geometry.template.json 是格式模板，里面的点和矩阵不是已校准事实，不能仅将 reviewed 改为 true 来运行。

每个任务的 default 或 scenes[场景编号] 需要提供：

- frame_type / frame_name：实际交互 geom 或 site，支持 {prefix}、{knob} 名称替换。
- point_local：交互表面点在该坐标系中的位置（米），不是默认采用转轴中心。
- normal_local：朝外的接近法向。
- eef_rotation_local：末端 site 的目标旋转矩阵，以同一交互坐标系表示；须核对末端轴约定。
- source：几何校准记录来源；reviewed：核对完成后的标记。

旋钮和水龙头要按具体资产找到表面点、把手和正确接近姿态；不能仅根据家具中心差向量推算。训练位姿CSV中的 target_x/target_y 必须是同一交互点，target_yaw 必须是同一家具朝向，否则映射会错位。

当前检查止于接触前的安全间隙，默认1.5厘米；它不保证完整的按压、旋转轨迹。MuJoCo 已禁用的碰撞对不会凭空产生接触，需要在资产校准时检查 collision masks。严格 6D trajectory IK 失败仅是保守诊断，不能反证真正不可达。

## 运行

先由带来源字段的 demonstration CSV 创建 pose registry：

```bash
python3 prepare_gate_a_pose_registry.py --demo-csv /absolute/path/demos.csv --mobipi-csv /absolute/path/mobipi_selector.csv --output /absolute/path/poses.json
export GATEA_POSE_REGISTRY=/absolute/path/poses.json
export GATEA_GEOMETRY_CONFIG=/absolute/path/reviewed_geometry.json
bash run_gate_a_parallel.sh
```

默认结果目录为 gate_a_v7_test。分析器拒绝旧版数据，不会重解释2026-09-12的结果。训练数据与测试布局是否分离，仍需对实际 checkpoint 的数据来源核查。

## 验证边界

本地测试覆盖：几何选点不读取成功率、位置正确但朝向错误会失败、路径中间碰撞会失败、求解后状态恢复、同位置结果复用、负差距、异常拒绝、历史数据拒绝、随机种子范围。

正式仿真还需要实际策略权重、运行依赖和经过核对的接触配置；单元测试不代替真实资产上的几何校准。旧v5预注册文件与历史报告保留供追溯；当前入口是v7文件。

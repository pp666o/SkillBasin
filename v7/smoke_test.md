# Gate A v7 最小实验方案

目标：验证几何可达域是否不等于 frozen policy 的经验成功域。

主指标：H_geom = S_val(p_oracle) - S_val(p_geom)。p_oracle 仅是 sampled geometry-feasible set 内的 empirical oracle。

训练位置来自 demonstrations；默认 spawn 必须核对，不能自动冒充训练位置。
几何 hard gate 只含底盘 footprint、初始碰撞、渲染可见性和近交互点位置残差；严格 6D trajectory IK 仅作评分和诊断。
对几何可达样本做多次 rollout，输出 G(p) 对 Sπ(p) 散点图和 GT basin CSV；再用独立配对种子验证 p_geom、p_train、p_oracle 及可选 p_mobipi。
异常终止分片，不作为操作失败。结果只接受 protocol_version=7。

完整说明见 [GateA_v7_使用说明.md](./GateA_v7_使用说明.md)。
v5与2026-09-12报告属于旧协议，不能按新指标重解释。

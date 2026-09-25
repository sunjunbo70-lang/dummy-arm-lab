# 当前wall_cycle观测来源审查

L1静态代码核对；未训练、未接硬件。env.py:115 _scan调用D435Proxy.scan(material.wall)。sensor.py代理直接对真实高度场加偏差、倾斜、噪声、多帧融合近似与无效置信度，未通过相机投影/渲染深度/点云重建。_observe将height/confidence按2×4池化，形成厚度/误差/置信度输入。5mm原格对应名义10×20mm池化尺度，教师使用池化前scan地图。

不能声称当前观测完全无真值：_observe直接读blade_volume_m3、carry_loss_m3及归一化用supplied_m3+initial_m3；last_improvement来自真实quality_cost差；stall_count和fail_map也间接来自该真值改善。best_score来源为带噪声scan，不同字段应分别归类。奖励与最终评估读真值本身可接受，但真值派生量进入actor输入是部署差距。

v0.9实施前应增加观测契约：actor仅允许观测/可估计状态；用扫描估计改善/失败记忆，余料及损失设估计器或剔除；真值仅用于奖励/评估（若特权critic需明确单独设计）。真实渲染深度或D435实测重建为独立感知验收，不把当前代理叫完整相机模拟。对比oracle/proxy-clean/当前混合观测以量化差距。此为新发现补充，旧方案尚未实现该修正。

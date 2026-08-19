# FR-05 特定异常行为识别

本目录集中实现需求书 FR-05 的三类行为：

- 行人/二轮车突然加速（统一状态 `running`）
- 行人跌倒、二轮车侧倒及盲区跌倒记忆（`falling`）
- 机动车/摩托车同本车道逆行（`wrong_way`）

主入口是 `FR05Recognizer`。根目录 `pipeline.py` 与 `all/pipeline.py` 都在统一
目标检测、跟踪和姿态特征提取之后调用这个唯一实现，因此逆行检测直接复用
主流程的 `tracks`，不会再次运行 YOLO 或 ByteTrack。

处理顺序：

1. 绑定骑手与二轮车；
2. 基于时序速度和骨骼姿态判定奔跑/跌倒；
3. 基于车道、车辆朝向和连续帧状态机判定逆行；
4. 按 `falling > wrong_way > running > suspect_wrong_way > normal` 合并行为；
5. 将盲区跌倒记忆目标追加回主流程。

外包逆行模块中原有的独立检测器、跟踪器、Pipeline 和 Demo 已移除。保留的
`vehicle_orientation.py`、`yolopv2_lane.py`、`wrong_way.py` 及模型权重只负责
生成逆行判定所需特征。

## 接口

- `FR05Recognizer.process_frame()` 返回 `FR05Result`，同时兼容原有 tracks 就地更新。
- `interfaces.py` 定义最小 Track 输入契约、`FR05Event`、模块状态和降级诊断。
- 主流程会将状态写入 `FrameResult.fr05_*`，FR-09 报文输出顶层 `fr05`。
- 详细联调规范见 `docs/FR05上下游接口报告.pdf`。

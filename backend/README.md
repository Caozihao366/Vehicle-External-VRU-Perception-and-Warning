# 集成后端（Release Backend）

本目录是项目的独立集成入口，供命令行运行、桌面GUI和PyInstaller Release共同使用。

## 组成原则

- 以根目录已经验证的融合流水线为基线；
- 只吸收 `all/` 在提交 `b07069f` 中真正新增的异步QCNet预测能力；
- 不继承 `all/` 中旧的时间步长、TTC相对速度、报文和可视化回退；
- 根目录与 `all/` 保持不变，后续集成修改集中在本目录。

## 关键行为

- 检测、行为模块继续使用处理时钟；BEV、历史、QCNet和TTC使用视频/图片源时间；
- QCNet模型前向在独立线程中执行，主流水线维护完整源时间历史；
- 异步轨迹会按结果年龄推进到当前自车坐标系，超过0.5秒则不参与TTC；
- QCNet不可用、历史不足、结果过期或目标无有效轨迹时，TTC按目标匀速模型回退；
- GroundPlane输出已经是相对自车速度，匀速TTC不会再次扣除自车速度；
- CUDA不可用时检测和QCNet自动切换到CPU，CPU检测自动禁用FP16。

## 直接运行

在Python 3.10完整环境中：

```bat
cd /d D:\shixi\vruproject\Vehicle-External-VRU-Perception-and-Warning\backend
python main.py --video "视频路径.mp4" --ego-speed 10 --show-trajectories
```

仅进行匀速回退测试时，可由桌面接口传入 `enable_qcnet=False`；命令行暂保持原项目参数兼容。

## 桌面GUI接入

桌面壳的后端根目录应指向本目录，而不是仓库根目录或 `all/`：

```text
<repository>/backend
```

`Pipeline` 对桌面层提供以下可选构造参数：

- `enable_qcnet`
- `qcnet_device`
- `qcnet_min_history`
- `qcnet_frequency_hz`

输出 `FrameResult` 保留现有字段，并新增异步结果帧号和结果年龄字段。JSON报文在 `performance` 中同步输出这些状态。

## Release边界

源码仓库提交 `backend/`；PyInstaller生成的 `build/`、`dist/` 和测试视频不提交。最终Windows压缩包应作为GitHub Release附件发布。

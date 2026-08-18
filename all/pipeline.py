"""
主流水线编排器
==============
串联所有模块:
  传感器质量 → VRU检测跟踪 → FR-05异常行为 → 地面投影 → 轨迹预测 → TTC风险

单路前视视频模式（BEV 退化为 2D 地面平面投影）。
"""
import time
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from system_config import *
from vru_perception import VRUPerception

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(_PROJECT_ROOT))

from fr05 import FR05Recognizer
from ground_plane import GroundPlaneProjector
from ttc_engine import TTCEngine, RiskResult
from bev_undistort_pipeline3.async_trajectory_predictor import (
    AsyncTrajectoryPredictor,
)


@dataclass
class FrameResult:
    """一帧的完整处理结果，供 UI 渲染。"""
    frame_id: int
    timestamp: float
    fps: float
    frame: np.ndarray
    annotated_frame: np.ndarray = None

    # 2D 跟踪结果
    tracks: list = field(default_factory=list)
    ego_motion: dict = field(default_factory=dict)
    count: dict = field(default_factory=dict)

    # FR-05 异常行为模块
    fr05_status: str = "unavailable"
    fr05_events: list = field(default_factory=list)
    fr05_latency_ms: float = 0.0
    fr05_capabilities: dict = field(default_factory=dict)
    fr05_warnings: list = field(default_factory=list)

    # 传感器质量
    sensor_status: str = "normal"
    sensor_score: float = 1.0
    blind_zones: list = field(default_factory=list)

    # BEV / 投影
    bev_frame: object = None
    projected: list = field(default_factory=list)

    # 风险
    risk_results: List[RiskResult] = field(default_factory=list)

    # QCNet 状态
    qcnet_ready: bool = False
    qcnet_history_frames: int = 0

    # 处理延迟
    process_ms: float = 0.0

    # 自车速度 (km/h，供 UI 显示)
    ego_speed_kmh: float = 0.0
    ego_speed_mps: float = 0.0
    ego_yaw_rate: float = 0.0


class Pipeline:
    """
    主流水线。

    用法:
        pipe = Pipeline(video_source="test.mp4")
        for result in pipe:
            ui.render(result)
    """

    def __init__(
        self,
        video_source=None,
        ego_speed_mps: float = EGO_SPEED_MPS,
        ego_yaw_rate_rps: float = EGO_YAW_RATE_RPS,
        enable_sensor_quality: bool = ENABLE_SENSOR_QUALITY,
        enable_wrong_way: bool = ENABLE_WRONG_WAY,
        enable_yolopv2: bool = ENABLE_YOLOPV2,
        det_device=DET_DEVICE,
    ):
        self.video_source = video_source if video_source is not None else DEFAULT_VIDEO
        self.ego_speed = ego_speed_mps
        self.ego_yaw_rate = ego_yaw_rate_rps
        self.enable_sensor_quality = enable_sensor_quality
        self.enable_wrong_way = enable_wrong_way

        # 打开视频源（支持视频文件/摄像头/图片目录）
        self._image_list = None
        self._image_idx = 0
        src_path = Path(str(self.video_source))

        if src_path.is_dir():
            # 图片序列模式
            exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            self._image_list = sorted([
                str(p) for p in src_path.iterdir()
                if p.suffix.lower() in exts
            ])
            if not self._image_list:
                raise RuntimeError(f"图片目录为空: {src_path}")
            first = cv2.imread(self._image_list[0])
            if first is None:
                raise RuntimeError(f"无法读取图片: {self._image_list[0]}")
            self.height, self.width = first.shape[:2]
            self.fps_video = 10.0  # A2D2 序列约 2Hz，按 10Hz 处理
            print(f"[pipeline] 图片序列: {len(self._image_list)} 张, "
                  f"{self.width}x{self.height}")
        else:
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开视频源: {self.video_source}")
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps_video = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            print(f"[pipeline] 视频: {self.width}x{self.height} @ {self.fps_video:.1f}fps")

        self.frame_id = 0

        # ---- 初始化各模块 ----
        print("[pipeline] 初始化 VRU 感知模块...")
        self.perception = VRUPerception(
            det_classes=DET_CLASSES,
            device=det_device,
            conf=DET_CONF,
            half=DET_HALF,
        )

        print("[pipeline] 初始化 FR-05 异常行为识别...")
        self.fr05 = FR05Recognizer(
            enable_wrong_way=self.enable_wrong_way,
            enable_lane=enable_yolopv2,
            device="cpu" if det_device == "cpu" else "auto",
            orient_weights=str(FR05_ORIENT_WEIGHTS),
            yolopv2_weights=str(FR05_YOLOPV2_WEIGHTS),
        )

        if self.enable_sensor_quality:
            print("[pipeline] 初始化传感器质量模块...")
            from sensor_quality import SensorQualityManager
            self.sensor_mgr = SensorQualityManager()
        else:
            self.sensor_mgr = None

        print("[pipeline] 初始化地面投影...")
        self.projector = GroundPlaneProjector(
            image_size=(self.width, self.height),
            camera_height_m=CAMERA_HEIGHT_M,
            camera_pitch_deg=CAMERA_PITCH_DEG,
            hfov_deg=CAMERA_HFOV_DEG,
            max_dist_m=MAX_PROJECTION_DIST_M,
            min_dist_m=MIN_PROJECTION_DIST_M,
        )

        print("[pipeline] 初始化 TTC 引擎...")
        self.ttc = TTCEngine()

        # QCNet 只在这里初始化一次；逐帧 forward 由独立线程执行。
        self._trajectory_predictor = None
        self._start_trajectory_predictor()

        self._prev_time = None
        print("[pipeline] 初始化完成。")

    def __iter__(self):
        return self

    def __next__(self) -> FrameResult:
        if self._image_list is not None:
            if self._image_idx >= len(self._image_list):
                raise StopIteration
            frame = cv2.imread(self._image_list[self._image_idx])
            self._image_idx += 1
            if frame is None:
                raise StopIteration
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                raise StopIteration

        t_start = time.time()
        self.frame_id += 1

        now = time.time()
        dt = max(now - self._prev_time, 0.001) if self._prev_time else 1.0 / self.fps_video
        self._prev_time = now
        timestamp = now

        result = FrameResult(
            frame_id=self.frame_id,
            timestamp=timestamp,
            fps=1.0 / dt if dt > 0 else 0.0,
            frame=frame,
            ego_speed_kmh=self.ego_speed * 3.6,
            ego_speed_mps=self.ego_speed,
            ego_yaw_rate=self.ego_yaw_rate,
        )

        # ---- 1. 传感器质量 ----
        if self.sensor_mgr is not None:
            try:
                sq = self.sensor_mgr.process_frame(
                    frame, timestamp=timestamp, frame_id=self.frame_id
                )
                result.sensor_status = sq.global_status.value if hasattr(sq.global_status, 'value') else str(sq.global_status)
                result.sensor_score = sq.global_quality_score
                result.blind_zones = sq.blind_zones
            except Exception as e:
                print(f"[pipeline] 传感器质量检测异常: {e}")
                result.sensor_status = "unknown"

        # ---- 2. VRU 检测 + 跟踪 + 姿态特征 ----
        perc_out = self.perception.process_frame(frame, dt)
        result.tracks = perc_out["tracks"]
        result.ego_motion = perc_out["ego_motion"]
        result.count = perc_out["count"]

        # ---- 3. FR-05 异常行为（奔跑 / 跌倒 / 逆行）----
        fr05_output = self.fr05.process_frame(
            frame,
            result.tracks,
            dt,
            self.frame_id,
            timestamp=timestamp,
            sensor_status=result.sensor_status,
        )
        result.tracks = fr05_output.tracks
        result.fr05_status = fr05_output.status
        result.fr05_events = fr05_output.events
        result.fr05_latency_ms = fr05_output.latency_ms
        result.fr05_capabilities = fr05_output.capabilities
        result.fr05_warnings = fr05_output.warnings

        # ---- 4. 地面投影 → BEVFrame ----
        bev_frame, projected = self.projector.project_tracks(
            result.tracks, dt,
            ego_speed_mps=self.ego_speed,
            ego_yaw_rate_rps=self.ego_yaw_rate,
            frame_id=self.frame_id,
            timestamp=timestamp,
        )
        result.bev_frame = bev_frame
        result.projected = projected

        # ---- 5. 异步 QCNet 轨迹预测 ----
        qcnet_predictions = None
        if self._trajectory_predictor is not None:
            try:
                # submit() 只入队，不在 BEV 主线程中执行模型 forward。
                self._trajectory_predictor.submit(bev_frame)
                packet = self._trajectory_predictor.get_latest()
                if packet is not None:
                    qcnet_predictions = packet.result
                    result.qcnet_ready = True

                result.qcnet_history_frames = (
                    self._trajectory_predictor.history_frames
                )
                error = self._trajectory_predictor.get_latest_error()
                if error is not None:
                    print(f"[pipeline] QCNet 异步推理异常，使用匀速模型: {error}")
            except (TypeError, RuntimeError) as e:
                print(f"[pipeline] QCNet 预测线程不可用，使用匀速模型: {e}")

        # ---- 6. TTC 风险评估 ----
        result.risk_results = self.ttc.evaluate(
            bev_frame, projected,
            qcnet_predictions=qcnet_predictions,
            ego_speed_mps=self.ego_speed,
            ego_yaw_rate_rps=self.ego_yaw_rate,
            sensor_status=result.sensor_status,
        )

        result.process_ms = (time.time() - t_start) * 1000
        return result

    def _start_trajectory_predictor(self):
        """Load QCNet once and start its prediction worker."""

        # The current monocular projector emits each frame in that frame's Ego
        # coordinates and has no global pose/delta with which to express all
        # history in the latest fixed t0 frame. Per the QCNet coordinate
        # contract, only static-ego data is valid until that pose is supplied.
        if abs(self.ego_speed) > 1e-6 or abs(self.ego_yaw_rate) > 1e-6:
            print(
                "[pipeline] 当前 BEV 未提供固定 t0 Ego 坐标转换，"
                "非静止自车场景禁用 QCNet，TTC 使用匀速模型。"
            )
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            predictor = AsyncTrajectoryPredictor(
                checkpoint_path=str(QCNET_CHECKPOINT),
                device=device,
            )
            predictor.start()
            self._trajectory_predictor = predictor
            print(f"[pipeline] QCNet 异步轨迹预测已在 {device} 上启动。")
        except ImportError as e:
            print(f"[pipeline] torch_geometric 未安装，QCNet 不可用，TTC 使用匀速模型: {e}")
        except Exception as e:
            print(f"[pipeline] QCNet 初始化失败: {e}")

    def reset_history(self):
        """Reset QCNet history after a scene, localization, or MOT reset."""

        if self._trajectory_predictor is not None:
            self._trajectory_predictor.reset_history()

    def release(self):
        if self._trajectory_predictor is not None:
            try:
                self._trajectory_predictor.stop()
            except RuntimeError as e:
                print(f"[pipeline] QCNet 预测线程停止失败: {e}")
            finally:
                self._trajectory_predictor = None
        if self._image_list is None:
            self.cap.release()

"""
主流水线编排器
==============
串联所有模块:
  传感器质量 → VRU检测跟踪 → 逆行检测 → 地面投影 → 历史缓冲 → TTC风险

单路前视视频模式（BEV 退化为 2D 地面平面投影）。
"""
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from system_config import *
from vru_perception import VRUPerception
from wrong_way_adapter import WrongWayAdapter
from ground_plane import GroundPlaneProjector
from history_buffer import HistoryBuffer
from ttc_engine import TTCEngine, RiskResult


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
    qcnet_loaded: bool = False
    qcnet_active: bool = False
    qcnet_error: str = ""
    qcnet_predictions: dict = field(default_factory=dict)

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
        self._prev_source_timestamp = None
        self._live_camera = isinstance(self.video_source, int)

        # ---- 初始化各模块 ----
        print("[pipeline] 初始化 VRU 感知模块...")
        self.perception = VRUPerception(
            det_classes=DET_CLASSES,
            device=det_device,
            conf=DET_CONF,
            half=DET_HALF,
        )

        if self.enable_wrong_way:
            print("[pipeline] 初始化逆行检测模块...")
            self.wrong_way = WrongWayAdapter(
                device=str(det_device) if det_device != 0 else "cuda",
                enable_lane=enable_yolopv2,
            )
        else:
            self.wrong_way = None

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

        print("[pipeline] 初始化历史缓冲...")
        self.history = HistoryBuffer()

        print("[pipeline] 初始化 TTC 引擎...")
        self.ttc = TTCEngine()

        # QCNet（延迟加载，需要 torch_geometric）
        self._qcnet = None
        self._qcnet_tried = False
        self._qcnet_error = ""

        self._prev_time = None
        print("[pipeline] 初始化完成。")

    def __iter__(self):
        return self

    def __next__(self) -> FrameResult:
        if self._image_list is not None:
            if self._image_idx >= len(self._image_list):
                raise StopIteration
            image_path = self._image_list[self._image_idx]
            frame = cv2.imread(image_path)
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
        processing_dt = max(now - self._prev_time, 0.001) if self._prev_time else 1.0 / self.fps_video
        self._prev_time = now
        timestamp = now

        # 仅 BEV/TTC/QCNet 使用数据源时间。检测、跟踪和行为模块继续使用
        # 原有的处理时钟，避免本次 TTC 修复改变其他模块的运行语义。
        if self._image_list is not None:
            source_timestamp = self._image_timestamp(image_path)
            if source_timestamp is None:
                source_timestamp = (self.frame_id - 1) / self.fps_video
        elif self._live_camera:
            source_timestamp = now
        else:
            source_timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if source_timestamp <= 0.0:
                source_timestamp = (self.frame_id - 1) / self.fps_video

        if self._prev_source_timestamp is None:
            source_dt = 1.0 / self.fps_video
        else:
            source_dt = source_timestamp - self._prev_source_timestamp
            if source_dt <= 0.0:
                source_dt = 1.0 / self.fps_video
        self._prev_source_timestamp = source_timestamp

        result = FrameResult(
            frame_id=self.frame_id,
            timestamp=timestamp,
            fps=1.0 / processing_dt if processing_dt > 0 else 0.0,
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

        # ---- 2. VRU 检测 + 跟踪 + 姿态 + 行为 ----
        perc_out = self.perception.process_frame(frame, processing_dt)
        result.tracks = perc_out["tracks"]
        result.ego_motion = perc_out["ego_motion"]
        result.count = perc_out["count"]

        # ---- 3. 逆行检测 ----
        if self.wrong_way is not None:
            try:
                self.wrong_way.process_frame(frame, result.tracks, self.frame_id)
            except Exception as e:
                print(f"[pipeline] 逆行检测异常: {e}")

        # ---- 4. 地面投影 → BEVFrame ----
        bev_frame, projected = self.projector.project_tracks(
            result.tracks, source_dt,
            ego_speed_mps=self.ego_speed,
            ego_yaw_rate_rps=self.ego_yaw_rate,
            frame_id=self.frame_id,
            timestamp=source_timestamp,
        )
        result.bev_frame = bev_frame
        result.projected = projected

        # ---- 5. 历史缓冲 ----
        self.history.append(bev_frame)
        result.qcnet_history_frames = len(self.history)
        result.qcnet_ready = self.history.ready(QCNET_MIN_HISTORY)

        # ---- 6. QCNet 轨迹预测（如果可用）----
        qcnet_predictions = None
        if result.qcnet_ready and not self._qcnet_tried:
            self._init_qcnet()
        if self._qcnet is not None and result.qcnet_ready:
            try:
                qcnet_input = self.history.build_qcnet_input(
                    ego_speed_mps=self.ego_speed,
                    ego_yaw_rate_rps=self.ego_yaw_rate,
                    output_frequency_hz=QCNET_FREQUENCY_HZ,
                )
                if qcnet_input is not None:
                    eligible_ids = set(qcnet_input["agent_ids"])
                    target_ids = [
                        str(p.track_id) for p in projected
                        if str(p.track_id) in eligible_ids
                    ]
                    if target_ids:
                        qcnet_predictions = self._qcnet.predict(
                            qcnet_input, target_ids=target_ids
                        )
            except Exception as e:
                print(f"[pipeline] QCNet 推理异常，使用匀速模型: {e}")
                self._qcnet_error = str(e)
                qcnet_predictions = None

        # 保留本帧真实推理状态与轨迹，供可选UI读取。
        # qcnet_ready仅代表历史足够，qcnet_active才代表模型实际产生了轨迹。
        result.qcnet_loaded = self._qcnet is not None
        result.qcnet_predictions = qcnet_predictions or {}
        result.qcnet_active = bool(result.qcnet_predictions)
        result.qcnet_error = self._qcnet_error

        # ---- 7. TTC 风险评估 ----
        result.risk_results = self.ttc.evaluate(
            bev_frame, projected,
            qcnet_predictions=qcnet_predictions,
            ego_speed_mps=self.ego_speed,
            ego_yaw_rate_rps=self.ego_yaw_rate,
        )

        result.process_ms = (time.time() - t_start) * 1000
        return result

    def _init_qcnet(self):
        """延迟初始化 QCNet（需要 torch_geometric）。"""
        self._qcnet_tried = True
        try:
            from ttc.integrations.qcnet import QCNetTrajectoryProvider
            self._qcnet = QCNetTrajectoryProvider(
                package_root=QCNET_DIR,
                device=QCNET_DEVICE,
            )
            self._qcnet_error = ""
            print("[pipeline] QCNet 模型加载成功（多模态轨迹预测）。")
        except ImportError as e:
            self._qcnet_error = str(e)
            print(f"[pipeline] torch_geometric 未安装，QCNet 不可用，TTC 使用匀速模型: {e}")
        except Exception as e:
            self._qcnet_error = str(e)
            print(f"[pipeline] QCNet 初始化失败: {e}")

    def release(self):
        if self._image_list is None:
            self.cap.release()

    @staticmethod
    def _image_timestamp(path: str):
        """读取数据集图片文件名末尾的微秒时间戳。"""
        match = re.search(
            r"(\d{13,})\.(?:jpg|jpeg|png|bmp)$", str(path), re.IGNORECASE
        )
        return int(match.group(1)) / 1_000_000.0 if match else None

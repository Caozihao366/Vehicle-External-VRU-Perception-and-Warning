"""
主流水线编排器
==============
串联所有模块:
  传感器质量 → VRU检测跟踪 → FR-05异常行为 → 地面投影 → 轨迹预测 → TTC风险

单路前视视频模式（BEV 退化为 2D 地面平面投影）。
"""
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from system_config import *
from vru_perception import VRUPerception
from fr05 import FR05Recognizer
from ground_plane import GroundPlaneProjector
from history_buffer import HistoryBuffer
from ttc_engine import TTCEngine, RiskResult
from bev_undistort_pipeline3.async_trajectory_predictor import AsyncTrajectoryPredictor
from ttc.integrations.qcnet import QCNetTargetPrediction
from ttc.models import PredictionMode, VRUTrajectoryPoint


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
    qcnet_loaded: bool = False
    qcnet_active: bool = False
    qcnet_error: str = ""
    qcnet_predictions: dict = field(default_factory=dict)
    qcnet_prediction_frame_id: Optional[int] = None
    qcnet_prediction_age_s: Optional[float] = None

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
        enable_qcnet: bool = True,
        qcnet_device: Optional[str] = None,
        qcnet_min_history: Optional[int] = None,
        qcnet_frequency_hz: Optional[float] = None,
    ):
        self.video_source = video_source if video_source is not None else DEFAULT_VIDEO
        self.ego_speed = ego_speed_mps
        self.ego_yaw_rate = ego_yaw_rate_rps
        self.enable_sensor_quality = enable_sensor_quality
        self.enable_wrong_way = enable_wrong_way
        self.enable_qcnet = bool(enable_qcnet)
        self.qcnet_device = qcnet_device or QCNET_DEVICE
        self.qcnet_min_history = int(qcnet_min_history or QCNET_MIN_HISTORY)
        self.qcnet_frequency_hz = float(qcnet_frequency_hz or QCNET_FREQUENCY_HZ)

        if det_device == 0 and not torch.cuda.is_available():
            print("[pipeline] CUDA不可用，目标检测自动切换到CPU。")
            det_device = "cpu"
        det_half = bool(DET_HALF and det_device != "cpu" and torch.cuda.is_available())

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
            half=det_half,
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

        print("[pipeline] 初始化源时间轨迹历史...")
        self.history = HistoryBuffer()

        # all/ 最新提交新增的异步QCNet能力，在独立集成目录中使用。
        self._trajectory_predictor = None
        self._latest_qcnet_packet = None
        self._qcnet_error = ""
        if self.enable_qcnet:
            self._start_trajectory_predictor()
        else:
            self._qcnet_error = "QCNet已关闭"

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

        # ---- 2. VRU 检测 + 跟踪 + 姿态特征 ----
        perc_out = self.perception.process_frame(frame, processing_dt)
        result.tracks = perc_out["tracks"]
        result.ego_motion = perc_out["ego_motion"]
        result.count = perc_out["count"]

        # ---- 3. FR-05 异常行为（奔跑 / 跌倒 / 逆行）----
        fr05_output = self.fr05.process_frame(
            frame,
            result.tracks,
            processing_dt,
            self.frame_id,
            timestamp=source_timestamp,
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
            result.tracks, source_dt,
            ego_speed_mps=self.ego_speed,
            ego_yaw_rate_rps=self.ego_yaw_rate,
            frame_id=self.frame_id,
            timestamp=source_timestamp,
        )
        result.bev_frame = bev_frame
        result.projected = projected

        # ---- 5. 源时间历史 + 异步QCNet轨迹预测 ----
        self.history.append(bev_frame)
        result.qcnet_history_frames = len(self.history)
        result.qcnet_ready = self.history.ready(self.qcnet_min_history)
        qcnet_predictions = {}
        if self._trajectory_predictor is not None:
            try:
                if result.qcnet_ready:
                    model_input = self.history.build_qcnet_input(
                        ego_speed_mps=self.ego_speed,
                        ego_yaw_rate_rps=self.ego_yaw_rate,
                        output_frequency_hz=self.qcnet_frequency_hz,
                    )
                    if model_input is not None:
                        eligible_ids = set(model_input["agent_ids"])
                        target_ids = [
                            str(p.track_id) for p in projected
                            if str(p.track_id) in eligible_ids
                        ]
                        if target_ids:
                            self._trajectory_predictor.submit(
                                bev_frame, model_input, target_ids
                            )
                packet = self._trajectory_predictor.get_latest()
                if packet is not None:
                    self._latest_qcnet_packet = packet
                worker_error = self._trajectory_predictor.get_latest_error()
                if worker_error is not None:
                    self._qcnet_error = str(worker_error)

                if self._latest_qcnet_packet is not None:
                    packet = self._latest_qcnet_packet
                    prediction_age = float(source_timestamp - packet.timestamp)
                    result.qcnet_prediction_frame_id = packet.frame_id
                    result.qcnet_prediction_age_s = max(0.0, prediction_age)
                    if -1e-3 <= prediction_age <= QCNET_MAX_RESULT_AGE_S + 1e-6:
                        current_ids = {str(p.track_id) for p in projected}
                        qcnet_predictions = self._align_async_predictions(
                            packet.result,
                            max(0.0, prediction_age),
                            current_ids,
                        )
                        # 有新鲜结果时清除之前帧留下的过期/运行错误。
                        self._qcnet_error = ""
                    elif prediction_age > QCNET_MAX_RESULT_AGE_S:
                        self._qcnet_error = (
                            f"异步轨迹已过期({prediction_age:.2f}s)，本帧使用匀速回退"
                        )
            except Exception as e:
                print(f"[pipeline] QCNet异步推理异常，使用匀速模型: {e}")
                self._qcnet_error = str(e)

        result.qcnet_loaded = self._trajectory_predictor is not None
        result.qcnet_predictions = qcnet_predictions
        result.qcnet_active = bool(result.qcnet_predictions)
        result.qcnet_error = self._qcnet_error

        # ---- 6. TTC 风险评估 ----
        result.risk_results = self.ttc.evaluate(
            bev_frame, projected,
            qcnet_predictions=qcnet_predictions or None,
            ego_speed_mps=self.ego_speed,
            ego_yaw_rate_rps=self.ego_yaw_rate,
            sensor_status=result.sensor_status,
        )

        result.process_ms = (time.time() - t_start) * 1000
        return result

    def _start_trajectory_predictor(self):
        """加载一次QCNet并启动独立预测线程。"""
        try:
            device = self.qcnet_device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
                print("[pipeline] CUDA不可用，QCNet自动切换到CPU。")
            predictor = AsyncTrajectoryPredictor(
                checkpoint_path=str(QCNET_CHECKPOINT),
                device=device,
            )
            predictor.start()
            self._trajectory_predictor = predictor
            print(f"[pipeline] QCNet异步轨迹预测已在{device}上启动。")
        except ImportError as e:
            self._qcnet_error = str(e)
            print(f"[pipeline] torch_geometric 未安装，QCNet 不可用，TTC 使用匀速模型: {e}")
        except Exception as e:
            self._qcnet_error = str(e)
            print(f"[pipeline] QCNet 初始化失败: {e}")

    def reset_history(self):
        """场景切换、定位重置或跟踪器重置后清空轨迹历史。"""
        self._latest_qcnet_packet = None
        self._qcnet_error = ""
        self.history = HistoryBuffer()
        if self._trajectory_predictor is not None:
            self._trajectory_predictor.reset_history()

    def _align_async_predictions(self, predictions, age_s, current_ids):
        """将较早t0产生的异步轨迹推进并转换到当前自车坐标系。"""
        ego_x, ego_y, ego_yaw = self._future_ego_pose(age_s)
        c, s = np.cos(ego_yaw), np.sin(ego_yaw)
        aligned = {}
        for track_id, prediction in (predictions or {}).items():
            if str(track_id) not in current_ids:
                continue
            modes = []
            for mode in prediction.modes:
                points = []
                for point in mode.points:
                    remaining = float(point.t_s) - age_s
                    if remaining <= 1e-6:
                        continue
                    dx, dy = float(point.x) - ego_x, float(point.y) - ego_y
                    points.append(VRUTrajectoryPoint(
                        t_s=remaining,
                        x=c * dx + s * dy,
                        y=-s * dx + c * dy,
                        sigma_x_m=point.sigma_x_m,
                        sigma_y_m=point.sigma_y_m,
                    ))
                if points:
                    modes.append(PredictionMode(
                        mode_id=mode.mode_id,
                        probability=mode.probability,
                        points=tuple(points),
                    ))
            if modes:
                horizon = max(point.t_s for mode in modes for point in mode.points)
                aligned[str(track_id)] = QCNetTargetPrediction(
                    track_id=str(track_id),
                    modes=tuple(modes),
                    inference_time_ms=prediction.inference_time_ms,
                    prediction_horizon_s=horizon,
                    prediction_source=f"{prediction.prediction_source}_async",
                )
        return aligned

    def _future_ego_pose(self, elapsed_s):
        t = float(elapsed_s)
        speed = float(self.ego_speed)
        yaw_rate = float(self.ego_yaw_rate)
        yaw = yaw_rate * t
        if abs(yaw_rate) < 1e-8:
            return speed * t, 0.0, 0.0
        radius = speed / yaw_rate
        return (
            float(radius * np.sin(yaw)),
            float(radius * (1.0 - np.cos(yaw))),
            float(yaw),
        )

    def release(self):
        if self._trajectory_predictor is not None:
            try:
                self._trajectory_predictor.stop()
            finally:
                self._trajectory_predictor = None
        if self._image_list is None:
            self.cap.release()

    @staticmethod
    def _image_timestamp(path: str):
        """读取数据集图片文件名末尾的微秒时间戳。"""
        match = re.search(
            r"(\d{13,})\.(?:jpg|jpeg|png|bmp)$", str(path), re.IGNORECASE
        )
        return int(match.group(1)) / 1_000_000.0 if match else None

"""
FR-09 报文组装与总线输出
========================
将 FrameResult 转换为需求文档规定的 JSON 报文格式。

输出字段:
  timestamp, frame_id, ego_status, sensor_availability,
  vru_objects[], performance
"""
import json
import time
from typing import Any, Dict, List, Optional


# 行为状态码
_BEHAVIOR_CODE = {
    "normal": "NORMAL",
    "running": "RUNNING",
    "falling": "FALLING",
    "wrong_way": "WRONG_WAY",
    "suspect_wrong_way": "SUSPECT_WRONG_WAY",
    "uncertain": "UNCERTAIN",
}

# 风险等级码
_RISK_CODE = {
    "safe": "SAFE",
    "notice": "NOTICE",
    "warning": "WARNING",
    "danger": "DANGER",
}

# 传感器状态码
_SENSOR_CODE = {
    "normal": "AVAILABLE",
    "partial": "DEGRADED",
    "degraded": "DEGRADED",
    "unavailable": "UNAVAILABLE",
}


def build_message(result) -> Dict[str, Any]:
    """
    将 FrameResult 组装为对外 JSON 报文。

    Args:
        result: pipeline.FrameResult

    Returns:
        dict 可直接 json.dumps
    """
    # ---- ego_status ----
    ego_status = {
        "speed_mps": result.ego_speed_mps,
        "speed_kmh": result.ego_speed_kmh,
        "yaw_rate_rps": result.ego_yaw_rate,
    }

    # ---- sensor_availability ----
    sensor_raw = result.sensor_status or "normal"
    sensor_avail = _SENSOR_CODE.get(sensor_raw, "AVAILABLE")
    sensor_info = {
        "status": sensor_avail,
        "quality_score": round(result.sensor_score, 3) if result.sensor_score is not None else 1.0,
        "blind_zones": _serialize_blind_zones(result.blind_zones),
    }

    # ---- vru_objects ----
    # 建立 track_id -> risk_result 映射
    risk_map = {}
    for rr in result.risk_results:
        risk_map[rr.track_id] = rr

    vru_objects: List[Dict[str, Any]] = []
    for obj in result.tracks:
        tid = obj["track_id"]
        cls = obj["cls"]
        behavior = obj.get("behavior", "normal")
        behavior_conf = obj.get("behavior_conf", 0.0)
        bbox = obj.get("bbox", [0, 0, 0, 0])
        confidence = obj.get("conf", 0.0)

        # 运动学信息（从 projected 获取米制坐标）
        kinematics = {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0}
        proj = None
        for p in result.projected:
            if p.track_id == tid:
                proj = p
                kinematics = {
                    "x": round(p.position[0], 3),
                    "y": round(p.position[1], 3),
                    "vx": round(p.velocity[0], 3),
                    "vy": round(p.velocity[1], 3),
                }
                break

        # 风险信息
        risk_info = {"level": "SAFE", "ttc_ms": None, "major_reason": "no_collision_within_horizon"}
        rr = risk_map.get(tid)
        if rr is not None:
            risk_info = {
                "level": _RISK_CODE.get(rr.risk_level, "SAFE"),
                "ttc_ms": rr.ttc_ms,
                "major_reason": rr.major_reason,
                "collision_probability": round(rr.collision_probability, 3),
                "prediction_source": rr.prediction_source,
            }

        # 对外报文输出QCNet最高概率模态；全部模态仍保留给内部TTC使用。
        prediction = {
            "source": None,
            "traj_points": [],
            "horizon_s": 0.0,
            "num_modes": 0,
            "selected_mode": None,
            "selected_probability": None,
        }
        qcnet_pred = (getattr(result, "qcnet_predictions", {}) or {}).get(str(tid))
        if qcnet_pred is not None and qcnet_pred.modes:
            best_mode = max(qcnet_pred.modes, key=lambda mode: mode.probability)
            prediction = {
                "source": qcnet_pred.prediction_source,
                "traj_points": [
                    {
                        "t_s": round(float(point.t_s), 3),
                        "x": round(float(point.x), 3),
                        "y": round(float(point.y), 3),
                    }
                    for point in best_mode.points
                ],
                "horizon_s": round(float(qcnet_pred.prediction_horizon_s), 3),
                "num_modes": len(qcnet_pred.modes),
                "selected_mode": best_mode.mode_id,
                "selected_probability": round(float(best_mode.probability), 4),
            }

        vru_objects.append({
            "object_id": str(tid),
            "class": _class_name(cls),
            "confidence": round(float(confidence), 3),
            "bbox_px": [int(v) for v in bbox],
            "kinematics": kinematics,
            "behavior": {
                "state": _BEHAVIOR_CODE.get(behavior, "NORMAL"),
                "confidence": round(float(behavior_conf), 3),
            },
            "prediction": prediction,
            "risk": risk_info,
        })

    # ---- performance ----
    performance = {
        "total_delay_ms": round(result.process_ms, 1),
        "fps": round(result.fps, 1) if result.fps else 0.0,
        "qcnet_ready": result.qcnet_ready,
        "qcnet_loaded": bool(getattr(result, "qcnet_loaded", False)),
        "qcnet_active": bool(getattr(result, "qcnet_active", False)),
        "qcnet_predicted_targets": len(
            getattr(result, "qcnet_predictions", {}) or {}
        ),
        "qcnet_history_frames": result.qcnet_history_frames,
    }

    return {
        "timestamp": result.timestamp,
        "frame_id": result.frame_id,
        "ego_status": ego_status,
        "sensor_availability": sensor_info,
        "vru_objects": vru_objects,
        "performance": performance,
    }


def _class_name(cls_id: int) -> str:
    """COCO class id -> 英文类名。"""
    names = {
        0: "pedestrian",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }
    return names.get(cls_id, "unknown")


def _serialize_blind_zones(zones) -> list:
    """将 BlindZoneInfo 对象转为可 JSON 序列化的 dict。"""
    result = []
    for z in (zones or []):
        if hasattr(z, "__dict__"):
            result.append({k: v for k, v in vars(z).items() if not k.startswith("_")})
        elif isinstance(z, dict):
            result.append(z)
        else:
            result.append(str(z))
    return result


def format_message(msg: Dict[str, Any], indent: Optional[int] = None) -> str:
    """序列化为 JSON 字符串。"""
    return json.dumps(msg, ensure_ascii=False, indent=indent)


class MessageLogger:
    """
    报文输出器：支持写文件和/或控制台打印。

    用法:
        logger = MessageLogger(output_path="output/messages.jsonl")
        logger.log(result)
    """

    def __init__(self, output_path: Optional[str] = None, console: bool = False):
        self.output_path = output_path
        self.console = console
        self._fh = None
        if output_path:
            import os
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            self._fh = open(output_path, "w", encoding="utf-8")

    def log(self, result) -> Dict[str, Any]:
        """组装报文并输出，返回报文 dict。"""
        msg = build_message(result)
        line = format_message(msg)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()
        if self.console:
            print(line)
        return msg

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

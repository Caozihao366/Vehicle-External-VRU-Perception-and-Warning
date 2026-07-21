"""nuScenes 数据集读取器。

本文件负责把 nuScenes SDK 和元数据转换为同步管线需要的标准场景字典。
它是数据集相关代码；sync_pipeline.py 不会导入或读取 nuScenes。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Optional


class NuScenesDatasetSource:
    """
    nuScenes 数据集读取器。

    ``iter_scenes()`` 逐场返回统一格式：

    {
      "name": "scene_name",
      "streams": {
        "front": [{"path": "...", "timestamp_us": 123, "meta": {...}}],
        "left":  [...],
        "right": [...],
        "rear":  [...]
      },
      "ego_poses": [  # 可省略
        {"timestamp_us": 123, "translation": [x, y, z],
         "rotation": [w, x, y, z]}
      ]
    }

    换数据集时，新数据集应在自己的文件中提供相同的 ``iter_scenes()``
    输出；不要修改 sync_pipeline.py。
    """

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-mini",
        camera_map: Optional[Mapping[str, str]] = None,
    ) -> None:
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as exc:
            raise ImportError(
                "NuScenesDatasetSource 需要安装 nuscenes-devkit"
            ) from exc

        self.channels = dict(
            camera_map
            or {
                "front": "CAM_FRONT",
                "left": "CAM_FRONT_LEFT",
                "right": "CAM_FRONT_RIGHT",
                "rear": "CAM_BACK",
            }
        )
        self._nusc = NuScenes(
            version=version, dataroot=str(Path(dataroot)), verbose=False
        )

    def iter_scenes(self) -> Iterator[dict[str, Any]]:
        for scene in self._nusc.scene:
            yield self._read_scene(scene)

    def _read_scene(self, scene: Mapping[str, Any]) -> dict[str, Any]:
        first_sample = self._nusc.get("sample", scene["first_sample_token"])
        streams: dict[str, list[dict[str, Any]]] = {}

        for camera, channel in self.channels.items():
            token = first_sample["data"][channel]
            while self._nusc.get("sample_data", token)["prev"]:
                token = self._nusc.get("sample_data", token)["prev"]

            frames: list[dict[str, Any]] = []
            while token:
                sample_data = self._nusc.get("sample_data", token)
                frames.append(
                    {
                        "path": sample_data["filename"],
                        "timestamp_us": int(sample_data["timestamp"]),
                        "token": sample_data["token"],
                        "meta": {
                            "is_key_frame": bool(sample_data["is_key_frame"])
                        },
                    }
                )
                token = sample_data["next"]
            streams[camera] = frames

        token = first_sample["data"]["LIDAR_TOP"]
        while self._nusc.get("sample_data", token)["prev"]:
            token = self._nusc.get("sample_data", token)["prev"]

        poses: list[dict[str, Any]] = []
        seen = set()
        while token:
            sample_data = self._nusc.get("sample_data", token)
            pose = self._nusc.get("ego_pose", sample_data["ego_pose_token"])
            if pose["token"] not in seen:
                seen.add(pose["token"])
                poses.append(
                    {
                        "timestamp_us": int(pose["timestamp"]),
                        "translation": pose["translation"],
                        "rotation": pose["rotation"],
                    }
                )
            token = sample_data["next"]

        return {"name": scene["name"], "streams": streams, "ego_poses": poses}

"""使用 nuScenes 数据集运行时间对齐。"""

from pathlib import Path

from dataset_source import NuScenesDatasetSource
from sync_pipeline import SyncPipeline


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    SyncPipeline(
        source=NuScenesDatasetSource(
            dataroot=str(root / "data"),
            version="v1.0-mini",
        ),
        output_json=str(Path(__file__).with_name("fr01_sync_output.json")),
        sync_tolerance_ms=50.0,
        ref_mode="interval",
        ref_interval_ms=5.0,
        require_ego=True,
    ).run()

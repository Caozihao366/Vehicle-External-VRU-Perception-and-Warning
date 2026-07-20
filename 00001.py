if __name__ == "__main__":
    demo_temporal_aligner(tolerance_ms=40.0)
    # keyframe≈0.5s一帧；interval=100ms → 每个标注间隔内约 5 组；all≈12Hz全量
    SyncPipeline(
        sync_tolerance_ms=50.0,
        ref_mode="interval",
        ref_interval_ms=5.0,
    ).run()
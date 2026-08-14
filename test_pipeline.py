"""快速测试脚本：处理 3 帧，不显示 UI。"""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print('Initializing pipeline with CAM_FRONT images...')
from pipeline import Pipeline
pipe = Pipeline(
    video_source='input/CAM_FRONT',
    enable_sensor_quality=False,
    enable_wrong_way=False,
)
print('Processing 3 frames...')
for i, result in enumerate(pipe):
    print(f'  Frame {result.frame_id}: {len(result.tracks)} tracks, '
          f'{len(result.projected)} projected, '
          f'{len(result.risk_results)} risk, '
          f'{result.process_ms:.0f}ms')
    for t in result.tracks[:3]:
        print(f'    id={t["track_id"]} cls={t["cls"]} '
              f'behavior={t.get("behavior", "?")} bbox={t["bbox"]}')
    for p in result.projected[:3]:
        print(f'    projected id={p.track_id} '
              f'pos=({p.position[0]:.1f},{p.position[1]:.1f}) '
              f'vel=({p.velocity[0]:.1f},{p.velocity[1]:.1f})')
    for r in result.risk_results[:3]:
        print(f'    risk id={r.track_id} level={r.risk_level} '
              f'ttc={r.ttc_s} reason={r.major_reason}')
    if i >= 2:
        break
pipe.release()
print('Test passed!')

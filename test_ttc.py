"""测试 TTC：处理多帧，设置自车速度，观察风险变化。"""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
print('Initializing with ego speed 15 m/s...')
from pipeline import Pipeline
from ui import UnifiedUI

pipe = Pipeline(
    video_source='input/CAM_FRONT',
    ego_speed_mps=15.0,  # 54 km/h
    enable_sensor_quality=True,
    enable_wrong_way=False,
)
ui = UnifiedUI()

print('Processing 15 frames...')
for i, result in enumerate(pipe):
    n_warn = sum(1 for r in result.risk_results if r.risk_level != 'safe')
    if n_warn > 0 or i % 5 == 0:
        print(f'  Frame {result.frame_id}: {len(result.tracks)} tracks, '
              f'{n_warn} risk targets, {result.process_ms:.0f}ms')
        for r in result.risk_results:
            if r.risk_level != 'safe':
                print(f'    id={r.track_id} level={r.risk_level} '
                      f'ttc={r.ttc_ms}ms reason={r.major_reason} '
                      f'pos=({r.position[0]:.1f},{r.position[1]:.1f})')
    if i == 14:
        # 保存最后一帧
        canvas = ui.render(result)
        cv2.imwrite('test_ttc.png', canvas)
        print('Saved test_ttc.png')
        break
pipe.release()
print('TTC test done!')

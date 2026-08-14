"""测试 UI 渲染：处理 1 帧并保存为图片。"""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
print('Initializing...')
from pipeline import Pipeline
from ui import UnifiedUI

pipe = Pipeline(
    video_source='input/CAM_FRONT',
    enable_sensor_quality=True,
    enable_wrong_way=True,
    enable_yolopv2=False,  # 先不跑车道线，加快测试
)
ui = UnifiedUI()

print('Processing 1 frame...')
for i, result in enumerate(pipe):
    canvas = ui.render(result)
    out_path = 'test_ui_output.png'
    cv2.imwrite(out_path, canvas)
    print(f'Saved {out_path}, shape={canvas.shape}')
    print(f'  tracks={len(result.tracks)}, projected={len(result.projected)}, '
          f'risk={len(result.risk_results)}, sensor={result.sensor_status}')
    if i >= 0:
        break
pipe.release()
print('UI test passed!')

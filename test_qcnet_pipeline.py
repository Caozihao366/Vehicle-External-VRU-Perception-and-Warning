"""测试 QCNet 集成：处理足够帧数以触发多模态 TTC。"""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
print('Initializing pipeline with QCNet...')
from pipeline import Pipeline

pipe = Pipeline(
    video_source='input/CAM_FRONT',
    ego_speed_mps=10.0,
    enable_sensor_quality=False,
    enable_wrong_way=False,
)

print('Processing frames until QCNet activates...')
for i, result in enumerate(pipe):
    sources = set(r.prediction_source for r in result.risk_results)
    n_qcnet = sum(1 for r in result.risk_results if r.prediction_source != 'constant_velocity')
    print(f'  Frame {result.frame_id}: {len(result.tracks)} tracks, '
          f'history={result.qcnet_history_frames}, ready={result.qcnet_ready}, '
          f'QCNet preds={n_qcnet}, sources={sources}, '
          f'{result.process_ms:.0f}ms')
    if n_qcnet > 0:
        for r in result.risk_results:
            if r.prediction_source != 'constant_velocity':
                print(f'    id={r.track_id} level={r.risk_level} '
                      f'ttc={r.ttc_ms}ms coll_prob={r.collision_probability:.2f} '
                      f'source={r.prediction_source}')
        # 保存一帧
        from ui import UnifiedUI
        ui = UnifiedUI()
        canvas = ui.render(result)
        cv2.imwrite('test_qcnet_integration.png', canvas)
        print('Saved test_qcnet_integration.png')
        break
    if i >= 60:
        print('QCNet did not activate after 60 frames')
        break
pipe.release()
print('QCNet integration test done!')

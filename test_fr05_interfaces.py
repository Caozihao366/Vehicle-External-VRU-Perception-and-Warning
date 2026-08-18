"""FR-05 上下游接口契约测试（不加载模型权重）。"""

import unittest
from types import SimpleNamespace

from fr05.interfaces import FR05Event, FR05Result, validate_track
from message_builder import build_message
from ttc.risk import classify_risk


class FR05InterfaceTests(unittest.TestCase):
    def test_validate_track_contract(self):
        valid = {
            "track_id": 7,
            "cls": 0,
            "bbox": (10, 20, 30, 80),
            "bottom_center": (20, 80),
            "bbox_h": 60,
            "speed_norm": 0.3,
        }
        self.assertEqual(validate_track(valid), [])

        invalid = dict(valid)
        invalid.pop("speed_norm")
        invalid["bbox"] = (1, 2, 3)
        errors = validate_track(invalid)
        self.assertIn("speed_norm", errors)
        self.assertIn("bbox must contain four coordinates", errors)

    def test_result_and_event_are_serializable(self):
        event = FR05Event(
            frame_id=12,
            timestamp=123.5,
            object_id=7,
            class_id=0,
            state="running",
            confidence=0.88,
            source="motion_pose",
            bbox_px=(10, 20, 30, 80),
        )
        result = FR05Result(
            frame_id=12,
            timestamp=123.5,
            tracks=[],
            events=[event],
            status="available",
            latency_ms=2.3456,
            capabilities={"running_detection": True},
        )
        self.assertEqual(event.to_dict()["state"], "running")
        self.assertEqual(result.to_status_dict()["event_count"], 1)
        self.assertEqual(result.to_status_dict()["latency_ms"], 2.346)

    def test_suspect_wrong_way_is_accepted_by_ttc(self):
        level, reason = classify_risk(
            earliest_ttc_s=2.0,
            collision_probability=0.5,
            behavior_state="suspect_wrong_way",
        )
        self.assertEqual(level, "warning")
        self.assertEqual(reason, "suspect_wrong_way_trajectory_conflict")

    def test_message_contains_fr05_status_and_events(self):
        event = FR05Event(
            frame_id=1,
            timestamp=1.0,
            object_id=3,
            class_id=2,
            state="wrong_way",
            confidence=0.9,
            source="wrong_way_fusion",
            bbox_px=(1, 2, 3, 4),
        )
        frame_result = SimpleNamespace(
            timestamp=1.0,
            frame_id=1,
            ego_speed_mps=0.0,
            ego_speed_kmh=0.0,
            ego_yaw_rate=0.0,
            sensor_status="normal",
            sensor_score=1.0,
            blind_zones=[],
            risk_results=[],
            tracks=[],
            projected=[],
            process_ms=4.0,
            fps=25.0,
            qcnet_ready=False,
            qcnet_history_frames=0,
            fr05_status="available",
            fr05_capabilities={"wrong_way_detection": True},
            fr05_warnings=[],
            fr05_events=[event],
            fr05_latency_ms=1.2,
        )
        message = build_message(frame_result)
        self.assertEqual(message["fr05"]["status"], "available")
        self.assertEqual(message["fr05"]["events"][0]["state"], "wrong_way")
        self.assertEqual(message["performance"]["fr05_latency_ms"], 1.2)


if __name__ == "__main__":
    unittest.main()

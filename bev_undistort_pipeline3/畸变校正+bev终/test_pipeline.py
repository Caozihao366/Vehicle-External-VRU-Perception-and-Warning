import unittest
import numpy as np
from bev import Calibration, with_intrinsics, BEVProjector, load_calibrations, synchronized_frames, CAMERAS
from undistortion import load_distortions
from detect import backproject_to_ground


class PipelineTests(unittest.TestCase):
    def test_intrinsic_replacement_preserves_extrinsics(self):
        old = Calibration(np.eye(3), np.eye(3), np.array([0., 0., 2.]))
        new_k = np.diag([2., 2., 1.])
        updated = with_intrinsics({"cam": old}, {"cam": new_k})["cam"]
        np.testing.assert_array_equal(updated.intrinsic, new_k)
        np.testing.assert_array_equal(updated.rotation, old.rotation)
        np.testing.assert_array_equal(updated.translation, old.translation)

    def test_backprojection_rejects_ray_away_from_ground(self):
        calib = Calibration(np.eye(3), np.eye(3), np.array([0., 0., 2.]))
        self.assertIsNone(backproject_to_ground(calib, 0., 0.))

    def test_ground_pixel_center_and_bounds(self):
        p = BEVProjector({}, -10., 30., -15., 15., .05)
        col, row = p.ground_to_pixel(30., 15.)
        self.assertEqual((col, row), (0, 0))

    def test_a2d2_calibration(self):
        path = __import__("pathlib").Path(r"D:\test2\a2d2-preview\cams_lidars.json")
        if not path.exists():
            self.skipTest("A2D2 preview is not installed")
        calibrations = load_calibrations(path)
        distortions = load_distortions(path)
        self.assertEqual(set(calibrations), set(CAMERAS))
        self.assertEqual(set(distortions), set(CAMERAS))
        np.testing.assert_allclose(calibrations["front_center"].rotation.T @ calibrations["front_center"].rotation, np.eye(3), atol=1e-7)

    def test_a2d2_synchronization(self):
        root = __import__("pathlib").Path(r"D:\test2\a2d2-preview")
        if not root.exists():
            self.skipTest("A2D2 preview is not installed")
        scene, frame_id, paths = next(synchronized_frames(root, "20180810_150607"))
        self.assertEqual(scene, "20180810_150607")
        self.assertEqual(set(paths), set(CAMERAS))
        self.assertTrue(all(path.suffix == ".png" for path in paths.values()))


if __name__ == "__main__": unittest.main()

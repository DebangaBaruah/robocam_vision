"""
robot_vision_system.py
Orchestrates all detectors into a single process_frame() call.
"""

import time
from utils.detectors import ObjectDetector, FireDetector, SmokeDetector, WaterFallingDetector


class RobotVisionSystem:
    def __init__(self, model_cfg_path, model_weights_path, classes_file_path):
        print("[RobotVisionSystem] Initializing detectors…")
        self.object_detector       = ObjectDetector(model_cfg_path, model_weights_path, classes_file_path)
        self.fire_detector         = FireDetector()
        self.smoke_detector        = SmokeDetector()
        self.water_falling_detector = WaterFallingDetector()
        print("[RobotVisionSystem] Ready.")

    def reset_state(self):
        """Call before processing a new video to clear inter-frame state."""
        self.smoke_detector.reset()
        self.water_falling_detector.reset()

    def process_frame(self, frame,
                      run_objects=True, run_fire=True,
                      run_smoke=True, run_water=True):
        output = frame.copy()
        results = {}

        # 1. General objects
        if run_objects:
            t0 = time.time()
            detections = self.object_detector.detect(output)
            output = self.object_detector.draw(output, detections)
            results["objects"] = detections
            results["object_detection_time"] = round(time.time() - t0, 4)

        # 2. Fire
        if run_fire:
            t0 = time.time()
            fire_detected, fire_boxes = self.fire_detector.detect(output)
            output = self.fire_detector.draw(output, fire_detected, fire_boxes)
            results["fire_detected"] = fire_detected
            results["fire_detection_time"] = round(time.time() - t0, 4)

        # 3. Smoke
        if run_smoke:
            t0 = time.time()
            smoke_detected, smoke_boxes = self.smoke_detector.detect(output)
            output = self.smoke_detector.draw(output, smoke_detected, smoke_boxes)
            results["smoke_detected"] = smoke_detected
            results["smoke_detection_time"] = round(time.time() - t0, 4)

        # 4. Water falling
        if run_water:
            t0 = time.time()
            water_detected, water_boxes = self.water_falling_detector.detect(output)
            output = self.water_falling_detector.draw(output, water_detected, water_boxes)
            results["water_detected"] = water_detected
            results["water_detection_time"] = round(time.time() - t0, 4)

        return output, results

"""
detectors.py
All four detection classes extracted and cleaned from the original notebook.
"""

import cv2
import numpy as np
import os


# ── ObjectDetector ─────────────────────────────────────────────────────────
class ObjectDetector:
    """YOLOv3-based general object detector (80 COCO classes)."""

    def __init__(self, model_cfg_path, model_weights_path, classes_file_path,
                 confidence_threshold=0.5, nms_threshold=0.4):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        with open(classes_file_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.net = cv2.dnn.readNet(model_weights_path, model_cfg_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.output_layers = self.net.getUnconnectedOutLayersNames()

        # Assign a unique color to each class for prettier boxes
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype=np.uint8)

    def detect(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                objectness = float(detection[4])
                confidence = float(scores[class_id] * objectness)
                if confidence > self.confidence_threshold:
                    cx = int(detection[0] * width)
                    cy = int(detection[1] * height)
                    w  = int(detection[2] * width)
                    h  = int(detection[3] * height)
                    x  = int(cx - w / 2)
                    y  = int(cy - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        detections = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]]
                conf  = round(confidences[i], 2)
                color = [int(c) for c in self.colors[class_ids[i]]]
                detections.append({"label": label, "confidence": conf,
                                   "box": (x, y, w, h), "color": color})
        return detections

    def draw(self, frame, detections):
        height, width = frame.shape[:2]
        thickness = max(2, min(width, height) // 250)
        font_scale = 0.7
        font = cv2.FONT_HERSHEY_SIMPLEX

        for det in detections:
            x, y, w, h = det["box"]
            color = det["color"]
            label = f"{det['label']} {det['confidence']:.0%}"

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x + w)
            y2 = min(height - 1, y + h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            label_x1 = x1
            label_y1 = y1 - th - 10
            label_y2 = y1
            if label_y1 < 0:
                label_y1 = y1
                label_y2 = y1 + th + 10

            cv2.rectangle(frame, (label_x1, label_y1), (label_x1 + tw + 10, label_y2), color, -1)
            text_y = label_y2 - 5 if label_y1 < y1 else label_y2 - 5
            cv2.putText(frame, label, (label_x1 + 5, text_y),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return frame


# ── FireDetector ───────────────────────────────────────────────────────────
class FireDetector:
    """HSV color-range fire detector (red + orange hues)."""

    FIRE_COLOR = (0, 0, 255)   # red box

    def __init__(self, min_contour_area=500):
        self.lower_red1   = np.array([0,   120,  70])
        self.upper_red1   = np.array([10,  255, 255])
        self.lower_red2   = np.array([170, 120,  70])
        self.upper_red2   = np.array([180, 255, 255])
        self.lower_orange = np.array([10,  100, 100])
        self.upper_orange = np.array([25,  255, 255])
        self.min_contour_area = min_contour_area
        self.min_area_ratio = 0.001
        self._kernel = np.ones((5, 5), np.uint8)

    def detect(self, frame):
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        omask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        fire_mask = cv2.bitwise_or(mask1 + mask2, omask)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN,  self._kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected = False
        boxes = []
        frame_area = frame.shape[0] * frame.shape[1]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_contour_area and area / frame_area >= self.min_area_ratio:
                detected = True
                boxes.append(cv2.boundingRect(cnt))
        return detected, boxes

    def draw(self, frame, detected, boxes):
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.FIRE_COLOR, 2)
            _draw_label(frame, "🔥 Fire!", x, y, self.FIRE_COLOR)
        if detected:
            cv2.putText(frame, "FIRE DETECTED", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.FIRE_COLOR, 2, cv2.LINE_AA)
        return frame


# ── SmokeDetector ──────────────────────────────────────────────────────────
class SmokeDetector:
    """Motion + color heuristic smoke detector."""

    SMOKE_COLOR = (0, 220, 220)   # yellow-ish

    def __init__(self, motion_threshold=30, min_contour_area=1000,
                 avg_color_threshold=180, std_color_threshold=45, sat_threshold=80):
        self.motion_threshold     = motion_threshold
        self.min_contour_area     = min_contour_area
        self.avg_color_threshold  = avg_color_threshold
        self.std_color_threshold  = std_color_threshold
        self.sat_threshold        = sat_threshold
        self._kernel = np.ones((5, 5), np.uint8)
        self.prev_gray = None

    def reset(self):
        self.prev_gray = None

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = False
        boxes = []

        if self.prev_gray is not None:
            diff = cv2.absdiff(self.prev_gray, gray)
            _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, self._kernel, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) > self.min_contour_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    roi = frame[y:y + h, x:x + w]
                    if roi.size > 0:
                        avg = np.mean(roi)
                        std = np.std(roi)
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        sat = np.mean(hsv_roi[:, :, 1])
                        if avg < self.avg_color_threshold and std < self.std_color_threshold and sat < self.sat_threshold:
                            detected = True
                            boxes.append((x, y, w, h))

        self.prev_gray = gray
        return detected, boxes

    def draw(self, frame, detected, boxes):
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.SMOKE_COLOR, 2)
            _draw_label(frame, "💨 Smoke", x, y, self.SMOKE_COLOR)
        if detected:
            cv2.putText(frame, "SMOKE DETECTED", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.SMOKE_COLOR, 2, cv2.LINE_AA)
        return frame


# ── WaterFallingDetector ───────────────────────────────────────────────────
class WaterFallingDetector:
    """MOG2 background subtraction + aspect ratio heuristic water detector."""

    WATER_COLOR = (255, 100, 0)   # blue

    def __init__(self, min_contour_area=1200, aspect_ratio_h_w=2.8, min_height=70):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True)
        self.min_contour_area   = min_contour_area
        self.aspect_ratio_h_w   = aspect_ratio_h_w
        self.min_height         = min_height
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def reset(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True)

    def detect(self, frame):
        fgmask = self.fgbg.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self._kernel)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected = False
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > self.aspect_ratio_h_w * w and h > self.min_height:
                    roi = frame[y:y + h, x:x + w]
                    if roi.size > 0:
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        avg_sat = np.mean(hsv_roi[:, :, 1])
                        avg_val = np.mean(hsv_roi[:, :, 2])
                        if avg_sat < 95 and avg_val > 90:
                            detected = True
                            boxes.append((x, y, w, h))
        return detected, boxes

    def draw(self, frame, detected, boxes):
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.WATER_COLOR, 2)
            _draw_label(frame, "💧 Water", x, y, self.WATER_COLOR)
        if detected:
            cv2.putText(frame, "WATER DETECTED", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.WATER_COLOR, 2, cv2.LINE_AA)
        return frame


# ── Shared helper ──────────────────────────────────────────────────────────
def _draw_label(frame, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    by = max(y - 6, th + 6)
    cv2.rectangle(frame, (x, by - th - 4), (x + tw + 4, by + 2), color, -1)
    cv2.putText(frame, text, (x + 2, by - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

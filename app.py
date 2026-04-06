import gradio as gr
import cv2
import numpy as np
import os
import time
import tempfile
import sys

from utils.detectors import ObjectDetector, FireDetector, SmokeDetector, WaterFallingDetector
from utils.robot_vision_system import RobotVisionSystem
from utils.model_downloader import ensure_models_downloaded

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_CFG = os.path.join(MODEL_DIR, "yolov3.cfg")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "yolov3.weights")
CLASSES_FILE = os.path.join(MODEL_DIR, "coco.names")

# Global robot vision system (lazy-loaded after model download)
_robot_vision_system = None
_init_error = None

def get_robot_vision_system():
    """Load the robot vision system with better error handling."""
    global _robot_vision_system, _init_error
    if _robot_vision_system is None and _init_error is None:
        try:
            print("[App] Initializing models...", file=sys.stderr)
            ok = ensure_models_downloaded(MODEL_DIR, MODEL_CFG, MODEL_WEIGHTS, CLASSES_FILE)
            if not ok:
                raise RuntimeError("Model files could not be downloaded.")
            print("[App] Models downloaded successfully", file=sys.stderr)
            _robot_vision_system = RobotVisionSystem(MODEL_CFG, MODEL_WEIGHTS, CLASSES_FILE)
            print("[App] System ready!", file=sys.stderr)
        except Exception as e:
            _init_error = str(e)
            print(f"[App] Initialization error: {e}", file=sys.stderr)
            # Don't re-raise for Spaces - let the app start with error handling
            return None
    return _robot_vision_system


def process_video(input_video_path, confidence_threshold=0.5, skip_frames=2,
                  enable_objects=True, enable_fire=True, enable_smoke=True, enable_water=True,
                  progress=gr.Progress(track_tqdm=True)):
    """Core Gradio processing function — runs RobotVisionSystem on uploaded video."""
    if input_video_path is None:
        return None, "⚠️ Please upload a video first."

    try:
        system = get_robot_vision_system()
        if system is None:
            return None, f"❌ System initialization failed: {_init_error}"
    except RuntimeError as e:
        return None, f"❌ Model load error: {e}"

    # Update per-call confidence threshold
    system.object_detector.confidence_threshold = confidence_threshold

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return None, "❌ Could not open video file."

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output path
    out_path = os.path.join(tempfile.gettempdir(), f"robot_vision_output_{int(time.time())}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

    frame_idx = 0
    stats = {"objects": 0, "fire_frames": 0, "smoke_frames": 0, "water_frames": 0}

    system.reset_state()  # Reset inter-frame state for fresh video

    progress(0, desc="Processing video…")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames for speed, but always write original for skipped frames
        if frame_idx % (skip_frames + 1) != 0:
            writer.write(frame)
            continue

        annotated, results = system.process_frame(
            frame,
            run_objects=enable_objects,
            run_fire=enable_fire,
            run_smoke=enable_smoke,
            run_water=enable_water,
        )

        # Accumulate stats
        stats["objects"] += len(results.get("objects", []))
        if results.get("fire_detected"):   stats["fire_frames"]  += 1
        if results.get("smoke_detected"):  stats["smoke_frames"] += 1
        if results.get("water_detected"):  stats["water_frames"] += 1

        writer.write(annotated)

        if total_frames > 0:
            progress(frame_idx / total_frames, desc=f"Frame {frame_idx}/{total_frames}")

    cap.release()
    writer.release()

    summary = (
        f"✅ **Done!** Processed {frame_idx} frames\n\n"
        f"- 🟢 Objects detected: {stats['objects']} total instances\n"
        f"- 🔴 Fire frames: {stats['fire_frames']}\n"
        f"- 🟡 Smoke frames: {stats['smoke_frames']}\n"
        f"- 🔵 Water falling frames: {stats['water_frames']}"
    )
    return out_path, summary


def process_webcam_frame(frame, confidence_threshold=0.5,
                        enable_objects=True, enable_fire=True, enable_smoke=True, enable_water=True):
    """Process a single webcam frame in real-time."""
    if frame is None:
        return frame

    try:
        system = get_robot_vision_system()
        if system is None:
            return frame  # Return original frame if system not initialized
    except RuntimeError:
        return frame  # Return original frame if model not loaded

    # Update confidence threshold
    system.object_detector.confidence_threshold = confidence_threshold

    # Process the frame
    annotated, results = system.process_frame(
        frame,
        run_objects=enable_objects,
        run_fire=enable_fire,
        run_smoke=enable_smoke,
        run_water=enable_water,
    )

    return annotated

# ── Gradio UI ──────────────────────────────────────────────────────────────
with gr.Blocks(
    title="🤖 Robot Vision System"
) as demo:

    gr.Markdown(
        """
        <div class="title-block">
        <h1>🤖 Robot Vision System</h1>
        <p>Real-time object detection for <strong>objects</strong>, <strong>fire</strong>,
        <strong>smoke</strong>, and <strong>water falling</strong> with live webcam or video upload.</p>
        <p style="color: gray; font-size: 13px;">Powered by YOLOv3 + OpenCV · Open-source · No data stored</p>
        </div>
        """
    )

    with gr.Tabs():
        with gr.TabItem("📹 Live Webcam Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    webcam_input = gr.Image(sources=["webcam"], streaming=True, label="📷 Webcam Feed")

                    with gr.Accordion("⚙️ Detection Settings", open=True):
                        webcam_confidence = gr.Slider(
                            minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                            label="Confidence Threshold",
                            info="Lower = more object detections, higher = cleaner results"
                        )

                    with gr.Accordion("🔍 Enable / Disable Detectors", open=True):
                        webcam_objects = gr.Checkbox(value=True, label="🟢 General Objects (YOLO / COCO 80 classes)")
                        webcam_fire    = gr.Checkbox(value=True, label="🔴 Fire Detection")
                        webcam_smoke   = gr.Checkbox(value=True, label="🟡 Smoke Detection")
                        webcam_water   = gr.Checkbox(value=True, label="🔵 Water Falling Detection")

                with gr.Column(scale=1):
                    webcam_output = gr.Image(label="📥 Live Detection Results", streaming=True)
                    webcam_status = gr.Markdown("Point your webcam and watch real-time detection!")

            webcam_input.stream(
                fn=process_webcam_frame,
                inputs=[webcam_input, webcam_confidence, webcam_objects, webcam_fire, webcam_smoke, webcam_water],
                outputs=[webcam_output]
            )

        with gr.TabItem("📤 Video Upload Processing"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_video = gr.Video(label="📤 Upload Video", sources=["upload"])

                    with gr.Accordion("⚙️ Detection Settings", open=True):
                        confidence_slider = gr.Slider(
                            minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                            label="Confidence Threshold",
                            info="Lower = more object detections, higher = cleaner results"
                        )
                        skip_frames_slider = gr.Slider(
                            minimum=0, maximum=5, value=2, step=1,
                            label="Skip Frames (speed vs quality)",
                            info="0 = process every frame (slow). 2-3 recommended."
                        )

                    with gr.Accordion("🔍 Enable / Disable Detectors", open=True):
                        enable_objects = gr.Checkbox(value=True, label="🟢 General Objects (YOLO / COCO 80 classes)")
                        enable_fire    = gr.Checkbox(value=True, label="🔴 Fire Detection")
                        enable_smoke   = gr.Checkbox(value=True, label="🟡 Smoke Detection")
                        enable_water   = gr.Checkbox(value=True, label="🔵 Water Falling Detection")

                    run_btn = gr.Button("▶ Run Detection", variant="primary", size="lg")

                with gr.Column(scale=1):
                    output_video = gr.Video(label="📥 Annotated Output Video")
                    result_text  = gr.Markdown("Results will appear here after processing.")

            run_btn.click(
                fn=process_video,
                inputs=[input_video, confidence_slider, skip_frames_slider,
                        enable_objects, enable_fire, enable_smoke, enable_water],
                outputs=[output_video, result_text],
            )
    gr.Markdown(
        """
        ---
        ### 📦 What gets detected?
        | Detector | Method | Bounding Box Color |
        |---|---|---|
        | General Objects (80 COCO classes) | YOLOv3 deep learning | 🟢 Green |
        | Fire | HSV color segmentation (red/orange) | 🔴 Red |
        | Smoke | Frame-diff motion + gray heuristic | 🟡 Yellow |
        | Water Falling | MOG2 background subtraction + aspect ratio | 🔵 Blue |

        > **Note:** First run downloads YOLOv3 weights (~237 MB). Subsequent runs use the cached model.
        """
    )


if __name__ == "__main__":
    demo.launch(share=False, show_error=True)

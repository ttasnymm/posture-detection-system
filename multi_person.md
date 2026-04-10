README – multi_pose_live_yolo.py

Overview

This script runs a multi-person live pose and posture monitor using:
	•	YOLOv8-pose for person detection + 2D keypoints
	•	A simple HF (head-forward) + SA (shoulder asymmetry) rule per person

For each detected person it:
	•	Tracks the person with a YOLO bounding box
	•	Extracts nose + shoulders keypoints from the pose skeleton
	•	Computes HF and SA per person and applies EMA smoothing + dwell time
	•	Draws:
	•	Upper-body skeleton (shoulders + nose, etc.)
	•	A green or red box per person
	•	A small label: id:<N> OK or id:<N> POOR with HF/SA values

Requirements
	•	macOS with webcam
	•	Python 3.9
	•	YOLOv8-pose weights file yolov8n-pose.pt in the same folder
	•	Packages (inside a virtualenv):
	•	ultralytics
	•	opencv-python
	•	numpy

Setup (first time)
    cd <project_folder>         #folder containing multi_pose_live_yolo.py
    python3 -m venv .venv
    source .venv/bin/activate

    python3 -m pip install --upgrade pip
    python3 -m pip install ultralytics opencv-python numpy

Make sure the YOLO pose model is present:
	•	File name: yolov8n-pose.pt
	•	Location: same folder as multi_pose_live_yolo.py

How to Run
    cd <project_folder>
    source .venv/bin/activate
    python3 multi_pose_live_yolo.py

Then,
    •   Allow camera access when macOS asks.
	•	A window opens showing:
	•	One bounding box + skeleton per detected person
	•	Per-person label at the top (ID + OK/POOR + HF/SA)
	•	Posture is marked POOR if smoothed HF or SA are over their thresholds for longer than the dwell time.
	•	Press q to quit.

Notes
	•	Intended for small groups in view of one camera (e.g. classroom row).
	•	Performance depends on your Mac; more people and higher resolution = slower FPS.
	•	Again, processing is entirely local; the script only uses the webcam stream.

README – single_pose_mediapipe_live.py

Overview

This script runs a single-person live posture monitor using your laptop’s webcam and MediaPipe Pose.
It:
	•	Detects your upper body (nose + both shoulders) in real time
	•	Computes:
	•	HF (Head-Forward) angle – how far your head leans forwards relative to vertical
	•	SA (Shoulder Asymmetry) – vertical difference between left and right shoulder, normalised by shoulder width
	•	Applies exponential smoothing and a dwell time (~0.8 s) so posture has to stay poor briefly before raising an alert
	•	Overlays:
	•	Smoothed HF and SA values
	•	A “POOR POSTURE” warning + red border when thresholds are exceeded

Requirements
	•	macOS with working webcam
	•	Python 3.9
	•	Packages (inside a virtualenv):
	•	mediapipe
	•	opencv-python
	•	numpy

Setup (first time)
    cd <project_folder>     #folder containing single_pose_mediapipe_live.py
    python3 -m venv .venv
    source .venv/bin/activate

    python3 -m pip install --upgrade pip
    python3 -m pip install mediapipe opencv-python numpy

How to Run
    cd <project_folder>
    ource .venv/bin/activate
    python3 single_pose_mediapipe_live.py

then,
	•	macOS will prompt for camera access the first time – click Allow.
	•	A window titled e.g. “Posture Monitor” opens.
	•	Top-left text shows HF, SA and “poor frames” counter.
	•	When HF or SA stay over the thresholds for long enough, you see “POOR POSTURE” and a red border.
	•	Press q to quit.

Notes
	•	Designed for one person centred in the frame, at roughly arm’s-length distance.
	•	Works best with the webcam at eye level and reasonable lighting.
	•	All processing is local; no video is saved by default.

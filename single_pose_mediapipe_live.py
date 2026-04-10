import cv2
import numpy as np
import mediapipe as mp

#configuration
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

#smoothing (0 for very slow, 1 for no smoothing)
EMA_ALPHA = 0.4

#thresholds for poor posture
HF_THRESHOLD_DEG = 18.0   # head-forward angle
SA_THRESHOLD = 0.06       # normalised shoulder asymmetry

#for how much long posture must be poor before alert 
DWELL_SECONDS = 0.8
TARGET_FPS = 15  #used to convert seconds to frames


class EmaSmoother:
    """Simple exponential moving average for one value."""
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


def get_landmark(landmarks, idx, image_shape):
    """Convert a MediaPipe landmark to (x, y) pixel coordinates + visibility."""
    h, w, _ = image_shape
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32), lm.visibility


def compute_posture_metrics(landmarks, image_shape):
    """
    Compute HF (head-forward angle) and SA (shoulder asymmetry) from
    nose + both shoulders only.

    Returns (hf_deg, sa_norm) or (None, None) if unreliable.
    """
    from mediapipe.python.solutions.pose import PoseLandmark

    nose, vis_nose = get_landmark(landmarks, PoseLandmark.NOSE, image_shape)
    l_sh, vis_ls = get_landmark(landmarks, PoseLandmark.LEFT_SHOULDER, image_shape)
    r_sh, vis_rs = get_landmark(landmarks, PoseLandmark.RIGHT_SHOULDER, image_shape)

    #these are required these three landmarks to be reasonably visible
    vis_min = min(vis_nose, vis_ls, vis_rs)
    if vis_min < 0.5:
        return None, None

    shoulder_mid = 0.5 * (l_sh + r_sh)

    #reference verticalup vector in image coordinates (y grows down)
    ref_up = np.array([0.0, -1.0], dtype=np.float32)

    # ---- Head-forward (HF) angle: shoulder_mid -> nose vs vertical ----
    v_hf = nose - shoulder_mid
    if np.linalg.norm(v_hf) < 1e-3:
        hf_deg = None
    else:
        v_hf_norm = v_hf / np.linalg.norm(v_hf)
        cos_ang = np.clip(np.dot(v_hf_norm, ref_up), -1.0, 1.0)
        ang_rad = np.arccos(cos_ang)
        hf_deg = float(np.degrees(ang_rad))

    # ---- Shoulder asymmetry (SA): vertical difference/shoulder width ----
    dy = l_sh[1] - r_sh[1]
    shoulder_width = np.linalg.norm(l_sh - r_sh)
    if shoulder_width < 1e-3:
        sa_norm = None
    else:
        sa_norm = float(abs(dy) / shoulder_width)

    return hf_deg, sa_norm


def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    hf_smoother = EmaSmoother(EMA_ALPHA)
    sa_smoother = EmaSmoother(EMA_ALPHA)

    dwell_frames = int(DWELL_SECONDS * TARGET_FPS)
    poor_counter = 0
    slouching = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # MediaPipe expects RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        hf_deg = sa_norm = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            hf_deg, sa_norm = compute_posture_metrics(landmarks, frame.shape)

            #drew skeleton (full pose is fine, only used head/shoulders)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
            )

        if hf_deg is not None and sa_norm is not None:
            hf_s = hf_smoother.update(hf_deg)
            sa_s = sa_smoother.update(sa_norm)

                        #to decide if posture is poor
            hf_flag = hf_s > HF_THRESHOLD_DEG
            sa_flag = sa_s > SA_THRESHOLD
            poor = hf_flag or sa_flag

            if poor:
                poor_counter += 1
            else:
                poor_counter = 0
                slouching = False

            if poor_counter >= dwell_frames:
                slouching = True

            #overlay metrics + debug info
            text1 = f"HF: {hf_s:.1f}° (>{HF_THRESHOLD_DEG}°? {'Y' if hf_flag else 'N'})"
            text2 = f"SA: {sa_s:.2f} (>{SA_THRESHOLD:.2f}? {'Y' if sa_flag else 'N'})"
            text3 = f"Poor frames: {poor_counter}/{dwell_frames}"

            cv2.putText(frame, text1, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(frame, text2, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(frame, text3, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if slouching:
                cv2.putText(
                    frame,
                    "POOR POSTURE",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.rectangle(
                    frame,
                    (5, 5),
                    (FRAME_WIDTH - 5, FRAME_HEIGHT - 5),
                    (0, 0, 255),
                    2,
                )
        else:
            cv2.putText(
                frame,
                "Landmarks unreliable (move closer / adjust lighting)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        cv2.imshow("Posture Monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):  #q to quit 
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
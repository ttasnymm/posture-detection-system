import cv2
import numpy as np
from ultralytics import YOLO

#configuration
FRAME_WIDTH = 960
FRAME_HEIGHT = 720

#thresholds for “poor posture”
HF_THRESHOLD_DEG = 18.0      # head-forward angle 
SA_THRESHOLD = 0.06       # normalised shoulder asymmetry (|dy|/shoulder width)


def compute_posture_from_yolo_kpts(kp: np.ndarray):
    """
    kp: (17, 2) array of keypoints (YOLOv8 pose, pixel coords)

    Uses:
      - nose (0)
      - left shoulder (5)
      - right shoulder (6)

    Returns (hf_deg, sa_norm) or (None, None) if unreliable.
    """
    # YOLOv8 COCO keypoint indices
    NOSE = 0
    L_SH = 5
    R_SH = 6

    nose = kp[NOSE]
    l_sh = kp[L_SH]
    r_sh = kp[R_SH]

    #if 0, treat as unreliable
    if np.any(nose == 0) or np.any(l_sh == 0) or np.any(r_sh == 0):
        return None, None

    shoulder_mid = 0.5 * (l_sh + r_sh)

    #reference vertical-up vector in image coords (y grows down)
    ref_up = np.array([0.0, -1.0], dtype=np.float32)

    # ---- Head-forward angle (HF): shoulder_mid → nose vs vertical ----
    v_hf = nose - shoulder_mid
    if np.linalg.norm(v_hf) < 1e-3:
        hf_deg = None
    else:
        v_hf_norm = v_hf / np.linalg.norm(v_hf)
        cos_ang = np.clip(np.dot(v_hf_norm, ref_up), -1.0, 1.0)
        ang_rad = np.arccos(cos_ang)
        hf_deg = float(np.degrees(ang_rad))

    # ---- Shoulder asymmetry (SA): vertical diff/shoulder width ----
    dy = l_sh[1] - r_sh[1]
    shoulder_width = np.linalg.norm(l_sh - r_sh)
    if shoulder_width < 1e-3:
        sa_norm = None
    else:
        sa_norm = float(abs(dy) / shoulder_width)

    return hf_deg, sa_norm


def main():
    #load YOLOv8 pose model
    model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        #run YOLOv8 pose (single frame, no tracking)
        results = model(frame, verbose=False, conf=0.5)[0]

        #get an image with only the skeleton drawn
        annotated = results.plot(
            boxes=False,
            labels=False,
            probs=False,
            line_width=2,
        )

        kpts = results.keypoints
        boxes = results.boxes

        if kpts is not None and boxes is not None:
            kpts_xy = kpts.xy.cpu().numpy()             #(N, 17, 2)
            boxes_xyxy = boxes.xyxy.cpu().numpy()    #(N, 4)

            for kp, box in zip(kpts_xy, boxes_xyxy):
                x1, y1, x2, y2 = box.astype(int)

                hf_deg, sa_norm = compute_posture_from_yolo_kpts(kp)

                if hf_deg is None or sa_norm is None:
                    continue

                #poor if head-forward OR shoulders tilted
                hf_flag = hf_deg > HF_THRESHOLD_DEG
                sa_flag = sa_norm > SA_THRESHOLD
                poor = hf_flag or sa_flag

                colour = (0, 0, 255) if poor else (0, 180, 0)  # red / soft green
                status = "POOR" if poor else "OK"

                #thin rectangle around the person
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 1)

                label = f"{status}  HF:{hf_deg:.1f}  SA:{sa_norm:.2f}"
                label_y = max(y1 - 8, 18)

                cv2.putText(
                    annotated,
                    label,
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    3,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    colour,
                    1,
                )

        cv2.imshow("Multi-person Live Pose & Posture (YOLOv8)", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):     #ESC or q to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
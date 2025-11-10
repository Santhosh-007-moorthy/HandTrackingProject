import cv2
import mediapipe as mp
import time, os, math
from datetime import datetime

# =================== CONFIG ===================
FINGER_CLOSE_LIMIT = 5.0       # sec
STILL_LIMIT = 5.0              # sec
OUT_OF_FRAME_LIMIT = 5.0       # sec
TIED_DISTANCE_THRESH = 0.08    # normalized distance (wrists too close)
STILL_MOVEMENT_THRESH = 0.01   # normalized motion threshold
SAVE_DIR = "Recordings"
# ==============================================

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---- create folder & video writer ----
os.makedirs(SAVE_DIR, exist_ok=True)
filename = os.path.join(SAVE_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not found!")
    exit()

w, h = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(filename, fourcc, fps, (w, h))

finger_close_start = {"Left": [None]*5, "Right": [None]*5}
last_wrist_pos = {"Left": None, "Right": None}
still_start = {"Left": None, "Right": None}
out_of_frame_start = {"Left": None, "Right": None}
TIP_IDS = [4, 8, 12, 16, 20]

print(f"🎥 Recording started — press 'q' to quit.\nSaving to: {filename}")

while True:
    success, frame = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_res = pose.process(img_rgb)
    hands_res = hands.process(img_rgb)
    errors = []

    if pose_res.pose_landmarks:
        lm = pose_res.pose_landmarks.landmark
        chin_y = lm[mp_pose.PoseLandmark.NOSE].y
        hip_y = (lm[mp_pose.PoseLandmark.LEFT_HIP].y + lm[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
        mp_drawing.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        errors.append("❌ Pose not detected!")

    hand_info = {}
    if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
        for hlm, handed in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
            label = handed.classification[0].label
            hand_info[label] = hlm
            mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

    for label in ("Left", "Right"):
        now = time.time()
        if label not in hand_info:
            if out_of_frame_start[label] is None:
                out_of_frame_start[label] = now
            elif now - out_of_frame_start[label] > OUT_OF_FRAME_LIMIT:
                errors.append(f"❌ {label} hand out of frame > {int(OUT_OF_FRAME_LIMIT)} s!")
            continue
        else:
            out_of_frame_start[label] = None

        hlm = hand_info[label]
        wrist = hlm.landmark[mp_hands.HandLandmark.WRIST]

        if pose_res.pose_landmarks:
            out_of_range = False
            for idx in [mp_hands.HandLandmark.WRIST] + [4, 8, 12, 16, 20]:
                y = hlm.landmark[idx].y
                if y < chin_y or y > hip_y:
                    out_of_range = True
                    break
            if out_of_range:
                errors.append(f"⚠️ {label} hand out of chin–hip range!")

        for i, tip in enumerate(TIP_IDS):
            if tip == 4:
                closed = hlm.landmark[tip].x > hlm.landmark[3].x if label == "Right" else hlm.landmark[tip].x < hlm.landmark[3].x
            else:
                closed = hlm.landmark[tip].y > hlm.landmark[tip - 2].y
            if closed:
                if finger_close_start[label][i] is None:
                    finger_close_start[label][i] = now
                elif now - finger_close_start[label][i] > FINGER_CLOSE_LIMIT:
                    errors.append(f"❌ {label} finger {i+1} closed > {int(FINGER_CLOSE_LIMIT)} s!")
            else:
                finger_close_start[label][i] = None

        pos = (wrist.x, wrist.y)
        if last_wrist_pos[label] is None:
            last_wrist_pos[label] = pos
            still_start[label] = None
        else:
            dist = math.dist(pos, last_wrist_pos[label])
            if dist < STILL_MOVEMENT_THRESH:
                if still_start[label] is None:
                    still_start[label] = now
                elif now - still_start[label] > STILL_LIMIT:
                    errors.append(f"❌ {label} hand static > {int(STILL_LIMIT)} s!")
            else:
                still_start[label] = None
            last_wrist_pos[label] = pos

    if "Left" in hand_info and "Right" in hand_info:
        lw = hand_info["Left"].landmark[mp_hands.HandLandmark.WRIST]
        rw = hand_info["Right"].landmark[mp_hands.HandLandmark.WRIST]
        dist = math.dist((lw.x, lw.y), (rw.x, rw.y))
        if dist < TIED_DISTANCE_THRESH:
            errors.append("❌ Hands too close (tied together)!")

    if errors:
        for i, e in enumerate(dict.fromkeys(errors)):
            cv2.putText(frame, e, (20, 40 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    else:
        cv2.putText(frame, "✅ All conditions OK", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    out.write(frame)
    cv2.imshow("Hand Pose Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n✅ Video saved to: {filename}")

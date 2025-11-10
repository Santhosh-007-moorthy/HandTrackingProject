import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(img_rgb)
    hands_result = hands.process(img_rgb)

    # Draw pose
    if pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            pose_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )

        # Pose landmarks
        landmarks = pose_result.pose_landmarks.landmark
        chin_y = landmarks[mp_pose.PoseLandmark.NOSE].y
        left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
        right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
        hip_y = (left_hip_y + right_hip_y) / 2

        # Process hands
        if hands_result.multi_hand_landmarks:
            for hand_landmarks in hands_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y

                # Height condition
                if wrist_y < chin_y:
                    height_status = "Hand too HIGH"
                    height_color = (0, 0, 255)
                elif wrist_y > hip_y:
                    height_status = "Hand too LOW"
                    height_color = (0, 255, 255)
                else:
                    height_status = "Hand in correct range"
                    height_color = (0, 255, 0)

                # --- Finger counting logic ---
                tip_ids = [4, 8, 12, 16, 20]
                fingers = []

                for i in range(1, 5):  # for all except thumb
                    if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Thumb detection (horizontal)
                if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
                    fingers.append(1)
                else:
                    fingers.append(0)

                total_fingers = fingers.count(1)

                # --- Finger condition ---
                if total_fingers >= 4:
                    finger_status = "✅ Valid pointing gesture"
                    finger_color = (0, 255, 0)
                else:
                    finger_status = "⚠️ Error: Use at least 4 fingers"
                    finger_color = (0, 0, 255)

                # Display both results
                cv2.putText(frame, height_status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, height_color, 3)
                cv2.putText(frame, finger_status, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, finger_color, 3)

    # Show output
    cv2.imshow("Pose + Hand + Finger Check", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

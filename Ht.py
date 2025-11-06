import cv2
import mediapipe as mp
import time
cap =cv2.VideoCapture(1)
mpHands=mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

#hands=mpHands.Hands()
while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    cv2.imshow("Image",img)
    cv2.waitkey(1)

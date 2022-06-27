import random

import cv2
import pafy
import mediapipe as mp
import os

m_draw = mp.solutions.drawing_utils
m = mp.solutions.pose
po = m.Pose()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

p = pafy.new("https://www.youtube.com/watch?v=arlMneoLl90")
best = p.getbest(preftype="mp4")
webcam = cv2.VideoCapture("TOP 10 BEST Dance Groups Around The World 2019 _ Got Talent Global.mp4")
print(best.url)

while webcam.isOpened():
    sucessful_frame_read, frame = webcam.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = po.process(frame_rgb)
    if results.pose_landmarks:
        m_draw.draw_landmarks(frame,results.pose_landmarks,m.POSE_CONNECTIONS)

    cv2.imshow("webcam", frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

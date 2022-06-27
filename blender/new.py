import bpy
import math
import sys
import threading
import time

from urllib.request import urlopen

import numpy as np

import cv2
import mediapipe as mp
from numpy import ceil

width = 1280
height = 720

webcam = cv2.imread("download (2).jpg")

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

count = 0
time_begin = time.perf_counter()
right_pointer_x = 0
right_pointer_y = 0
left_pointer_x = 0
left_pointer_y = 0
left_angle = 0
right_angle = 0
im = ""
right_closed = False
left_closed = False
left_edit = True
right_edit = True
speed = 0


def angle(li_st, points):
    x1, y1 = li_st[points[0]].x, li_st[points[0]].y
    x2, y2 = li_st[points[1]].x, li_st[points[1]].y
    x3, y3 = li_st[points[2]].x, li_st[points[2]].y

    return math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))


def pre_wok():
    frame = webcam
    cv2.imshow("shit", frame)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        t1 = threading.Thread(target=work, args=[holistic, frame])
        t1.start()
        t1.join()
        cv2.imshow("dh", im)
        k = cv2.waitKey(100000)
        if k == 27:
            webcam.release()
            cv2.destroyAllWindows()


def work(holistic, frame):
    global height, width, right_closed, left_closed, left_edit, right_edit, speed, time_begin

    global count, left_angle, right_angle, right_pointer_x, right_pointer_y, left_pointer_x, left_pointer_y, im
    # Recolor Feed
    cx = 0
    cy = 0

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    # Make Detections
    results = holistic.process(image)
    # print(results.face_landmarks)
    # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
    # Recolor image back to BGR for rendering

    # Draw face landmarks
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

    h, w, c = height, width, image.shape[-1]
    r = results.right_hand_landmarks
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        # 5,9,13,17

        if r.landmark[9].y >= r.landmark[12].y:
            right_closed = False



        else:

            right_closed = True
        if right_closed and right_edit:
            x, y = [], []
            for i in [0, 5, 9, 13, 17]:
                x.append(r.landmark[i].x * w)
                y.append(r.landmark[i].y * h)
            cx = sum(x) / len(x)
            cy = sum(y) / len(y)
            right_pointer_x = cx
            right_pointer_y = cy
        print("right:", right_closed)

        cv2.circle(image, [int(cx), int(cy)], 3, (204, 34, 55), -1)

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    l = results.left_hand_landmarks
    if results.left_hand_landmarks:
        # 5,9,13,17

        if l.landmark[9].y >= l.landmark[12].y:
            left_closed = False
        else:

            left_closed = True
        if left_closed and left_edit:
            x, y = [], []
            for i in [0, 5, 9, 13, 17]:
                x.append(l.landmark[i].x * w)
                y.append(l.landmark[i].y * h)
            cx = sum(x) / len(x)
            cy = sum(y) / len(y)
            cv2.circle(image, [int(cx), int(cy)], 3, (204, 34, 55), -1)
            left_pointer_x = cx
            left_pointer_y = cy
        print("left", left_closed)
    p = results.pose_landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if p:
        print(p.landmark)
        print(len(p.landmark))
        for i, li in enumerate(p.landmark):
            # print(h)
            # print(w)
            # print(li)
            print(i)

    im = image
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)


def onrelease():
    global left_pointer_y, left_pointer_x, right_pointer_y, right_pointer_x, left_closed, right_closed, left_edit
    global right_edit

    if not (int(left_pointer_x) == 0 and int(left_pointer_y) == 0):

        if not left_closed:
            left_edit = False

    if not (int(right_pointer_x) == 0 and int(right_pointer_y) == 0):
        if not right_closed:
            right_edit = False


t = threading.Thread(target=pre_wok)
t.start()
score = 0

bpy.ops.mesh.primitive_monkey_add()
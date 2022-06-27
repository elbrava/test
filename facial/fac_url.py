
import random

import cv2
import pafy
from fer import FER
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
detector = FER()
print(detector)
p = pafy.new("https://www.youtube.com/watch?v=nD-Zu6aayS0")
best = p.getbest(preftype="mp4")
webcam = cv2.VideoCapture(best.url)
print(best.url)

while webcam.isOpened():
    sucessful_frame_read, frame = webcam.read()

    results = detector.detect_emotions(frame)
    print(list(results).__len__())
    # bounding_box = result[0]["box"]
    # emotions = result[0]["emotions"]
    for result in list(results):
        bounding_box = result["box"]
        print(bounding_box)
        color = (random.randrange(256), random.randrange(256),random.randrange(256))
        emotions = result["emotions"]
        # print(emotions)
        cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (
            bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      color, 2, )
        emotion_name, score = detector.top_emotion(frame)
        if score is not None:
            val = f"{score * 100}%"
        else:
            val = f"{score}"
        cv2.putText(frame, f"{emotion_name}:{val}",
                    (bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + 2 * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA, )
        cv2.imshow("webcam", frame)

    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

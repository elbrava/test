import math

from cmath import atan
from statistics import linear_regression
from time import sleep

import cv2
import cvzone
import numpy as np


def get_quad(lin1, lin2):
    a = lin1 ** 2
    b = 2 * lin1 * lin2
    c = lin2 ** 2

    return a, b, c


def solve_quad(a, b, c):
    num1 = -b + (b ** 2 - 4 * a * c) ** 0.5
    num2 = -b - (b ** 2 - 4 * a * c) ** 0.5
    denom = 2 * a
    return [num1 / denom, num2 / denom]


def positive(num):
    if num > 0:
        return True


def distance_(point_s):
    x1, y1 = point_s[0]
    x2, y2 = point_s[1]
    g = (y2 - y1) ** 2 + (x2 - x1) ** 2
    return g ** 0.5


def angle(point_s):
    x1, y1 = point_s[0]
    x2, y2 = point_s[1]
    # g = (y2 - y1) / (x2 - x1)
    val = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
    return val


width, height = 800, 800

x = 0
y = 0

webcam = cv2.VideoCapture(0)
blur = 3
k = 0
linesize = 7

while webcam.isOpened():
    break
    distance = 300

    _, frame = webcam.read()
    mask = np.zeros(frame.shape, np.uint8)
    m = mask.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayBlur = cv2.medianBlur(gray, blur)
    edges = cv2.adaptiveThreshold(grayBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, linesize, blur)
    cv2.imshow("jj", edges)

    imgCont, hie = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in imgCont:
        # cv2.drawContours(m, i, -1, (0, 90, 89), 1)
        peri = cv2.arcLength(i, True)
        pprox = [i[0] for i in cv2.approxPolyDP(i, 0.02 * peri, True)]

        print(type(pprox))
        cv2.fillPoly(m, pprox, (0, 80, 50))
        cv2.imshow("mas", m)
        cv2.waitKey(1)
    cv2.waitKey(1)


def angle_T(points):
    x1, y1 = points[0][0], points[0][1]
    x2, y2 = points[1][0], points[1][1]
    x3, y3 = points[2][0], points[2][1]

    return math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))


def rot(x1, y1, ii):
    x = x1 + distance * math.cos(math.radians(ii))
    y = y1 + distance * math.sin(math.radians(ii))
    # print(x, y)
    p = ii % 360
    points = [(int(x), int(y))]
    s = angle(points)
    return x, y


def closest(an, the_point, points):
    print("p", the_point)
    try:
        gradient = -math.tan(an) ** -1
    except ZeroDivisionError:
        gradient = 1
    min_ = 10000000000000000000000000000000000000000000000000000
    point = []
    y_ = the_point[1]
    for p in points:
        eqn1 = gradient * p[0] + y_
        eqn = abs(p[1] - eqn1)
        if eqn <= min_:
            min_ = eqn
            point = p
    return point, min_


def triplet(points):
    result = []
    print(points)
    if len(points) <= 2:
        raise ValueError("length of points should be greater than 3")
    else:
        l = len(points)
        p = points.reverse()
        for j, _ in enumerate(points):
            try:
                p = points[j + 1]
            except:
                p = points[0]

            cap = (points[j - 1], points[j], p)
            result.append(cap)
    return result


def calc(point_s, thresh=120):
    t = triplet(point_s)
    print(t)
    for a in t:
        print(abs(angle_T(a)))
        an = angle([[0, 0], a[1]])

        print(closest(an, a[1], point_s))


calc([[0, 0], [90, 0], [180, 0], [90, 0]])

# linear regression

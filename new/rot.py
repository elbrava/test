import math

math.tan
from cmath import atan

import cv2
import numpy as np
from numpy import angle


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
    g = (y2 - y1) / (x2 - x1)
    val = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
    return val


width, height = 800, 800
mask = np.zeros((width, height), np.uint8)
ii = 360
x = 0
y = 0
while True:
    # mask = np.zeros((width, height), np.uint8)
    distance = 300
    try:
        p = (ii % 360)

        if p >= 179:
            g = math.tan(math.radians(-(ii - 90)))
            x = distance ** 2 / (1 + g ** 2)
            y = height // 2 + g * x ** 0.5
            # x = width // 2 - x ** 0.5

            print(p)

            x = width // 2 - x ** 0.5



        else:
            g = math.tan(math.radians(ii - 90))
            x = distance ** 2 / (1 + g ** 2)
            y = height // 2 + g * x ** 0.5
            # x = width // 2 - x ** 0.5

            print(p)
            # y = height // 2 + g * x ** 0.5
            x = width // 2 + x ** 0.5
        # quad
        # val=-b+

        print("i", ii)
        # print(x, y)
        points = [(width // 2, height // 2), (int(x), int(y))]
        s = angle(points)

        print("p", p / s)
        cv2.line(mask, *points, (55, 90, 0), 20)


    except Exception as e:
        print("e", e)
    finally:
        cv2.imshow("mask", mask)
        ii += 0.01
        cv2.waitKey(1)

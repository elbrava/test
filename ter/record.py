import sounddevice as sd
from scipy.io.wavfile import write
import cv2
import cvzone
FPS = 60
fs = 44100  # Sample rate
seconds = 3  # Duration of recording
webcam = cv2.VideoCapture(0)
webcam.set(5, 60)
myrecording = ""


def main():
    global myrecording
    with open("main.wav", "ab") as o:
        while webcam.isOpened():
            _, frame = webcam.read()

            cv2.imshow("f", frame)
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
            try:

                write('output.wav', fs, myrecording)
                f = open("output.wav", "rb")
                o.write(f.read())

                # Save as WAV file
            except Exception as e:
                print(e)
            cv2.imwrite("wh.png", frame)
            sd.wait()
            cv2.waitKey(1)

        # Wait until recording is finished


def record():
    global myrecording
    with open("main.wav", "ab") as o:
        while True:
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
            try:

                write('output.wav', fs, myrecording)
                f = open("output.wav", "rb")
                o.write(f.read())

                # Save as WAV file
            except Exception as e:
                print(e)
            sd.wait()


if __name__ == '__main__':
    record()

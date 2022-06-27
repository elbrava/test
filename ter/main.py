import subprocess
import threading
import time
from operator import __iconcat__
from moviepy.editor import AudioFileClip, ImageClip

import cv2

import argparse
import tempfile
import queue
import sys

import ffmpeg
import sounddevice as sd
import soundfile as sf
import numpy  # Make sure NumPy is loaded before it is used in the callback

assert numpy  # avoid "imported but unused" message (W0611)
audio_feed = True

webcam = cv2.VideoCapture(0)
count = 770
movie_count = -7
time_begin = time.perf_counter()


def add_static_image_to_audio():
    global movie_count, count
    while movie_count << count:
        print(movie_count)
        movie_count += 1
        if movie_count <= -1:

            continue
        else:
            print("here")
            """Create and save a video file to `output_path` after 
            combining a static image that is located in `image_path` 
            with an audio file in `audio_path`"""
            try:
                audio_path = f"sound/{movie_count}.wav"
                image_path = f"img/{movie_count}.png"
                output_path = f"vid/{movie_count}.mp4"
                # create the audio clip object
                audio_clip = AudioFileClip(audio_path)
                # create the image clip object
                image_clip = ImageClip(image_path)
                # use set_audio method from image clip to combine the audio with the image
                video_clip = image_clip.set_audio(audio_clip)
                # specify the duration of the new clip to be the duration of the audio clip
                video_clip.duration = audio_clip.duration
                # set the FPS to 1
                time_n = time.perf_counter() - time_begin
                video_clip.fps = 1 / time_n
                # write the resuling video clip
                video_clip.write_videofile(output_path)
            except Exception as e:
                print(e)
                continue
            finally:
               pass


def sound_main():
    def int_or_str(text):

        try:
            return int(text)
        except ValueError:
            return text

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        'filename', nargs='?', metavar='FILENAME',
        help='audio file to store recording to')
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-r', '--samplerate', type=int, help='sampling rate')
    parser.add_argument(
        '-c', '--channels', type=int, default=1, help='number of input channels')
    parser.add_argument(
        '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
    args = parser.parse_args(remaining)

    q = queue.Queue()

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    try:
        if args.samplerate is None:
            device_info = sd.query_devices(args.device, 'input')
            # soundfile expects an int, sounddevice provides a float:
            args.samplerate = int(device_info['default_samplerate'])
        if args.filename is None:
            args.filename = f"sound/{count}.wav"

        # Make sure the file is opened before recording anything:
        with sf.SoundFile(args.filename, mode='x', samplerate=args.samplerate,
                          channels=args.channels, subtype=args.subtype) as file:
            with sd.InputStream(samplerate=args.samplerate, device=args.device,
                                channels=args.channels, callback=callback):
                print('#' * 80)
                print('press Ctrl+C to stop the recording')
                print('#' * 80)
                while True:
                    if not audio_feed:
                        break
                    file.write(q.get())
    except KeyboardInterrupt:
        print('\nRecording finished: ' + repr(args.filename))
        parser.exit(0)
    except Exception as e:
        parser.exit(type(e).__name__ + ': ' )
def sound():

def main():
    global count

    while webcam.isOpened():
        audio_feed = True
        threading.Thread(target=sound_main).start()
        _, frame = webcam.read()

        cv2.imshow("f", frame)
        cv2.imwrite(fr"img/{count}.png", frame)
        audio_feed = False

        count += 1
        cv2.waitKey(1)

    # Wait until recording is f


if __name__ == '__main__':
    threading.Thread(target=add_static_image_to_audio).start()
    # main()

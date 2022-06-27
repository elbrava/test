from encodings.utf_16 import decode

import pydub
import requests
import sounddevice as sd

from pydub.playback import play

from pydub import AudioSegment

stream_url = 'https://ep256.hostingradio.ru:8052/europaplus256.mp3'
r = requests.get(stream_url, stream=True)

with r:
    try:
        for block in r.iter_content(1024):
            print(decode(block))
            play(AudioSegment.from_raw(block, sample_width=1024, frame_rate=44100, channels=1))
    except KeyboardInterrupt:
        pass

"""import subprocess

l = subprocess.Popen("ngrok http 8000", shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                     stderr=subprocess.PIPE)
print(l.stdout.read())
"""
import pyngrok
from pyngrok import ngrok
s=ngrok.connect(8000)
t=ngrok._current_tunnels
print(t)
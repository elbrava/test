from django.conf.urls import url
from django.urls import path, re_path

from . import consumers

from django.urls import re_path

from . import consumers

websocket_urlpatterns = [

re_path(r'ws/call/', consumers.CallConsumer.as_asgi()),

]


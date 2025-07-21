# backend/gesture/routing.py

from django.urls import path # re_path 대신 path를 import
from . import consumers

websocket_urlpatterns = [
    # 정규표현식(re_path) 대신 간단한 path로 변경
    path('ws/gesture/', consumers.GestureConsumer.as_asgi()),
]
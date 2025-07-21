import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import gesture.routing # gesture 앱의 routing을 가져옵니다.

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings') # 'config'로 변경

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            gesture.routing.websocket_urlpatterns
        )
    ),
})
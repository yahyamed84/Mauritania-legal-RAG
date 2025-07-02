"""
WSGI config for mauritania_rag project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

# Make sure the DJANGO_SETTINGS_MODULE environment variable is set
# Adjust the path if your settings file is located differently
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mauritania_rag.settings")

application = get_wsgi_application()


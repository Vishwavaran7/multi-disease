# Root-level app entry point for gunicorn (app:app)
from wsgi import app

__all__ = ['app']

import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'medisense-ai-secret-key-2024'
    
    # Use absolute path for SQLite so it works on any hosting environment
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or f'sqlite:///{os.path.join(_BASE_DIR, "medisense.db")}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') or 'AIzaSyAWQouTsVaatOo760akGvNZH2C8e5YTnSg'
    GEMINI_MODEL = 'gemini-2.5-flash'
    
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = 'ayushtiwari.creatorslab@gmail.com'
    MAIL_PASSWORD = 'tecx bcym vxdz dtni'
    MAIL_DEFAULT_SENDER = 'ayushtiwari.creatorslab@gmail.com'
    
    NOMINATIM_USER_AGENT = 'medisense-ai-app'
    
    MODELS_PATH = 'models/'
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'medisense-ai-secret-key-2024'
    
    SQLALCHEMY_DATABASE_URI = 'sqlite:///medisense.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    GEMINI_API_KEY = 'AIzaSyBSn79F5in1GwL4IZMF-RkB2QotYylFv7Y'
    GEMINI_MODEL = 'gemini-2.5-flash'
    
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = 'ayushtiwari.creatorslab@gmail.com'
    MAIL_PASSWORD = 'tecx bcym vxdz dtni'
    MAIL_DEFAULT_SENDER = 'ayushtiwari.creatorslab@gmail.com'
    
    NOMINATIM_USER_AGENT = 'medisense-ai-app'
    
    MODELS_PATH = 'models/'
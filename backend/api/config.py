"""
Configuration Settings for Flask API
"""

import os
from pathlib import Path


class Config:
    """Base configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = False
    TESTING = False
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    # Data directories
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / 'data'
    UPLOAD_FOLDER = DATA_DIR / 'raw'
    PROCESSED_FOLDER = DATA_DIR / 'processed'
    OUTPUT_FOLDER = DATA_DIR / 'outputs'
    
    # KNN Imputation settings
    DEFAULT_N_NEIGHBORS = 5
    DEFAULT_WEIGHTS = 'uniform'
    
    # CORS settings
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:8501']  # React, Streamlit
    
    @staticmethod
    def init_app(app):
        """Initialize application"""
        # Create directories if they don't exist
        Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        Config.PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
        Config.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
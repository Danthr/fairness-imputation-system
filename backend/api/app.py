"""
Flask Application Factory
"""

from flask import Flask
from flask_cors import CORS
from .config import config
import logging


def create_app(config_name='development'):
    """
    Create and configure Flask application
    
    Args:
        config_name: Configuration to use (development, production, testing)
        
    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    
    # Enable CORS
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Register blueprints
    from .routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Health check endpoint
    @app.route('/health')
    def health():
        return {'status': 'healthy', 'message': 'API is running'}, 200
    
    @app.route('/')
    def index():
        return {
            'message': 'Fairness Imputation System API',
            'version': '1.0.0',
            'endpoints': {
                'health': '/health',
                'upload': '/api/upload',
                'process': '/api/process',
                'results': '/api/results/<dataset_id>',
                'datasets': '/api/datasets'
            }
        }, 200
    
    return app


if __name__ == '__main__':
    app = create_app('development')
    app.run(host='0.0.0.0', port=5000, debug=True)
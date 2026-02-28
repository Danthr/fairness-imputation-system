"""
Main entry point to run the Flask API
"""

from backend.api import create_app

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Fairness Imputation System API")
    print("="*60)
    print("API running at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    print("="*60 + "\n")
    
    app = create_app('development')
    app.run(host='0.0.0.0', port=5000, debug=True)
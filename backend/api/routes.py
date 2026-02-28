"""
API Routes
Endpoints for the ML backend
"""

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import uuid
import os
from pathlib import Path
import logging

from backend.data_processing import DataIngestion, DataValidator
from backend.imputation import ImputationEngine, EnhancedImputationEngine
from backend.fairness_audit import FairnessAuditor

logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__)

# Store results in memory (in production, use database)
results_store = {}


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}


@api_bp.route('/upload', methods=['POST'])
def upload_file():
    """
    Upload dataset file
    
    Returns:
        JSON with dataset_id and basic info
    """
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Use CSV or Excel files'}), 400
        
        # Generate unique ID for this dataset
        dataset_id = str(uuid.uuid4())
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = Path('data/raw') / f"{dataset_id}_{filename}"
        file.save(str(file_path))
        
        logger.info(f"File uploaded: {filename} with ID: {dataset_id}")
        
        # Load and validate dataset
        ingestion = DataIngestion()
        df, message = ingestion.load_dataset(str(file_path))
        
        if df is None:
            return jsonify({'error': message}), 400
        
        # Get basic statistics
        stats = ingestion.get_basic_stats(df)
        
        # Convert numpy types
        stats = convert_numpy_types(stats)
        
        # Store dataset info
        results_store[dataset_id] = {
            'filename': filename,
            'file_path': str(file_path),
            'upload_status': 'success',
            'stats': stats,
            'processed': False
        }
        
        return jsonify({
            'dataset_id': dataset_id,
            'filename': filename,
            'message': 'File uploaded successfully',
            'stats': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/process', methods=['POST'])
def process_dataset():
    """
    Process dataset: imputation + fairness audit
    
    Request body:
        {
            "dataset_id": "uuid",
            "method": "knn" | "mice" | "random_forest" | "compare_all",  # optional, default: knn
            "n_neighbors": 5,  # optional
            "include_confidence": true | false,  # optional, default: false
            "protected_attributes": ["gender", "race"]  # optional
        }
    
    Returns:
        JSON with processing results
    """
    try:
        data = request.get_json()
        
        # Validate request
        if not data or 'dataset_id' not in data:
            return jsonify({'error': 'dataset_id required'}), 400
        
        dataset_id = data['dataset_id']
        
        # Check if dataset exists
        if dataset_id not in results_store:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Get parameters
        n_neighbors = data.get('n_neighbors', 5)
        method = data.get('method', 'knn')
        include_confidence = data.get('include_confidence', False)
        protected_attrs = data.get('protected_attributes', None)
        
        logger.info(f"Processing dataset: {dataset_id} with method: {method}")
        
        # Load dataset
        file_path = results_store[dataset_id]['file_path']
        ingestion = DataIngestion()
        df, _ = ingestion.load_dataset(file_path)
        
        # Step 1: Preprocess
        logger.info("Step 1: Preprocessing...")
        df_processed = ingestion.preprocess_dataset(df)
        
        # Step 2: Imputation with method selection
        logger.info(f"Step 2: Running {method} imputation...")
        
        comparison_results = None
        confidence_scores = None
        
        if method == 'compare_all':
            # Compare all methods
            engine = EnhancedImputationEngine(n_neighbors=n_neighbors)
            comparison_results = engine.compare_all_methods(df_processed)
            # Use KNN results for final dataset
            df_imputed, imputation_stats, confidence_scores = engine.knn_impute_with_confidence(df_processed)
            
        elif method == 'knn' and include_confidence:
            # KNN with confidence scores
            engine = EnhancedImputationEngine(n_neighbors=n_neighbors)
            df_imputed, imputation_stats, confidence_scores = engine.knn_impute_with_confidence(df_processed)
            
        elif method == 'mice':
            # MICE imputation
            engine = EnhancedImputationEngine(n_neighbors=n_neighbors)
            df_imputed, imputation_stats = engine.mice_impute(df_processed)
            
        elif method == 'random_forest':
            # Random Forest imputation
            engine = EnhancedImputationEngine(n_neighbors=n_neighbors)
            df_imputed, imputation_stats = engine.random_forest_impute(df_processed)
            
        else:
            # Default: basic KNN without confidence
            engine = ImputationEngine(n_neighbors=n_neighbors)
            df_imputed, imputation_stats = engine.knn_impute(df_processed)
        
        # Get imputation report
        imputation_report = engine.get_imputation_report(df_processed, df_imputed)
        
        # Step 3: Fairness Audit
        logger.info("Step 3: Running fairness audit...")
        auditor = FairnessAuditor()
        
        # Auto-detect protected attributes if not provided
        if protected_attrs is None:
            protected_attrs = auditor.detect_protected_attributes(df_imputed)
        
        # Run audit on each protected attribute
        fairness_results = {}
        if protected_attrs:
            # Use first numerical column as outcome
            numerical_cols = df_imputed.select_dtypes(include=['number']).columns.tolist()
            if numerical_cols:
                outcome_attr = numerical_cols[0]
                
                for attr in protected_attrs:
                    try:
                        audit_result = auditor.audit_simple(
                            df_imputed, 
                            attr, 
                            outcome_attr
                        )
                        fairness_results[attr] = audit_result
                    except Exception as e:
                        logger.error(f"Error auditing {attr}: {str(e)}")
                        fairness_results[attr] = {'error': str(e)}
        
        # Save processed dataset
        output_filename = f"{dataset_id}_processed.csv"
        output_path = ingestion.save_dataset(df_imputed, output_filename, 'outputs')
        
        # Convert all results to JSON-serializable types
        imputation_stats = convert_numpy_types(imputation_stats)
        imputation_report_dict = convert_numpy_types(imputation_report.to_dict('records'))
        fairness_results = convert_numpy_types(fairness_results)
        
        # Store results
        results_store[dataset_id].update({
            'processed': True,
            'imputation_stats': imputation_stats,
            'imputation_report': imputation_report_dict,
            'fairness_results': fairness_results,
            'protected_attributes': protected_attrs,
            'output_path': output_path,
            'n_neighbors_used': n_neighbors,
            'method_used': method
        })
        
        logger.info(f"Processing complete for dataset: {dataset_id}")
        
        # Build response with optional fields
        response = {
            'dataset_id': dataset_id,
            'message': 'Processing complete',
            'imputation': {
                'stats': imputation_stats,
                'report': imputation_report_dict
            },
            'fairness_audit': fairness_results,
            'output_file': output_path
        }
        
        # Add optional fields if available
        if confidence_scores:
            response['confidence_scores'] = convert_numpy_types(confidence_scores)
        
        if comparison_results:
            response['method_comparison'] = convert_numpy_types(comparison_results)
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/results/<dataset_id>', methods=['GET'])
def get_results(dataset_id):
    """
    Get results for a processed dataset
    
    Returns:
        JSON with all results
    """
    try:
        if dataset_id not in results_store:
            return jsonify({'error': 'Dataset not found'}), 404
        
        results = results_store[dataset_id]
        
        if not results.get('processed', False):
            return jsonify({
                'dataset_id': dataset_id,
                'status': 'uploaded',
                'message': 'Dataset not processed yet. Call /process first.',
                'stats': results.get('stats', {})
            }), 200
        
        return jsonify({
            'dataset_id': dataset_id,
            'status': 'processed',
            'filename': results['filename'],
            'stats': results['stats'],
            'imputation': {
                'stats': results['imputation_stats'],
                'report': results['imputation_report']
            },
            'fairness_audit': results['fairness_results'],
            'protected_attributes': results['protected_attributes'],
            'output_file': results['output_path'],
            'method_used': results.get('method_used', 'knn')
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/datasets', methods=['GET'])
def list_datasets():
    """
    List all uploaded datasets
    
    Returns:
        JSON with list of datasets
    """
    try:
        datasets = []
        for dataset_id, info in results_store.items():
            datasets.append({
                'dataset_id': dataset_id,
                'filename': info['filename'],
                'processed': info.get('processed', False),
                'total_rows': info['stats']['total_rows'],
                'total_columns': info['stats']['total_columns'],
                'method_used': info.get('method_used', 'N/A')
            })
        
        return jsonify({
            'total_datasets': len(datasets),
            'datasets': datasets
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        return jsonify({'error': str(e)}), 500
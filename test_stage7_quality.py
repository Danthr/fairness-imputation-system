"""
Test Suite - Stage 7 Feature #4: Imputation Quality Metrics
Tests RMSE, MAE, cross-validation, and method comparison
"""

import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.imputation.quality_metrics import ImputationQualityMetrics


# ------------------------------------------------------------------
# Shared test data
# ------------------------------------------------------------------

def get_test_df():
    """Same dataset used in test_stage7.py for consistency"""
    return pd.DataFrame({
        'age':       [25.0, np.nan, 35.0, 45.0, np.nan, 55.0, 60.0, 30.0, 40.0, 50.0],
        'income':    [50000.0, np.nan, 70000.0, 80000.0, np.nan, 90000.0, np.nan, 55000.0, 65000.0, 75000.0],
        'gender':    ['M', 'F', np.nan, 'M', 'F', 'M', 'F', np.nan, 'M', 'F'],
        'education': ['BS', 'MS', 'BS', np.nan, 'PhD', 'BS', 'MS', 'BS', 'MS', 'BS']
    })


def get_complete_df():
    """A fully complete dataset - useful for masking tests"""
    return pd.DataFrame({
        'age':    [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0],
        'income': [50000.0, 55000.0, 60000.0, 65000.0, 70000.0,
                   75000.0, 80000.0, 85000.0, 90000.0, 95000.0],
    })


# ------------------------------------------------------------------
# Test 1: Core metric functions
# ------------------------------------------------------------------

def test_calculate_rmse():
    print("\nTest 1a: calculate_rmse")
    qm = ImputationQualityMetrics()

    actual    = np.array([10.0, 20.0, 30.0, 40.0])
    predicted = np.array([12.0, 18.0, 33.0, 39.0])

    rmse = qm.calculate_rmse(actual, predicted)
    print(f"  actual:    {actual}")
    print(f"  predicted: {predicted}")
    print(f"  RMSE:      {rmse:.4f}")

    assert rmse > 0, "RMSE should be positive"
    assert isinstance(rmse, float), "RMSE should be a float"
    print("  PASSED")


def test_calculate_mae():
    print("\nTest 1b: calculate_mae")
    qm = ImputationQualityMetrics()

    actual    = np.array([10.0, 20.0, 30.0, 40.0])
    predicted = np.array([12.0, 18.0, 33.0, 39.0])

    mae = qm.calculate_mae(actual, predicted)
    print(f"  actual:    {actual}")
    print(f"  predicted: {predicted}")
    print(f"  MAE:       {mae:.4f}")

    assert mae > 0, "MAE should be positive"
    assert mae <= qm.calculate_rmse(actual, predicted), "MAE should be <= RMSE"
    print("  PASSED")


def test_perfect_prediction():
    print("\nTest 1c: perfect prediction (RMSE and MAE should be 0)")
    qm = ImputationQualityMetrics()

    actual    = np.array([10.0, 20.0, 30.0])
    predicted = np.array([10.0, 20.0, 30.0])

    rmse = qm.calculate_rmse(actual, predicted)
    mae  = qm.calculate_mae(actual, predicted)

    print(f"  RMSE: {rmse}, MAE: {mae}")
    assert rmse == 0.0, "RMSE should be 0 for perfect prediction"
    assert mae  == 0.0, "MAE should be 0 for perfect prediction"
    print("  PASSED")


def test_categorical_accuracy():
    print("\nTest 1d: calculate_categorical_accuracy")
    qm = ImputationQualityMetrics()

    actual    = np.array(['M', 'F', 'M', 'F', 'M'])
    predicted = np.array(['M', 'F', 'F', 'F', 'M'])  # 4/5 correct

    acc = qm.calculate_categorical_accuracy(actual, predicted)
    print(f"  actual:    {actual}")
    print(f"  predicted: {predicted}")
    print(f"  accuracy:  {acc:.4f}")

    assert acc == 0.8, f"Expected 0.8, got {acc}"
    print("  PASSED")


# ------------------------------------------------------------------
# Test 2: Single method evaluation
# ------------------------------------------------------------------

def test_evaluate_method_knn():
    print("\nTest 2a: evaluate_method - KNN")
    qm = ImputationQualityMetrics()
    df = get_complete_df()

    result = qm.evaluate_method(df, method='knn')
    print(f"  method:      {result['method']}")
    print(f"  RMSE:        {result['rmse']}")
    print(f"  MAE:         {result['mae']}")
    print(f"  n_evaluated: {result['n_evaluated']}")
    print(f"  eval_time_s: {result['eval_time_s']}s")
    print(f"  per_column:  {result['per_column']}")

    assert result['method'] == 'knn'
    assert result['rmse'] >= 0
    assert result['mae']  >= 0
    assert result['mae']  <= result['rmse'], "MAE should be <= RMSE"
    assert result['n_evaluated'] > 0
    assert 'per_column' in result
    print("  PASSED")


def test_evaluate_method_mice():
    print("\nTest 2b: evaluate_method - MICE")
    qm = ImputationQualityMetrics()
    df = get_complete_df()

    result = qm.evaluate_method(df, method='mice')
    print(f"  RMSE: {result['rmse']}, MAE: {result['mae']}")

    assert result['method'] == 'mice'
    assert result['rmse'] >= 0
    print("  PASSED")


def test_evaluate_method_rf():
    print("\nTest 2c: evaluate_method - Random Forest")
    qm = ImputationQualityMetrics()
    df = get_complete_df()

    result = qm.evaluate_method(df, method='rf')
    print(f"  RMSE: {result['rmse']}, MAE: {result['mae']}")

    assert result['method'] == 'rf'
    assert result['rmse'] >= 0
    print("  PASSED")


def test_evaluate_method_mean():
    print("\nTest 2d: evaluate_method - Mean baseline")
    qm = ImputationQualityMetrics()
    df = get_complete_df()

    result = qm.evaluate_method(df, method='mean')
    print(f"  RMSE: {result['rmse']}, MAE: {result['mae']}")

    assert result['method'] == 'mean'
    assert result['rmse'] >= 0
    print("  PASSED")


def test_evaluate_on_df_with_missing():
    print("\nTest 2e: evaluate_method on df that already has missing values")
    qm = ImputationQualityMetrics()
    df = get_test_df()

    result = qm.evaluate_method(df, method='knn')
    print(f"  RMSE: {result['rmse']}, MAE: {result['mae']}, n_evaluated: {result['n_evaluated']}")

    assert 'error' not in result, f"Got error: {result.get('error')}"
    assert result['rmse'] >= 0
    print("  PASSED")


# ------------------------------------------------------------------
# Test 3: Cross-validation
# ------------------------------------------------------------------

def test_cross_validate_knn():
    print("\nTest 3a: cross_validate_method - KNN (3 folds)")
    qm = ImputationQualityMetrics()
    df = get_complete_df()

    result = qm.cross_validate_method(df, method='knn', n_folds=3)
    print(f"  method:        {result['method']}")
    print(f"  n_folds:       {result['n_folds']}")
    print(f"  cv_rmse_mean:  {result['cv_rmse_mean']}")
    print(f"  cv_rmse_std:   {result['cv_rmse_std']}")
    print(f"  cv_mae_mean:   {result['cv_mae_mean']}")
    print(f"  cv_mae_std:    {result['cv_mae_std']}")
    print(f"  fold_scores:   {result['fold_scores']}")

    assert result['method'] == 'knn'
    assert result['n_folds'] == 3
    assert result['cv_rmse_mean'] >= 0
    assert result['cv_mae_mean']  >= 0
    assert len(result['fold_scores']) == 3
    print("  PASSED")


def test_cross_validate_fold_scores_structure():
    print("\nTest 3b: fold scores have correct structure")
    qm = ImputationQualityMetrics()
    df = get_complete_df()

    result = qm.cross_validate_method(df, method='mean', n_folds=3)

    for fold in result['fold_scores']:
        assert 'fold' in fold
        assert 'rmse' in fold
        assert 'mae'  in fold
        assert fold['rmse'] >= 0
        assert fold['mae']  >= 0

    print(f"  All {len(result['fold_scores'])} folds have correct structure")
    print("  PASSED")


# ------------------------------------------------------------------
# Test 4: Multi-method comparison
# ------------------------------------------------------------------

def test_compare_method_quality():
    print("\nTest 4a: compare_method_quality - all 4 methods")
    qm = ImputationQualityMetrics()
    df = get_complete_df()

    result = qm.compare_method_quality(df, methods=['knn', 'mice', 'rf', 'mean'])

    print(f"  ranking:      {result['ranking']}")
    print(f"  best_method:  {result['best_method']}")
    print(f"  summary:")
    for method, scores in result['summary'].items():
        print(f"    {method}: RMSE={scores['rmse']}, MAE={scores['mae']}")

    assert 'methods'     in result
    assert 'ranking'     in result
    assert 'best_method' in result
    assert 'summary'     in result
    assert len(result['ranking']) == 4
    assert result['best_method'] in ['knn', 'mice', 'rf', 'mean']
    print("  PASSED")


def test_compare_ranking_is_sorted():
    print("\nTest 4b: ranking is sorted by RMSE (best first)")
    qm = ImputationQualityMetrics()
    df = get_complete_df()

    result = qm.compare_method_quality(df, methods=['knn', 'mean'])

    ranking = result['ranking']
    rmse_values = [result['methods'][m]['rmse'] for m in ranking]

    print(f"  ranking:     {ranking}")
    print(f"  rmse_values: {rmse_values}")

    assert rmse_values == sorted(rmse_values), "Ranking should be sorted by RMSE ascending"
    print("  PASSED")


def test_compare_with_cv():
    print("\nTest 4c: compare_method_quality with include_cv=True")
    qm = ImputationQualityMetrics()
    df = get_complete_df()

    result = qm.compare_method_quality(
        df,
        methods=['knn', 'mean'],
        include_cv=True
    )

    for method in ['knn', 'mean']:
        assert 'cross_validation' in result['methods'][method], \
            f"Missing cross_validation for {method}"
        cv = result['methods'][method]['cross_validation']
        assert 'cv_rmse_mean' in cv
        assert 'cv_mae_mean'  in cv
        print(f"  {method} CV RMSE: {cv['cv_rmse_mean']} ± {cv['cv_rmse_std']}")

    print("  PASSED")


# ------------------------------------------------------------------
# Test 5: Response structure (matches what API will return)
# ------------------------------------------------------------------

def test_api_response_structure():
    print("\nTest 5: quality_metrics response structure matches API spec")
    qm = ImputationQualityMetrics()
    df = get_test_df()

    result = qm.compare_method_quality(df, methods=['knn', 'mice', 'rf', 'mean'])

    # These are the exact keys the API will put in response['quality_metrics']
    assert 'methods'     in result
    assert 'ranking'     in result
    assert 'best_method' in result
    assert 'summary'     in result

    # Each method entry should have rmse, mae, per_column, n_evaluated
    for method_name, method_data in result['methods'].items():
        if 'error' not in method_data:
            assert 'rmse'        in method_data, f"Missing rmse for {method_name}"
            assert 'mae'         in method_data, f"Missing mae for {method_name}"
            assert 'per_column'  in method_data, f"Missing per_column for {method_name}"
            assert 'n_evaluated' in method_data, f"Missing n_evaluated for {method_name}"

    print(f"  best_method: {result['best_method']}")
    print(f"  ranking:     {result['ranking']}")
    print("  PASSED")


# ------------------------------------------------------------------
# Run all tests
# ------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Stage 7 Feature #4: Imputation Quality Metrics Tests")
    print("=" * 60)

    tests = [
        # Core metrics
        test_calculate_rmse,
        test_calculate_mae,
        test_perfect_prediction,
        test_categorical_accuracy,
        # Single method evaluation
        test_evaluate_method_knn,
        test_evaluate_method_mice,
        test_evaluate_method_rf,
        test_evaluate_method_mean,
        test_evaluate_on_df_with_missing,
        # Cross-validation
        test_cross_validate_knn,
        test_cross_validate_fold_scores_structure,
        # Method comparison
        test_compare_method_quality,
        test_compare_ranking_is_sorted,
        test_compare_with_cv,
        # API structure
        test_api_response_structure,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED - check output above")
    print("=" * 60)
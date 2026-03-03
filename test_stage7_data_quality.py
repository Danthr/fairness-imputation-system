"""
Test Suite - Stage 7 Feature #9: Data Quality Scoring
Tests completeness, validity, consistency, uniqueness, and overall scoring
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.imputation.data_quality_scorer import DataQualityScorer


# ------------------------------------------------------------------
# Shared test data
# ------------------------------------------------------------------

def get_clean_df():
    """Perfect dataset - no issues"""
    return pd.DataFrame({
        'age':       [25.0, 30.0, 35.0, 40.0, 45.0],
        'income':    [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        'gender':    ['M', 'F', 'M', 'F', 'M'],
        'education': ['BS', 'MS', 'BS', 'PhD', 'MS'],
    })


def get_missing_df():
    """Dataset with missing values"""
    return pd.DataFrame({
        'age':       [25.0, np.nan, 35.0, 45.0, np.nan],
        'income':    [50000.0, np.nan, 70000.0, np.nan, np.nan],
        'gender':    ['M', 'F', np.nan, 'M', 'F'],
        'education': ['BS', 'MS', 'BS', np.nan, 'PhD'],
    })


def get_invalid_df():
    """Dataset with invalid values (negative age/income)"""
    return pd.DataFrame({
        'age':    [25.0, -5.0, 35.0, 200.0, 45.0],
        'income': [50000.0, 60000.0, -1000.0, 80000.0, 90000.0],
    })


def get_duplicate_df():
    """Dataset with duplicate rows"""
    return pd.DataFrame({
        'age':    [25.0, 30.0, 25.0, 40.0, 25.0],
        'income': [50000.0, 60000.0, 50000.0, 80000.0, 50000.0],
        'gender': ['M', 'F', 'M', 'F', 'M'],
    })


def get_inconsistent_df():
    """Dataset with cross-column inconsistencies"""
    return pd.DataFrame({
        'age':       [25.0, 15.0, 35.0, 12.0, 45.0],
        'income':    [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        'gender':    ['M', 'F', 'M', 'F', 'M'],
        'education': ['BS', 'PhD', 'MS', 'MS', 'BS'],  # age 15 has PhD, age 12 has MS
    })


# ------------------------------------------------------------------
# Test 1: Completeness
# ------------------------------------------------------------------

def test_completeness_perfect():
    print("\nTest 1a: completeness - perfect dataset (no missing)")
    scorer = DataQualityScorer()
    result = scorer.score_completeness(get_clean_df())

    print(f"  score:         {result['score']}")
    print(f"  missing_cells: {result['missing_cells']}")
    print(f"  grade:         {result['grade']}")

    assert result['score'] == 1.0, "Perfect dataset should score 1.0"
    assert result['missing_cells'] == 0
    assert result['grade'] == 'A'
    print("  PASSED")


def test_completeness_with_missing():
    print("\nTest 1b: completeness - dataset with missing values")
    scorer = DataQualityScorer()
    result = scorer.score_completeness(get_missing_df())

    print(f"  score:         {result['score']}")
    print(f"  missing_cells: {result['missing_cells']}")
    print(f"  missing_pct:   {result['missing_pct']}%")
    print(f"  grade:         {result['grade']}")

    assert result['score'] < 1.0, "Should score below 1.0 with missing values"
    assert result['missing_cells'] > 0
    assert 'per_column' in result
    print("  PASSED")


def test_completeness_per_column():
    print("\nTest 1c: completeness - per column breakdown")
    scorer = DataQualityScorer()
    result = scorer.score_completeness(get_missing_df())

    for col, stats in result['per_column'].items():
        print(f"  {col}: missing={stats['missing']}, score={stats['score']}")
        assert 'missing'  in stats
        assert 'present'  in stats
        assert 'score'    in stats
        assert 0.0 <= stats['score'] <= 1.0

    print("  PASSED")


# ------------------------------------------------------------------
# Test 2: Validity
# ------------------------------------------------------------------

def test_validity_perfect():
    print("\nTest 2a: validity - clean dataset (no invalid values)")
    scorer = DataQualityScorer()
    result = scorer.score_validity(get_clean_df())

    print(f"  score:         {result['score']}")
    print(f"  total_invalid: {result['total_invalid']}")
    print(f"  grade:         {result['grade']}")

    assert result['score'] == 1.0, "Clean dataset should score 1.0"
    assert result['total_invalid'] == 0
    print("  PASSED")


def test_validity_with_invalid():
    print("\nTest 2b: validity - dataset with negative age/income")
    scorer = DataQualityScorer()
    result = scorer.score_validity(get_invalid_df())

    print(f"  score:         {result['score']}")
    print(f"  total_invalid: {result['total_invalid']}")
    print(f"  grade:         {result['grade']}")
    for col, stats in result['per_column'].items():
        print(f"  {col}: invalid={stats['n_invalid']}, violations={stats['violations']}")

    assert result['score'] < 1.0, "Should detect invalid values"
    assert result['total_invalid'] > 0
    print("  PASSED")


def test_validity_custom_rules():
    print("\nTest 2c: validity - custom rules")
    custom_rules = {
        'age':    {'min': 18, 'max': 65},
        'income': {'min': 0, 'max': 500000},
    }
    scorer = DataQualityScorer(validity_rules=custom_rules)
    df = pd.DataFrame({
        'age':    [25.0, 15.0, 70.0, 40.0],   # 15 and 70 are out of range
        'income': [50000.0, 60000.0, 70000.0, -100.0],  # -100 invalid
    })
    result = scorer.score_validity(df)

    print(f"  score:         {result['score']}")
    print(f"  total_invalid: {result['total_invalid']}")
    for col, stats in result['per_column'].items():
        print(f"  {col}: invalid={stats['n_invalid']}, violations={stats['violations']}")

    assert result['total_invalid'] == 3, f"Expected 3 invalid, got {result['total_invalid']}"
    print("  PASSED")


# ------------------------------------------------------------------
# Test 3: Consistency
# ------------------------------------------------------------------

def test_consistency_perfect():
    print("\nTest 3a: consistency - clean dataset (no contradictions)")
    scorer = DataQualityScorer()
    result = scorer.score_consistency(get_clean_df())

    print(f"  score:        {result['score']}")
    print(f"  n_violations: {result['n_violations']}")
    print(f"  grade:        {result['grade']}")

    assert result['score'] == 1.0
    assert result['n_violations'] == 0
    print("  PASSED")


def test_consistency_with_violations():
    print("\nTest 3b: consistency - age/education contradictions")
    scorer = DataQualityScorer()
    result = scorer.score_consistency(get_inconsistent_df())

    print(f"  score:        {result['score']}")
    print(f"  n_violations: {result['n_violations']}")
    print(f"  violations:   {result['violations']}")
    print(f"  grade:        {result['grade']}")

    assert result['score'] < 1.0, "Should detect inconsistencies"
    assert result['n_violations'] > 0
    print("  PASSED")


def test_consistency_custom_rules():
    print("\nTest 3c: consistency - custom rules")
    custom_rules = [
        {
            'if_col':      'age',
            'if_max':      18,
            'then_col':    'education',
            'then_not_in': ['PhD', 'MS'],
            'description': 'Under 18 should not have advanced degree',
        }
    ]
    scorer = DataQualityScorer(consistency_rules=custom_rules)
    result = scorer.score_consistency(get_inconsistent_df())

    print(f"  score:        {result['score']}")
    print(f"  n_violations: {result['n_violations']}")

    assert result['n_violations'] > 0
    print("  PASSED")


def test_consistency_no_applicable_columns():
    print("\nTest 3d: consistency - df with no age/education columns")
    scorer = DataQualityScorer()
    df = pd.DataFrame({
        'price':    [100.0, 200.0, 300.0],
        'quantity': [1, 2, 3],
    })
    result = scorer.score_consistency(df)

    print(f"  score: {result['score']}")
    print(f"  note:  {result.get('note', '')}")

    assert result['score'] == 1.0, "No rules = no violations = perfect score"
    print("  PASSED")


# ------------------------------------------------------------------
# Test 4: Uniqueness
# ------------------------------------------------------------------

def test_uniqueness_perfect():
    print("\nTest 4a: uniqueness - no duplicates")
    scorer = DataQualityScorer()
    result = scorer.score_uniqueness(get_clean_df())

    print(f"  score:          {result['score']}")
    print(f"  duplicate_rows: {result['duplicate_rows']}")
    print(f"  grade:          {result['grade']}")

    assert result['score'] == 1.0
    assert result['duplicate_rows'] == 0
    print("  PASSED")


def test_uniqueness_with_duplicates():
    print("\nTest 4b: uniqueness - dataset with duplicates")
    scorer = DataQualityScorer()
    result = scorer.score_uniqueness(get_duplicate_df())

    print(f"  score:             {result['score']}")
    print(f"  duplicate_rows:    {result['duplicate_rows']}")
    print(f"  duplicate_pct:     {result['duplicate_pct']}%")
    print(f"  duplicate_indices: {result['duplicate_indices']}")
    print(f"  grade:             {result['grade']}")

    assert result['score'] < 1.0
    assert result['duplicate_rows'] == 2   # rows 2 and 4 are duplicates of row 0
    assert result['duplicate_pct'] == 40.0
    print("  PASSED")


# ------------------------------------------------------------------
# Test 5: Overall score_all
# ------------------------------------------------------------------

def test_score_all_clean():
    print("\nTest 5a: score_all - perfect dataset")
    scorer = DataQualityScorer()
    result = scorer.score_all(get_clean_df())

    print(f"  overall_score: {result['overall_score']}")
    print(f"  overall_grade: {result['overall_grade']}")
    print(f"  recommendation: {result['recommendation']}")
    print(f"  dimension_scores: {result['dimension_scores']}")

    assert result['overall_score'] == 1.0
    assert result['overall_grade'] == 'A'
    assert 'completeness' in result
    assert 'validity'     in result
    assert 'consistency'  in result
    assert 'uniqueness'   in result
    assert 'dimension_scores' in result
    assert 'recommendation'   in result
    print("  PASSED")


def test_score_all_with_issues():
    print("\nTest 5b: score_all - dataset with missing values")
    scorer = DataQualityScorer()
    result = scorer.score_all(get_missing_df())

    print(f"  overall_score:    {result['overall_score']}")
    print(f"  overall_grade:    {result['overall_grade']}")
    print(f"  recommendation:   {result['recommendation']}")
    print(f"  dimension_scores: {result['dimension_scores']}")

    assert result['overall_score'] < 1.0
    assert result['dimension_scores']['completeness'] < 1.0
    print("  PASSED")


def test_score_all_dimension_scores_structure():
    print("\nTest 5c: score_all - dimension_scores structure")
    scorer = DataQualityScorer()
    result = scorer.score_all(get_clean_df())

    dims = result['dimension_scores']
    for dim in ['completeness', 'validity', 'consistency', 'uniqueness']:
        assert dim in dims, f"Missing dimension: {dim}"
        assert 0.0 <= dims[dim] <= 1.0, f"{dim} score out of range"
        print(f"  {dim}: {dims[dim]}")

    print("  PASSED")


def test_score_all_weighted():
    print("\nTest 5d: score_all - custom weights")
    custom_weights = {
        'completeness': 0.50,
        'validity':     0.20,
        'consistency':  0.20,
        'uniqueness':   0.10,
    }
    scorer = DataQualityScorer(weights=custom_weights)
    result = scorer.score_all(get_missing_df())

    print(f"  overall_score: {result['overall_score']}")
    print(f"  weights_used:  {result['weights_used']}")

    assert result['weights_used'] == custom_weights
    assert 0.0 <= result['overall_score'] <= 1.0
    print("  PASSED")


def test_invalid_weights_raise_error():
    print("\nTest 5e: invalid weights should raise ValueError")
    try:
        DataQualityScorer(weights={
            'completeness': 0.50,
            'validity':     0.50,
            'consistency':  0.50,   # sum > 1.0
            'uniqueness':   0.50,
        })
        print("  FAILED - should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")
        print("  PASSED")


# ------------------------------------------------------------------
# Test 6: Grade helper
# ------------------------------------------------------------------

def test_grade_boundaries():
    print("\nTest 6: grade boundaries")
    scorer = DataQualityScorer()

    cases = [
        (1.00, 'A'),
        (0.95, 'A'),
        (0.90, 'B'),
        (0.85, 'B'),
        (0.75, 'C'),
        (0.70, 'C'),
        (0.60, 'D'),
        (0.50, 'D'),
        (0.49, 'F'),
        (0.00, 'F'),
    ]
    for score, expected_grade in cases:
        grade = scorer._grade(score)
        print(f"  score={score} → grade={grade} (expected {expected_grade})")
        assert grade == expected_grade, f"Score {score}: expected {expected_grade}, got {grade}"

    print("  PASSED")


# ------------------------------------------------------------------
# Test 7: Recommendation text
# ------------------------------------------------------------------

def test_recommendation_good_data():
    print("\nTest 7a: recommendation - good data")
    scorer = DataQualityScorer()
    result = scorer.score_all(get_clean_df())

    print(f"  recommendation: {result['recommendation']}")
    assert 'safe to proceed' in result['recommendation'].lower()
    print("  PASSED")


def test_recommendation_bad_data():
    print("\nTest 7b: recommendation - data with missing values")
    scorer = DataQualityScorer()
    result = scorer.score_all(get_missing_df())

    print(f"  recommendation: {result['recommendation']}")
    assert 'issues found' in result['recommendation'].lower()
    print("  PASSED")


# ------------------------------------------------------------------
# Run all tests
# ------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Stage 7 Feature #9: Data Quality Scoring Tests")
    print("=" * 60)

    tests = [
        # Completeness
        test_completeness_perfect,
        test_completeness_with_missing,
        test_completeness_per_column,
        # Validity
        test_validity_perfect,
        test_validity_with_invalid,
        test_validity_custom_rules,
        # Consistency
        test_consistency_perfect,
        test_consistency_with_violations,
        test_consistency_custom_rules,
        test_consistency_no_applicable_columns,
        # Uniqueness
        test_uniqueness_perfect,
        test_uniqueness_with_duplicates,
        # Overall
        test_score_all_clean,
        test_score_all_with_issues,
        test_score_all_dimension_scores_structure,
        test_score_all_weighted,
        test_invalid_weights_raise_error,
        # Grade
        test_grade_boundaries,
        # Recommendation
        test_recommendation_good_data,
        test_recommendation_bad_data,
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
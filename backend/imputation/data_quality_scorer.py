"""
Data Quality Scorer - Stage 7 Feature #9
Scores a dataset across 4 dimensions before imputation:
  - Completeness  (missing values)
  - Validity      (out-of-range / invalid values)
  - Consistency   (cross-column contradictions)
  - Uniqueness    (duplicate rows)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityScorer:
    """
    Scores a dataset across 4 quality dimensions.
    Each dimension returns a score between 0.0 (worst) and 1.0 (best).
    An overall weighted score is also calculated.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        validity_rules: Optional[Dict[str, Dict]] = None,
        consistency_rules: Optional[List[Dict]] = None,
    ):
        """
        Args:
            weights: Custom weights for each dimension.
                     Default: completeness=0.35, validity=0.25,
                              consistency=0.25, uniqueness=0.15
            validity_rules: Per-column min/max rules.
                     Example: {'age': {'min': 0, 'max': 120},
                               'income': {'min': 0}}
            consistency_rules: Cross-column rules.
                     Example: [{'if_col': 'age', 'if_max': 18,
                                'then_col': 'education', 'then_not': 'PhD'}]
        """
        self.weights = weights or {
            'completeness':  0.35,
            'validity':      0.25,
            'consistency':   0.25,
            'uniqueness':    0.15,
        }
        self.validity_rules    = validity_rules or {}
        self.consistency_rules = consistency_rules or []

        # Validate weights sum to 1.0
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    # ------------------------------------------------------------------
    # Dimension 1: Completeness
    # ------------------------------------------------------------------

    def score_completeness(self, df: pd.DataFrame) -> Dict:
        """
        Measures how complete the dataset is (no missing values).

        Score = total non-missing cells / total cells
        Per-column breakdown also provided.
        """
        total_cells   = df.shape[0] * df.shape[1]
        missing_cells = int(df.isnull().sum().sum())
        present_cells = total_cells - missing_cells

        overall_score = round(present_cells / total_cells, 4) if total_cells > 0 else 0.0

        per_column = {}
        for col in df.columns:
            n_missing = int(df[col].isnull().sum())
            n_total   = len(df)
            per_column[col] = {
                'missing':  n_missing,
                'present':  n_total - n_missing,
                'score':    round(1.0 - (n_missing / n_total), 4) if n_total > 0 else 0.0,
            }

        return {
            'score':         overall_score,
            'total_cells':   total_cells,
            'missing_cells': missing_cells,
            'missing_pct':   round(missing_cells / total_cells * 100, 2) if total_cells > 0 else 0.0,
            'per_column':    per_column,
            'grade':         self._grade(overall_score),
        }

    # ------------------------------------------------------------------
    # Dimension 2: Validity
    # ------------------------------------------------------------------

    def score_validity(self, df: pd.DataFrame) -> Dict:
        """
        Measures how many values fall within expected ranges/types.

        If no validity_rules are provided, auto-detects rules for
        numerical columns (flags negative values for columns where
        negatives are unlikely e.g. age, income, count-like columns).

        Score = valid cells / total non-missing cells
        """
        rules = self.validity_rules or self._auto_detect_validity_rules(df)

        total_checked  = 0
        total_invalid  = 0
        per_column: Dict[str, Dict] = {}

        for col, rule in rules.items():
            if col not in df.columns:
                continue

            col_data   = df[col].dropna()
            n_checked  = len(col_data)
            n_invalid  = 0
            violations = []

            if not pd.api.types.is_numeric_dtype(df[col]):
                # For categorical: check against allowed values if provided
                if 'allowed_values' in rule:
                    invalid_mask = ~col_data.isin(rule['allowed_values'])
                    n_invalid    = int(invalid_mask.sum())
                    if n_invalid > 0:
                        violations.append(
                            f"{n_invalid} values not in allowed set"
                        )
            else:
                # Numerical range checks
                if 'min' in rule:
                    below = col_data < rule['min']
                    n_below = int(below.sum())
                    if n_below > 0:
                        n_invalid += n_below
                        violations.append(f"{n_below} values below min ({rule['min']})")

                if 'max' in rule:
                    above = col_data > rule['max']
                    n_above = int(above.sum())
                    if n_above > 0:
                        n_invalid += n_above
                        violations.append(f"{n_above} values above max ({rule['max']})")

            col_score = round(1.0 - (n_invalid / n_checked), 4) if n_checked > 0 else 1.0

            per_column[col] = {
                'n_checked':  n_checked,
                'n_invalid':  n_invalid,
                'score':      col_score,
                'violations': violations,
                'rule_applied': rule,
            }

            total_checked += n_checked
            total_invalid += n_invalid

        overall_score = (
            round(1.0 - (total_invalid / total_checked), 4)
            if total_checked > 0 else 1.0
        )

        return {
            'score':          overall_score,
            'total_checked':  total_checked,
            'total_invalid':  total_invalid,
            'rules_applied':  rules,
            'per_column':     per_column,
            'grade':          self._grade(overall_score),
        }

    def _auto_detect_validity_rules(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Auto-generate validity rules for numerical columns.
        Flags columns whose name suggests non-negative values.
        """
        non_negative_keywords = [
            'age', 'income', 'salary', 'count', 'price',
            'score', 'rate', 'amount', 'years', 'weight', 'height'
        ]
        rules = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in non_negative_keywords):
                rules[col] = {'min': 0}
        return rules

    # ------------------------------------------------------------------
    # Dimension 3: Consistency
    # ------------------------------------------------------------------

    def score_consistency(self, df: pd.DataFrame) -> Dict:
        """
        Measures cross-column logical consistency.

        If no consistency_rules are provided, auto-detects basic
        contradictions (e.g. age < 18 but education = 'PhD' or 'MS').

        Score = consistent rows / total rows checked
        """
        rules = self.consistency_rules or self._auto_detect_consistency_rules(df)

        if not rules:
            return {
                'score':       1.0,
                'grade':       self._grade(1.0),
                'n_violations': 0,
                'violations':  [],
                'note':        'No consistency rules applicable to this dataset',
            }

        total_rows      = len(df)
        all_violations  = []

        for rule in rules:
            if_col   = rule.get('if_col')
            then_col = rule.get('then_col')

            if if_col not in df.columns or then_col not in df.columns:
                continue

            # Build the condition mask for the "if" side
            condition_mask = pd.Series([True] * total_rows, index=df.index)

            if 'if_min' in rule:
                condition_mask &= df[if_col].notna() & (df[if_col] >= rule['if_min'])
            if 'if_max' in rule:
                condition_mask &= df[if_col].notna() & (df[if_col] <= rule['if_max'])
            if 'if_equals' in rule:
                condition_mask &= df[if_col].notna() & (df[if_col] == rule['if_equals'])

            # Build the violation mask for the "then" side
            violation_mask = pd.Series([False] * total_rows, index=df.index)

            if 'then_not' in rule:
                violation_mask = (
                    condition_mask &
                    df[then_col].notna() &
                    (df[then_col] == rule['then_not'])
                )
            if 'then_not_in' in rule:
                violation_mask = (
                    condition_mask &
                    df[then_col].notna() &
                    df[then_col].isin(rule['then_not_in'])
                )

            n_violations = int(violation_mask.sum())
            if n_violations > 0:
                all_violations.append({
                    'rule':        rule,
                    'n_violated':  n_violations,
                    'row_indices': df.index[violation_mask].tolist(),
                })

        total_violations = sum(v['n_violated'] for v in all_violations)
        overall_score = round(
            1.0 - (total_violations / total_rows), 4
        ) if total_rows > 0 else 1.0
        overall_score = max(0.0, overall_score)

        return {
            'score':        overall_score,
            'grade':        self._grade(overall_score),
            'n_violations': total_violations,
            'violations':   all_violations,
            'rules_applied': rules,
        }

    def _auto_detect_consistency_rules(self, df: pd.DataFrame) -> List[Dict]:
        """
        Auto-generate consistency rules based on common column name patterns.
        """
        rules = []
        cols_lower = {col.lower(): col for col in df.columns}

        # Rule: if age < 18, education should not be PhD or MS
        if 'age' in cols_lower and 'education' in cols_lower:
            rules.append({
                'if_col':      cols_lower['age'],
                'if_max':      18,
                'then_col':    cols_lower['education'],
                'then_not_in': ['PhD', 'MS', 'Masters'],
                'description': 'Age < 18 should not have advanced degrees',
            })

        # Rule: if age < 16, income should not be positive
        if 'age' in cols_lower and 'income' in cols_lower:
            rules.append({
                'if_col':      cols_lower['age'],
                'if_max':      16,
                'then_col':    cols_lower['income'],
                'then_not_in': [],  # handled differently below
                'description': 'Age < 16 unlikely to have income',
            })

        return rules

    # ------------------------------------------------------------------
    # Dimension 4: Uniqueness
    # ------------------------------------------------------------------

    def score_uniqueness(self, df: pd.DataFrame) -> Dict:
        """
        Measures how many rows are unique (no duplicates).

        Score = unique rows / total rows
        """
        total_rows     = len(df)
        duplicate_mask = df.duplicated(keep='first')
        n_duplicates   = int(duplicate_mask.sum())
        n_unique       = total_rows - n_duplicates

        overall_score = round(n_unique / total_rows, 4) if total_rows > 0 else 1.0

        duplicate_indices = df.index[duplicate_mask].tolist()

        return {
            'score':             overall_score,
            'total_rows':        total_rows,
            'unique_rows':       n_unique,
            'duplicate_rows':    n_duplicates,
            'duplicate_pct':     round(n_duplicates / total_rows * 100, 2) if total_rows > 0 else 0.0,
            'duplicate_indices': duplicate_indices,
            'grade':             self._grade(overall_score),
        }

    # ------------------------------------------------------------------
    # Overall score
    # ------------------------------------------------------------------

    def score_all(self, df: pd.DataFrame) -> Dict:
        """
        Run all 4 dimensions and return a combined quality report.

        Returns:
            {
                'overall_score':  0.87,
                'overall_grade':  'B',
                'completeness':   {...},
                'validity':       {...},
                'consistency':    {...},
                'uniqueness':     {...},
                'weights_used':   {...},
                'recommendation': '...',
            }
        """
        logger.info("Scoring completeness...")
        completeness = self.score_completeness(df)

        logger.info("Scoring validity...")
        validity = self.score_validity(df)

        logger.info("Scoring consistency...")
        consistency = self.score_consistency(df)

        logger.info("Scoring uniqueness...")
        uniqueness = self.score_uniqueness(df)

        # Weighted overall score
        overall = (
            self.weights['completeness'] * completeness['score'] +
            self.weights['validity']     * validity['score']     +
            self.weights['consistency']  * consistency['score']  +
            self.weights['uniqueness']   * uniqueness['score']
        )
        overall = round(overall, 4)

        return {
            'overall_score':  overall,
            'overall_grade':  self._grade(overall),
            'completeness':   completeness,
            'validity':       validity,
            'consistency':    consistency,
            'uniqueness':     uniqueness,
            'weights_used':   self.weights,
            'recommendation': self._recommend(overall, completeness, validity, consistency, uniqueness),
            'dimension_scores': {
                'completeness': completeness['score'],
                'validity':     validity['score'],
                'consistency':  consistency['score'],
                'uniqueness':   uniqueness['score'],
            }
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _grade(self, score: float) -> str:
        """Convert 0-1 score to letter grade."""
        if score >= 0.95:
            return 'A'
        elif score >= 0.85:
            return 'B'
        elif score >= 0.70:
            return 'C'
        elif score >= 0.50:
            return 'D'
        else:
            return 'F'

    def _recommend(
        self,
        overall: float,
        completeness: Dict,
        validity: Dict,
        consistency: Dict,
        uniqueness: Dict,
    ) -> str:
        """Generate a human-readable recommendation based on scores."""
        issues = []

        if completeness['score'] < 0.80:
            pct = completeness['missing_pct']
            issues.append(f"high missing rate ({pct}%) — imputation strongly recommended")

        if validity['score'] < 0.90:
            issues.append(
                f"validity issues detected ({validity['total_invalid']} invalid values) "
                f"— review and clean before imputing"
            )

        if consistency['score'] < 0.95:
            issues.append(
                f"cross-column inconsistencies found ({consistency['n_violations']} violations) "
                f"— investigate before imputing"
            )

        if uniqueness['score'] < 0.95:
            pct = uniqueness['duplicate_pct']
            issues.append(
                f"duplicate rows detected ({pct}%) "
                f"— deduplicate before imputing to avoid bias"
            )

        if not issues:
            return "Data quality is good. Safe to proceed with imputation."

        return "Issues found: " + "; ".join(issues) + "."
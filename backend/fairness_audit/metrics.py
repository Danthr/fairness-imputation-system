"""
Fairness Metrics
Implements various fairness metric calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FairnessMetrics:
    """Calculates fairness metrics for bias detection"""
    
    def __init__(self):
        self.metrics_results = {}
    
    def disparate_impact(
        self, 
        df: pd.DataFrame, 
        protected_attr: str, 
        outcome_attr: str,
        favorable_outcome: any,
        privileged_group: any,
        unprivileged_group: any
    ) -> Dict:
        """
        Calculate Disparate Impact (4/5ths rule)
        
        Disparate Impact = P(favorable outcome | unprivileged) / P(favorable outcome | privileged)
        Fair if DI >= 0.8 (4/5ths rule)
        
        Args:
            df: Dataset
            protected_attr: Protected attribute column (e.g., 'gender')
            outcome_attr: Outcome column (e.g., 'income', 'hired')
            favorable_outcome: Value considered favorable
            privileged_group: Privileged group value
            unprivileged_group: Unprivileged group value
            
        Returns:
            Dictionary with disparate impact score and interpretation
        """
        logger.info(f"Calculating Disparate Impact for {protected_attr}")
        
        # Get favorable outcome rates for each group
        privileged_mask = df[protected_attr] == privileged_group
        unprivileged_mask = df[protected_attr] == unprivileged_group
        
        # Calculate favorable outcome rates
        privileged_favorable = (df[privileged_mask][outcome_attr] == favorable_outcome).sum()
        privileged_total = privileged_mask.sum()
        
        unprivileged_favorable = (df[unprivileged_mask][outcome_attr] == favorable_outcome).sum()
        unprivileged_total = unprivileged_mask.sum()
        
        # Calculate rates
        privileged_rate = privileged_favorable / privileged_total if privileged_total > 0 else 0
        unprivileged_rate = unprivileged_favorable / unprivileged_total if unprivileged_total > 0 else 0
        
        # Calculate disparate impact
        di_score = unprivileged_rate / privileged_rate if privileged_rate > 0 else 0
        
        # Interpretation
        is_fair = di_score >= 0.8
        interpretation = "Fair (passes 4/5ths rule)" if is_fair else "Potential bias detected"
        
        result = {
            'metric': 'Disparate Impact',
            'protected_attribute': protected_attr,
            'privileged_group': privileged_group,
            'unprivileged_group': unprivileged_group,
            'privileged_rate': round(privileged_rate, 4),
            'unprivileged_rate': round(unprivileged_rate, 4),
            'disparate_impact_score': round(di_score, 4),
            'threshold': 0.8,
            'is_fair': is_fair,
            'interpretation': interpretation
        }
        
        return result
    
    def demographic_parity(
        self,
        df: pd.DataFrame,
        protected_attr: str,
        outcome_attr: str,
        favorable_outcome: any
    ) -> Dict:
        """
        Calculate Demographic Parity Difference
        
        Demographic Parity = |P(favorable | group A) - P(favorable | group B)|
        Fair if difference is close to 0 (typically < 0.1)
        
        Args:
            df: Dataset
            protected_attr: Protected attribute column
            outcome_attr: Outcome column
            favorable_outcome: Value considered favorable
            
        Returns:
            Dictionary with demographic parity metrics
        """
        logger.info(f"Calculating Demographic Parity for {protected_attr}")
        
        groups = df[protected_attr].unique()
        group_rates = {}
        
        for group in groups:
            group_mask = df[protected_attr] == group
            favorable_count = (df[group_mask][outcome_attr] == favorable_outcome).sum()
            total_count = group_mask.sum()
            rate = favorable_count / total_count if total_count > 0 else 0
            group_rates[str(group)] = round(rate, 4)
        
        # Calculate max difference
        rates = list(group_rates.values())
        max_diff = max(rates) - min(rates) if rates else 0
        
        is_fair = max_diff < 0.1
        interpretation = "Fair (demographic parity achieved)" if is_fair else "Disparity detected between groups"
        
        result = {
            'metric': 'Demographic Parity',
            'protected_attribute': protected_attr,
            'group_rates': group_rates,
            'max_difference': round(max_diff, 4),
            'threshold': 0.1,
            'is_fair': is_fair,
            'interpretation': interpretation
        }
        
        return result
    
    def equal_opportunity(
        self,
        df: pd.DataFrame,
        protected_attr: str,
        outcome_attr: str,
        true_label_attr: str,
        favorable_outcome: any,
        favorable_label: any
    ) -> Dict:
        """
        Calculate Equal Opportunity Difference
        
        Measures difference in True Positive Rates between groups
        Equal Opportunity = |TPR(group A) - TPR(group B)|
        
        Args:
            df: Dataset
            protected_attr: Protected attribute column
            outcome_attr: Predicted outcome column
            true_label_attr: True label column
            favorable_outcome: Favorable predicted outcome
            favorable_label: Favorable true label
            
        Returns:
            Dictionary with equal opportunity metrics
        """
        logger.info(f"Calculating Equal Opportunity for {protected_attr}")
        
        groups = df[protected_attr].unique()
        tpr_by_group = {}
        
        for group in groups:
            group_mask = df[protected_attr] == group
            # True positives: correctly predicted favorable outcomes
            true_positive_mask = (
                (df[outcome_attr] == favorable_outcome) & 
                (df[true_label_attr] == favorable_label) & 
                group_mask
            )
            # All actual positives in this group
            actual_positive_mask = (df[true_label_attr] == favorable_label) & group_mask
            
            tp_count = true_positive_mask.sum()
            ap_count = actual_positive_mask.sum()
            
            tpr = tp_count / ap_count if ap_count > 0 else 0
            tpr_by_group[str(group)] = round(tpr, 4)
        
        # Calculate max difference
        tprs = list(tpr_by_group.values())
        max_diff = max(tprs) - min(tprs) if tprs else 0
        
        is_fair = max_diff < 0.1
        interpretation = "Fair (equal opportunity achieved)" if is_fair else "Opportunity disparity detected"
        
        result = {
            'metric': 'Equal Opportunity',
            'protected_attribute': protected_attr,
            'true_positive_rates': tpr_by_group,
            'max_difference': round(max_diff, 4),
            'threshold': 0.1,
            'is_fair': is_fair,
            'interpretation': interpretation
        }
        
        return result
    
    def statistical_parity_difference(
        self,
        df: pd.DataFrame,
        protected_attr: str,
        outcome_attr: str,
        favorable_outcome: any,
        reference_group: any
    ) -> Dict:
        """
        Calculate Statistical Parity Difference relative to reference group
        
        SPD = P(favorable | protected group) - P(favorable | reference group)
        Fair if SPD is close to 0
        
        Args:
            df: Dataset
            protected_attr: Protected attribute column
            outcome_attr: Outcome column
            favorable_outcome: Favorable outcome value
            reference_group: Reference group value
            
        Returns:
            Dictionary with statistical parity metrics
        """
        logger.info(f"Calculating Statistical Parity Difference for {protected_attr}")
        
        # Reference group rate
        ref_mask = df[protected_attr] == reference_group
        ref_favorable = (df[ref_mask][outcome_attr] == favorable_outcome).sum()
        ref_total = ref_mask.sum()
        ref_rate = ref_favorable / ref_total if ref_total > 0 else 0
        
        # Calculate SPD for all groups
        groups = df[protected_attr].unique()
        spd_by_group = {}
        
        for group in groups:
            if group == reference_group:
                continue
            
            group_mask = df[protected_attr] == group
            group_favorable = (df[group_mask][outcome_attr] == favorable_outcome).sum()
            group_total = group_mask.sum()
            group_rate = group_favorable / group_total if group_total > 0 else 0
            
            spd = group_rate - ref_rate
            spd_by_group[str(group)] = round(spd, 4)
        
        result = {
            'metric': 'Statistical Parity Difference',
            'protected_attribute': protected_attr,
            'reference_group': reference_group,
            'reference_rate': round(ref_rate, 4),
            'spd_by_group': spd_by_group,
            'interpretation': 'Values close to 0 indicate fairness'
        }
        
        return result
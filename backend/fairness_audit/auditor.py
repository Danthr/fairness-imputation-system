"""
Fairness Auditor
Main engine for auditing datasets for algorithmic bias
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from .metrics import FairnessMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FairnessAuditor:
    """Audits datasets for fairness and bias"""
    
    def __init__(self):
        self.metrics = FairnessMetrics()
        self.audit_results = {}
    
    def detect_protected_attributes(self, df: pd.DataFrame) -> List[str]:
        """
        Automatically detect protected attributes in dataset
        
        Returns:
            List of column names that are likely protected attributes
        """
        protected_keywords = [
            'gender', 'sex', 'race', 'ethnicity', 'age', 
            'religion', 'disability', 'marital', 'nationality', 'color'
        ]
        
        protected_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in protected_keywords):
                protected_cols.append(col)
        
        logger.info(f"Detected {len(protected_cols)} protected attributes: {protected_cols}")
        return protected_cols
    
    def audit_simple(
        self,
        df: pd.DataFrame,
        protected_attr: str,
        outcome_attr: str,
        favorable_outcome: any = None
    ) -> Dict:
        """
        Simple fairness audit for datasets without true labels
        
        Args:
            df: Dataset to audit
            protected_attr: Protected attribute column (e.g., 'gender')
            outcome_attr: Outcome column (e.g., 'income')
            favorable_outcome: What counts as favorable (auto-detect if None)
            
        Returns:
            Dictionary with audit results
        """
        logger.info(f"Starting simple fairness audit on {protected_attr}")
        
        # Auto-detect favorable outcome if not provided
        if favorable_outcome is None:
            if pd.api.types.is_numeric_dtype(df[outcome_attr]):
                # For numeric: use median as threshold
                median_val = df[outcome_attr].median()
                df['_favorable'] = df[outcome_attr] > median_val
                favorable_outcome = True
                outcome_attr_temp = '_favorable'
            else:
                # For categorical: use most common value
                favorable_outcome = df[outcome_attr].mode()[0]
                outcome_attr_temp = outcome_attr
        else:
            outcome_attr_temp = outcome_attr
        
        # Get unique groups
        groups = df[protected_attr].unique()
        
        if len(groups) < 2:
            return {
                'error': 'Need at least 2 groups in protected attribute',
                'protected_attribute': protected_attr
            }
        
        # Determine privileged/unprivileged groups (use first two groups)
        privileged_group = groups[0]
        unprivileged_group = groups[1]
        
        # Calculate metrics
        results = {
            'protected_attribute': protected_attr,
            'outcome_attribute': outcome_attr,
            'favorable_outcome': str(favorable_outcome),
            'total_samples': len(df),
            'groups_detected': [str(g) for g in groups],
            'metrics': {}
        }
        
        # Disparate Impact
        di_result = self.metrics.disparate_impact(
            df, protected_attr, outcome_attr_temp, favorable_outcome,
            privileged_group, unprivileged_group
        )
        results['metrics']['disparate_impact'] = di_result
        
        # Demographic Parity
        dp_result = self.metrics.demographic_parity(
            df, protected_attr, outcome_attr_temp, favorable_outcome
        )
        results['metrics']['demographic_parity'] = dp_result
        
        # Statistical Parity
        sp_result = self.metrics.statistical_parity_difference(
            df, protected_attr, outcome_attr_temp, favorable_outcome, privileged_group
        )
        results['metrics']['statistical_parity'] = sp_result
        
        # Overall assessment
        fair_count = sum([
            di_result['is_fair'],
            dp_result['is_fair']
        ])
        
        results['overall_assessment'] = {
            'metrics_passed': fair_count,
            'total_metrics': 2,
            'is_fair': fair_count >= 1,
            'summary': 'Dataset passes fairness checks' if fair_count >= 1 else 'Potential bias detected'
        }
        
        # Clean up temporary column
        if '_favorable' in df.columns:
            df.drop('_favorable', axis=1, inplace=True)
        
        logger.info(f"Audit complete: {fair_count}/2 metrics passed")
        
        return results
    
    def audit_multiple_attributes(
        self,
        df: pd.DataFrame,
        protected_attrs: List[str],
        outcome_attr: str,
        favorable_outcome: any = None
    ) -> Dict:
        """
        Audit multiple protected attributes
        
        Args:
            df: Dataset
            protected_attrs: List of protected attribute columns
            outcome_attr: Outcome column
            favorable_outcome: Favorable outcome value
            
        Returns:
            Dictionary with results for each protected attribute
        """
        logger.info(f"Auditing {len(protected_attrs)} protected attributes")
        
        results = {
            'total_attributes_audited': len(protected_attrs),
            'outcome_attribute': outcome_attr,
            'audits': {}
        }
        
        for attr in protected_attrs:
            logger.info(f"Auditing attribute: {attr}")
            audit_result = self.audit_simple(df, attr, outcome_attr, favorable_outcome)
            results['audits'][attr] = audit_result
        
        # Summary across all attributes
        total_fair = sum(
            1 for audit in results['audits'].values() 
            if audit.get('overall_assessment', {}).get('is_fair', False)
        )
        
        results['overall_summary'] = {
            'fair_attributes': total_fair,
            'total_attributes': len(protected_attrs),
            'overall_fairness': 'Fair' if total_fair == len(protected_attrs) else 'Bias detected'
        }
        
        return results
    
    def generate_audit_report(self, audit_results: Dict) -> str:
        """
        Generate human-readable audit report
        
        Args:
            audit_results: Results from audit_simple or audit_multiple_attributes
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*60)
        report.append("FAIRNESS AUDIT REPORT")
        report.append("="*60)
        
        if 'audits' in audit_results:
            # Multiple attributes
            report.append(f"\nAttributes Audited: {audit_results['total_attributes_audited']}")
            report.append(f"Outcome Attribute: {audit_results['outcome_attribute']}")
            report.append(f"\nOverall: {audit_results['overall_summary']['overall_fairness']}")
            report.append(f"Fair Attributes: {audit_results['overall_summary']['fair_attributes']}/{audit_results['overall_summary']['total_attributes']}")
            
            for attr, result in audit_results['audits'].items():
                report.append(f"\n{'-'*60}")
                report.append(f"Protected Attribute: {attr}")
                report.append(f"Assessment: {result['overall_assessment']['summary']}")
                
                for metric_name, metric_data in result['metrics'].items():
                    report.append(f"\n  {metric_data['metric']}:")
                    report.append(f"    Status: {metric_data['interpretation']}")
        else:
            # Single attribute
            report.append(f"\nProtected Attribute: {audit_results['protected_attribute']}")
            report.append(f"Outcome Attribute: {audit_results['outcome_attribute']}")
            report.append(f"Total Samples: {audit_results['total_samples']}")
            report.append(f"\nOverall: {audit_results['overall_assessment']['summary']}")
            report.append(f"Metrics Passed: {audit_results['overall_assessment']['metrics_passed']}/{audit_results['overall_assessment']['total_metrics']}")
            
            for metric_name, metric_data in audit_results['metrics'].items():
                report.append(f"\n{metric_data['metric']}:")
                report.append(f"  Status: {metric_data['interpretation']}")
                if 'disparate_impact_score' in metric_data:
                    report.append(f"  Score: {metric_data['disparate_impact_score']} (threshold: {metric_data['threshold']})")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
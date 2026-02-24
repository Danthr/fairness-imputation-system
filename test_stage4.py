"""
Test Stage 4: Fairness Auditing Module
"""

import pandas as pd
import numpy as np
from backend.fairness_audit import FairnessAuditor

print("="*60)
print("STAGE 4: FAIRNESS AUDITING TEST")
print("="*60)

# Create a sample dataset with potential bias
# Simulating a hiring dataset
np.random.seed(42)

data = {
    'age': [25, 30, 35, 40, 45, 50, 55, 28, 32, 38, 42, 48, 52, 26, 31],
    'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M'],
    'education': ['BS', 'MS', 'PhD', 'BS', 'MS', 'PhD', 'BS', 'MS', 'PhD', 'BS', 'MS', 'PhD', 'BS', 'MS', 'PhD'],
    'income': [50000, 48000, 70000, 52000, 75000, 60000, 80000, 49000, 72000, 51000, 76000, 62000, 82000, 47000, 73000]
}

df = pd.DataFrame(data)

print("\n" + "="*60)
print("SAMPLE DATASET (Hiring Data)")
print("="*60)
print(df)
print(f"\nTotal records: {len(df)}")

# Initialize auditor
auditor = FairnessAuditor()

# Detect protected attributes
print("\n" + "="*60)
print("DETECTING PROTECTED ATTRIBUTES")
print("="*60)
protected_attrs = auditor.detect_protected_attributes(df)
print(f"Protected attributes found: {protected_attrs}")

# Audit gender fairness in income
print("\n" + "="*60)
print("AUDITING: Gender Fairness in Income")
print("="*60)
print("Checking if income distribution is fair across genders...")

# For income, we'll check if high income (>60000) is fairly distributed
audit_result = auditor.audit_simple(
    df=df,
    protected_attr='gender',
    outcome_attr='income',
    favorable_outcome=None  # Auto-detect based on median
)

# Display results
print("\n" + "-"*60)
print("AUDIT RESULTS")
print("-"*60)

if 'error' in audit_result:
    print(f"Error: {audit_result['error']}")
else:
    print(f"Protected Attribute: {audit_result['protected_attribute']}")
    print(f"Outcome Attribute: {audit_result['outcome_attribute']}")
    print(f"Groups Detected: {audit_result['groups_detected']}")
    print(f"Total Samples: {audit_result['total_samples']}")
    
    print("\n" + "-"*60)
    print("METRIC RESULTS")
    print("-"*60)
    
    # Disparate Impact
    di = audit_result['metrics']['disparate_impact']
    print(f"\n1. DISPARATE IMPACT")
    print(f"   Score: {di['disparate_impact_score']}")
    print(f"   Threshold: {di['threshold']} (4/5ths rule)")
    print(f"   Status: {di['interpretation']}")
    print(f"   Privileged group ({di['privileged_group']}): {di['privileged_rate']:.2%}")
    print(f"   Unprivileged group ({di['unprivileged_group']}): {di['unprivileged_rate']:.2%}")
    
    # Demographic Parity
    dp = audit_result['metrics']['demographic_parity']
    print(f"\n2. DEMOGRAPHIC PARITY")
    print(f"   Max Difference: {dp['max_difference']}")
    print(f"   Threshold: {dp['threshold']}")
    print(f"   Status: {dp['interpretation']}")
    print(f"   Group Rates:")
    for group, rate in dp['group_rates'].items():
        print(f"     {group}: {rate:.2%}")
    
    # Overall Assessment
    print("\n" + "-"*60)
    print("OVERALL ASSESSMENT")
    print("-"*60)
    assessment = audit_result['overall_assessment']
    print(f"Metrics Passed: {assessment['metrics_passed']}/{assessment['total_metrics']}")
    print(f"Fair: {assessment['is_fair']}")
    print(f"Summary: {assessment['summary']}")

# Generate formatted report
print("\n" + "="*60)
print("GENERATING FORMATTED REPORT")
print("="*60)
report = auditor.generate_audit_report(audit_result)
print(report)

# Test with multiple protected attributes
print("\n" + "="*60)
print("AUDITING MULTIPLE ATTRIBUTES")
print("="*60)

multi_audit = auditor.audit_multiple_attributes(
    df=df,
    protected_attrs=['gender', 'age'],
    outcome_attr='income'
)

print(f"\nAttributes audited: {multi_audit['total_attributes_audited']}")
print(f"Overall fairness: {multi_audit['overall_summary']['overall_fairness']}")
print(f"Fair attributes: {multi_audit['overall_summary']['fair_attributes']}/{multi_audit['overall_summary']['total_attributes']}")

print("\n" + "="*60)
print("✅ STAGE 4 TEST COMPLETE")
print("="*60)
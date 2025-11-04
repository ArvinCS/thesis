#!/usr/bin/env python3
"""
Validation script to test fair Zipfian distribution (α=0.5) in audit queries
vs. hierarchical Zipfian distribution (α=1.1) in transactional queries.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os

# Add the offchain directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'offchain', 'python'))

from access_patterns_enhanced import TransactionalPattern, AuditPattern
from basic_data_structure import Property, Document
from workload_generator import WorkloadGenerator

def generate_test_dataset(num_properties=30):
    """Generate a test dataset with known province distribution"""
    properties = []
    
    # Focus on top 3 provinces for clear distribution analysis
    top_provinces = [
        ("JAWA BARAT", 10),
        ("JAWA TENGAH", 8), 
        ("JAWA TIMUR", 6),
        ("DKI JAKARTA", 3),
        ("BANTEN", 3)
    ]
    
    prop_id = 1
    for province, count in top_provinces:
        for i in range(count):
            property_id = f"prop_{province}_{prop_id:03d}"
            property_obj = Property(
                property_id=property_id,
                province=province,
                documents=[]
            )
            
            # Add 4-6 documents per property
            num_docs = np.random.randint(4, 7)
            for doc_idx in range(num_docs):
                doc = Document(
                    doc_id=f"doc_{prop_id}_{doc_idx}",
                    doc_type="certificate",
                    content=f"Content for property {prop_id} document {doc_idx}",
                    province=province,
                    property_id=property_id
                )
                property_obj.documents.append(doc)
            
            properties.append(property_obj)
            prop_id += 1
    
    return properties

def test_distribution_fairness():
    """Test the fairness of audit vs transactional distributions"""
    print("🔬 Testing Fair Zipfian Distribution in Audit Queries")
    print("=" * 60)
    
    # Generate test dataset
    properties = generate_test_dataset()
    print(f"📊 Generated dataset: {len(properties)} properties")
    
    # Group properties by province
    province_properties = defaultdict(list)
    for prop in properties:
        province_properties[prop.province].append(prop)
    
    print(f"📍 Province distribution:")
    for province, props in province_properties.items():
        print(f"  {province}: {len(props)} properties")
    
    # Create document importance map for initialization
    document_importance_map = {
        "certificate": 100,
        "deed": 80,
        "tax": 60,
        "permit": 40
    }
    
    # Initialize patterns
    transactional_pattern = TransactionalPattern(
        document_importance_map=document_importance_map,
        alpha_threshold=0.15,
        zipf_parameter=1.3,
        property_zipf_parameter=1.1
    )
    
    audit_pattern = AuditPattern(
        document_importance_map=document_importance_map,
        alpha_threshold=0.15,
        zipf_parameter=1.3,
        property_zipf_parameter=1.1
    )
    
    # Test 1: Hierarchical distribution in transactional queries
    print(f"\n1️⃣ Testing Transactional Query Distribution (α=1.1):")
    trans_selections = defaultdict(int)
    
    for _ in range(1000):
        # Use hierarchical selection method directly
        province_weights = transactional_pattern.get_province_zipfian_weights(properties)
        selected_province = np.random.choice(
            list(province_weights.keys()),
            p=list(province_weights.values())
        )
        
        # Get properties in selected province
        province_props = [p for p in properties if p.province == selected_province]
        if province_props:
            # Use within-province Zipfian to select one property
            within_weights = transactional_pattern.get_within_province_zipfian_weights(province_props)
            selected_prop = np.random.choice(
                list(within_weights.keys()),
                p=list(within_weights.values())
            )
            trans_selections[selected_prop.property_id] += 1
    
    # Analyze transactional distribution
    jawa_barat_props = [p for p in properties if p.province == "JAWA BARAT"]
    jawa_barat_selections = sum(trans_selections[p.property_id] for p in jawa_barat_props)
    total_trans = sum(trans_selections.values())
    
    print(f"   Total transactional selections: {total_trans}")
    print(f"   JAWA BARAT selections: {jawa_barat_selections} ({jawa_barat_selections/total_trans*100:.1f}%)")
    
    # Check if highly skewed (should be for α=1.1)
    top_3_props = sorted([(count, prop_id) for prop_id, count in trans_selections.items()], reverse=True)[:3]
    top_3_total = sum(count for count, _ in top_3_props)
    print(f"   Top 3 properties: {top_3_total}/{total_trans} = {top_3_total/total_trans*100:.1f}% (should be heavily skewed)")
    
    # Test 2: Fair distribution in regional audit queries
    print(f"\n2️⃣ Testing Regional Audit Query Distribution (α=0.5):")
    
    # Focus on JAWA BARAT for regional audit testing
    jawa_barat_audit_selections = defaultdict(int)
    
    for _ in range(500):
        # Test regional audit sampling within JAWA BARAT
        selected_properties = audit_pattern.get_regional_audit_zipfian_sample(
            province_properties=jawa_barat_props,
            num_properties=3  # Select 3 properties
        )
        
        for prop in selected_properties:
            jawa_barat_audit_selections[prop.property_id] += 1
    
    # Analyze regional audit distribution
    total_regional = sum(jawa_barat_audit_selections.values())
    print(f"   Total regional audit selections: {total_regional}")
    
    # Check if more balanced (should be for α=0.5)
    audit_top_3 = sorted([(count, prop_id) for prop_id, count in jawa_barat_audit_selections.items()], reverse=True)[:3]
    audit_top_3_total = sum(count for count, _ in audit_top_3)
    print(f"   Top 3 properties: {audit_top_3_total}/{total_regional} = {audit_top_3_total/total_regional*100:.1f}% (should be more balanced)")
    
    # Test 3: Compare distribution fairness
    print(f"\n3️⃣ Distribution Fairness Comparison:")
    
    # Calculate Gini coefficient for both distributions
    def gini_coefficient(values):
        """Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality)"""
        values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(values, 1))) / (n * sum(values))
    
    # Get selection counts for JAWA BARAT properties only (for fair comparison)
    trans_jb_counts = [trans_selections[p.property_id] for p in jawa_barat_props]
    audit_jb_counts = [jawa_barat_audit_selections[p.property_id] for p in jawa_barat_props]
    
    trans_gini = gini_coefficient(trans_jb_counts)
    audit_gini = gini_coefficient(audit_jb_counts)
    
    print(f"   Transactional Gini coefficient: {trans_gini:.3f} (higher = more skewed)")
    print(f"   Regional Audit Gini coefficient: {audit_gini:.3f} (lower = more fair)")
    print(f"   Fairness improvement: {((trans_gini - audit_gini) / trans_gini * 100):.1f}%")
    
    # Test 4: National audit across provinces
    print(f"\n4️⃣ Testing National Audit Distribution:")
    
    national_selections = defaultdict(int)
    province_coverage = defaultdict(int)
    
    for _ in range(200):
        selected_properties = audit_pattern.get_national_audit_zipfian_sample(
            all_properties=properties,
            num_properties=5  # Select 5 properties across provinces
        )
        
        provinces_in_query = set()
        for prop in selected_properties:
            national_selections[prop.property_id] += 1
            provinces_in_query.add(prop.province)
        
        # Count how many provinces were covered
        province_coverage[len(provinces_in_query)] += 1
    
    print(f"   Province coverage in queries:")
    for num_provinces, count in sorted(province_coverage.items()):
        print(f"     {num_provinces} provinces: {count} queries ({count/200*100:.1f}%)")
    
    # Validation Results
    print(f"\n✅ Validation Results:")
    print(f"   • Transactional queries use hierarchical Zipfian (α=1.1): {'✅' if trans_gini > 0.4 else '❌'}")
    print(f"   • Regional audit uses fair Zipfian (α=0.5): {'✅' if audit_gini < trans_gini * 0.8 else '❌'}")
    print(f"   • National audit covers multiple provinces: {'✅' if any(k > 1 for k in province_coverage.keys()) else '❌'}")
    
    return {
        'transactional_gini': trans_gini,
        'audit_gini': audit_gini,
        'fairness_improvement': (trans_gini - audit_gini) / trans_gini * 100,
        'multi_province_coverage': sum(count for provinces, count in province_coverage.items() if provinces > 1) / 200 * 100
    }

if __name__ == "__main__":
    results = test_distribution_fairness()
    
    print(f"\n🎯 Summary:")
    print(f"   Fair Zipfian (α=0.5) achieves {results['fairness_improvement']:.1f}% improvement in distribution fairness")
    print(f"   National audit queries cover multiple provinces in {results['multi_province_coverage']:.1f}% of cases")
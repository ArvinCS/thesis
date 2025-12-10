#!/usr/bin/env python3
"""
Generate sensitivity analysis figures for the thesis paper.
Creates two publication-quality charts:
1. Gas Cost vs Zipf s parameter (transactional workload)
2. Gas Cost vs Audit λ parameter (audit workload)

Matches the style of generate_charts.py
"""
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Approach display names (matching generate_charts.py style)
APPROACH_NAMES = {
    'traditional_multiproof': 'Standard Balanced Tree',
    'traditional_property_level_huffman': 'Global Huffman Tree',
    'clustered_province': 'Clustered Tree',
    'clustered_province_with_document_huffman': 'Hierarchical Traffic-Aware Tree'
}

# Short names for legend (paper notation style)
APPROACH_SHORT_NAMES = {
    'traditional_multiproof': 'M_baseline',
    'traditional_property_level_huffman': 'M_prop_huffman',
    'clustered_province': 'M_prop_cluster',
    'clustered_province_with_document_huffman': 'M_hybrid (Ours)'
}

MODEL_COLORS = {
    'traditional_multiproof': '#E74C3C',  # Red
    'traditional_property_level_huffman': '#3498DB',  # Blue
    'clustered_province': '#2ECC71',  # Green
    'clustered_province_with_document_huffman': '#9B59B6'  # Purple
}

MODEL_MARKERS = {
    'traditional_multiproof': 'o',
    'traditional_property_level_huffman': 's',
    'clustered_province': '^',
    'clustered_province_with_document_huffman': 'D'
}

def load_sensitivity_data(base_path, param_values, param_name):
    """Load data from sensitivity test folders."""
    data = {model: [] for model in APPROACH_NAMES.keys()}
    
    for val in param_values:
        folder = os.path.join(base_path, str(val))
        files = glob.glob(os.path.join(folder, 'raw_*.json'))
        
        if files:
            with open(files[0]) as f:
                results = json.load(f)
            
            for model in APPROACH_NAMES.keys():
                if model in results:
                    avg_gas = results[model]['total_gas_used'] / results[model]['total_queries']
                    data[model].append(avg_gas)
                else:
                    data[model].append(None)
        else:
            print(f"  Warning: No data found in {folder}")
            for model in APPROACH_NAMES.keys():
                data[model].append(None)
    
    return data

def plot_zipf_sensitivity(output_path, base_path):
    """Generate Zipf s parameter sensitivity figure."""
    s_values = [1.1, 1.2, 1.3]
    data = load_sensitivity_data(os.path.join(base_path, 'zipf'), s_values, 's')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for model_key in APPROACH_NAMES.keys():
        model_name = APPROACH_NAMES[model_key]  # Use full names for legend
        if all(v is not None for v in data[model_key]):
            ax.plot(s_values, [v/1000 for v in data[model_key]], 
                   marker=MODEL_MARKERS[model_key],
                   color=MODEL_COLORS[model_key],
                   label=model_name,
                   linewidth=2.5,
                   markersize=10)
    
    ax.set_xlabel('Zipfian Exponent (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Gas Cost (×1000)', fontsize=12, fontweight='bold')
    ax.set_title('Sensitivity Analysis: Transaction Concentration', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(s_values)
    ax.legend(loc='center right', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(1.05, 1.35)
    
    # Add improvement annotation (positioned above green line to avoid overlap)
    baseline_avg = np.mean(data['traditional_multiproof'])
    mhybrid_avg = np.mean(data['clustered_province_with_document_huffman'])
    clustered_avg = np.mean(data['clustered_province'])
    improvement = (baseline_avg - mhybrid_avg) / baseline_avg * 100
    ax.annotate(f'M_hybrid: {improvement:.1f}% avg improvement vs baseline',
                xy=(1.2, mhybrid_avg/1000), 
                xytext=(1.12, clustered_avg/1000 + 5),  # Position above green line
                fontsize=9, style='italic', color='#9B59B6',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")

def plot_audit_sensitivity(output_path, base_path):
    """Generate audit λ parameter sensitivity figure."""
    lambda_values = [50, 100, 150]
    data = load_sensitivity_data(os.path.join(base_path, 'audit'), lambda_values, 'lambda')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for model_key in APPROACH_NAMES.keys():
        model_name = APPROACH_NAMES[model_key]  # Use full names for legend
        if all(v is not None for v in data[model_key]):
            ax.plot(lambda_values, [v/1000 for v in data[model_key]], 
                   marker=MODEL_MARKERS[model_key],
                   color=MODEL_COLORS[model_key],
                   label=model_name,
                   linewidth=2.5,
                   markersize=10)
    
    ax.set_xlabel('Audit Batch Size (λ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Gas Cost (×1000)', fontsize=12, fontweight='bold')
    ax.set_title('Sensitivity Analysis: Audit Workload Intensity', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(lambda_values)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(40, 160)
    
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add scaling annotation
    if all(v is not None for v in data['clustered_province']) and all(v is not None for v in data['traditional_property_level_huffman']):
        clustered_50 = data['clustered_province'][0]
        clustered_150 = data['clustered_province'][2]
        huffman_50 = data['traditional_property_level_huffman'][0]
        huffman_150 = data['traditional_property_level_huffman'][2]
        
        advantage_50 = (huffman_50 - clustered_50) / huffman_50 * 100
        advantage_150 = (huffman_150 - clustered_150) / huffman_150 * 100
        
        ax.annotate(f'Clustered models:\n{advantage_50:.1f}%-{advantage_150:.1f}% better\nthan Global Huffman',
                    xy=(100, data['clustered_province'][1]/1000),
                    xytext=(115, data['clustered_province'][1]/1000 - 250),
                    fontsize=9, style='italic', color='#2ECC71',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='#2ECC71', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")

def main():
    # Base path for test reports
    script_dir = Path(__file__).parent
    base_path = script_dir / 'test_reports'
    
    # Create output directory
    output_dir = base_path / 'sensitivity_figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 Generating Sensitivity Analysis Figures...")
    print(f"   Data source: {base_path}")
    print(f"   Output: {output_dir}")
    print()
    
    # Generate figures
    plot_zipf_sensitivity(str(output_dir / 'zipf_sensitivity.png'), str(base_path))
    plot_audit_sensitivity(str(output_dir / 'audit_sensitivity.png'), str(base_path))
    
    print()
    print(f"📁 All figures saved to: {output_dir}")
    print("   - zipf_sensitivity.png / .pdf")
    print("   - audit_sensitivity.png / .pdf")

if __name__ == "__main__":
    main()

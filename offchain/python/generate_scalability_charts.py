#!/usr/bin/env python3
"""
Generate grouped bar charts comparing different approaches across multiple property scales.
Reads multiple raw_results JSON files and creates comparison charts.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# Approach display names
APPROACH_NAMES = {
    'traditional_multiproof': 'Standard Balance',
    'traditional_property_level_huffman': 'Global Traffic-Aware',
    'clustered_province': 'Clustered Balanced',
    'clustered_province_with_document_huffman': 'Hierarchical Traffic-Aware',
}

def load_results(file_paths):
    """Load results from multiple JSON files."""
    results = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                results.append(data)
                print(f"✓ Loaded: {file_path}")
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}")
    return results

def extract_metrics(results_list, scale_labels):
    """Extract metrics from loaded results."""
    approaches = []
    gas_data = {}
    proof_data = {}
    build_time_data = {}
    
    # Get all approaches from first result file
    if results_list:
        approaches = [app for app in results_list[0].keys()]
    
    # Initialize data structures
    for approach in approaches:
        gas_data[approach] = []
        proof_data[approach] = []
        build_time_data[approach] = []
    
    # Extract data from each scale
    for results in results_list:
        for approach in approaches:
            if approach in results:
                data = results[approach]
                
                # Average gas per query
                avg_gas = data.get('average_gas_per_query', 0)
                gas_data[approach].append(avg_gas)
                
                # Average proof size (calculate from query_results if available)
                query_results = data.get('query_results', [])
                if query_results:
                    avg_proof = sum(q.get('proof_size', 0) for q in query_results) / len(query_results)
                else:
                    avg_proof = 0
                proof_data[approach].append(avg_proof)
                
                # Build time
                build_time = data.get('build_time', 0)
                build_time_data[approach].append(build_time)
            else:
                # Missing data for this approach at this scale
                gas_data[approach].append(0)
                proof_data[approach].append(0)
                build_time_data[approach].append(0)
    
    return approaches, gas_data, proof_data, build_time_data

def create_grouped_bar_chart(data_dict, scale_labels, ylabel, title, output_path, 
                             approach_names=None, use_log_scale=False, format_func=None):
    """Create a grouped bar chart."""
    approaches = list(data_dict.keys())
    n_groups = len(scale_labels)
    n_approaches = len(approaches)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set the width of bars and positions
    bar_width = 0.8 / n_approaches
    index = np.arange(n_groups)
    
    # Colors for different approaches (Hybrid in green/orange to highlight proposed model)
    color_map = {
        'traditional_multiproof': '#1f77b4',  # Blue
        'traditional_property_level_huffman': '#ff7f0e',  # Orange
        'clustered_province': '#d62728',  # Red
        'clustered_province_with_document_huffman': '#2ca02c',  # Green (proposed model)
    }
    
    # Create bars for each approach
    for i, approach in enumerate(approaches):
        values = data_dict[approach]
        position = index + (i - n_approaches/2 + 0.5) * bar_width
        label = approach_names.get(approach, approach) if approach_names else approach
        
        # Get color for this approach (use color map, fallback to default palette)
        color = color_map.get(approach, ['#9467bd', '#8c564b'][i % 2])
        
        bars = ax.bar(position, values, bar_width, label=label, color=color)
        
        # Add value labels on top of bars
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value > 0:
                height = bar.get_height()
                if format_func:
                    label_text = format_func(value)
                else:
                    label_text = f'{value:,.0f}'
                
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label_text,
                       ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Customize the chart
    ax.set_xlabel('Number of Properties', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(index)
    ax.set_xticklabels(scale_labels)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    if use_log_scale:
        ax.set_yscale('log')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def format_time(seconds):
    """Format time in appropriate units."""
    if seconds < 1:
        return f'{seconds*1000:.1f}ms'
    elif seconds < 60:
        return f'{seconds:.2f}s'
    else:
        return f'{seconds/60:.1f}m'

def format_bytes(bytes_val):
    """Format bytes in KB."""
    return f'{bytes_val/1024:.1f}KB'

def main():
    parser = argparse.ArgumentParser(description='Generate scalability comparison charts')
    parser.add_argument('--files', nargs='+', required=True, 
                       help='Paths to raw_results JSON files (in order: 1k, 5k, 10k, 50k, 100k)')
    parser.add_argument('--labels', nargs='+', 
                       default=['1K', '5K', '10K', '50K', '100K'],
                       help='Labels for each scale')
    parser.add_argument('--output-dir', default='test_reports/scalability_charts',
                       help='Output directory for charts')
    
    args = parser.parse_args()
    
    # Validate input
    if len(args.files) != len(args.labels):
        print(f"Error: Number of files ({len(args.files)}) must match number of labels ({len(args.labels)})")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating Scalability Charts")
    print(f"{'='*60}\n")
    
    # Load all results
    print("Loading result files...")
    results_list = load_results(args.files)
    
    if not results_list:
        print("Error: No valid result files loaded")
        return
    
    print(f"\n{'='*60}")
    print("Extracting metrics...")
    approaches, gas_data, proof_data, build_time_data = extract_metrics(results_list, args.labels)
    
    print(f"\nFound {len(approaches)} approaches: {', '.join([APPROACH_NAMES.get(a, a) for a in approaches])}")
    print(f"Scales: {', '.join(args.labels)}")
    
    # Generate charts
    print(f"\n{'='*60}")
    print("Generating charts...")
    
    # 1. Gas Usage Comparison
    create_grouped_bar_chart(
        gas_data,
        args.labels,
        ylabel='Average Gas per Query',
        title='Gas Consumption Across Different Scales',
        output_path=output_dir / 'gas_consumption_scalability.png',
        approach_names=APPROACH_NAMES
    )
    
    # 2. Proof Size Comparison
    create_grouped_bar_chart(
        proof_data,
        args.labels,
        ylabel='Average Proof Size (bytes)',
        title='Proof Size Across Different Scales',
        output_path=output_dir / 'proof_size_scalability.png',
        approach_names=APPROACH_NAMES,
        format_func=format_bytes
    )
    
    # 3. Build Time Comparison (log scale)
    create_grouped_bar_chart(
        build_time_data,
        args.labels,
        ylabel='Build Time (seconds, log scale)',
        title='Tree Build Time Across Different Scales',
        output_path=output_dir / 'build_time_scalability.png',
        approach_names=APPROACH_NAMES,
        use_log_scale=True,
        format_func=format_time
    )
    
    # 4. Gas Efficiency (Gas per 1000 properties)
    print("\nCalculating efficiency metrics...")
    properties_counts = []
    for label in args.labels:
        # Extract number from label (e.g., "1K" -> 1000)
        num_str = label.replace('K', '').replace('k', '')
        properties_counts.append(float(num_str) * 1000)
    
    gas_efficiency = {}
    for approach in approaches:
        gas_efficiency[approach] = [
            gas / (props / 1000) if props > 0 else 0
            for gas, props in zip(gas_data[approach], properties_counts)
        ]
    
    create_grouped_bar_chart(
        gas_efficiency,
        args.labels,
        ylabel='Gas per 1000 Properties',
        title='Gas Efficiency Across Different Scales',
        output_path=output_dir / 'gas_efficiency_scalability.png',
        approach_names=APPROACH_NAMES
    )
    
    # Generate summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}\n")
    
    for approach in approaches:
        display_name = APPROACH_NAMES.get(approach, approach)
        print(f"\n{display_name}:")
        print(f"  Gas Usage Range: {min(g for g in gas_data[approach] if g > 0):,.0f} - {max(gas_data[approach]):,.0f}")
        if any(proof_data[approach]):
            print(f"  Proof Size Range: {min(p for p in proof_data[approach] if p > 0):,.0f} - {max(proof_data[approach]):,.0f} bytes")
        print(f"  Build Time Range: {min(b for b in build_time_data[approach] if b > 0):.3f}s - {max(build_time_data[approach]):.3f}s")
    
    print(f"\n{'='*60}")
    print(f"✓ All charts saved to: {output_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

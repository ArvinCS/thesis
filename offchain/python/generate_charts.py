#!/usr/bin/env python3
"""
Chart Generator for Test Results
Generates grouped bar charts from JSON test results for different approaches.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Approach display names
APPROACH_NAMES = {
    'traditional_multiproof': 'Standard Balanced Tree',
    'traditional_property_level_huffman': 'Global Huffman Tree',
    'clustered_province': 'Clustered Tree',
    'clustered_province_with_document_huffman': 'Hierarchical Traffic-Aware Tree'
}

def load_json_data(filepath):
    """Load and parse JSON test results."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_gas_by_query_type(data, query_type):
    """Extract gas_used values for a specific query type across all approaches."""
    result = {}
    for approach, approach_data in data.items():
        gas_values = [
            q['gas_used'] for q in approach_data['query_results']
            if q['query_type'] == query_type and q['verification_success']
        ]
        result[approach] = gas_values
    return result

def extract_all_gas_usage(data):
    """Extract gas_used values for all query types across all approaches."""
    result = {}
    for approach, approach_data in data.items():
        gas_values = [
            q['gas_used'] for q in approach_data['query_results']
            if q['verification_success']
        ]
        result[approach] = gas_values
    return result

def extract_proof_sizes(data):
    """Extract proof_size values across all approaches (all query types)."""
    result = {}
    for approach, approach_data in data.items():
        proof_sizes = [
            q['proof_size'] for q in approach_data['query_results']
            if q['verification_success']
        ]
        result[approach] = proof_sizes
    return result

def extract_communication_costs(data):
    """Extract communication_cost values across all approaches (all query types)."""
    result = {}
    for approach, approach_data in data.items():
        costs = [
            q['communication_cost'] for q in approach_data['query_results']
            if q['verification_success'] and 'communication_cost' in q
        ]
        result[approach] = costs
    return result

def extract_verification_times(data):
    """Extract verification_time values across all approaches (all query types)."""
    result = {}
    for approach, approach_data in data.items():
        times = [
            q['verification_time'] for q in approach_data['query_results']
            if q['verification_success']
        ]
        result[approach] = times
    return result

def extract_build_times(data):
    """Extract build_time for each approach."""
    result = {}
    for approach, approach_data in data.items():
        if 'build_time' in approach_data:
            result[approach] = approach_data['build_time']
    return result

def create_grouped_bar_chart(data_dict, title, ylabel, filename, output_dir, use_milliseconds=False):
    """
    Create a bar chart showing average values for each approach.
    
    Args:
        data_dict: Dictionary mapping approach names to lists of values
        title: Chart title
        ylabel: Y-axis label
        filename: Output filename
        output_dir: Output directory path
        use_milliseconds: If True, convert seconds to milliseconds and format accordingly
    """
    approaches = list(data_dict.keys())
    num_approaches = len(approaches)
    
    if num_approaches == 0:
        print(f"Warning: No data for {title}")
        return
    
    # Calculate average for each approach
    averages = []
    display_names = []
    
    for approach in approaches:
        values = data_dict[approach]
        if len(values) > 0:
            avg = np.mean(values)
            # Convert to milliseconds if needed
            if use_milliseconds:
                avg = avg * 1000
            averages.append(avg)
        else:
            averages.append(0)
        display_names.append(APPROACH_NAMES.get(approach, approach))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors for different approaches
    colors = plt.cm.Set3(np.linspace(0, 1, num_approaches))
    
    # Plot bars
    x = np.arange(num_approaches)
    bars = ax.bar(x, averages, color=colors, width=0.6)
    
    # Customize chart
    ax.set_xlabel('Approach', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Format y-axis appropriately
    if use_milliseconds:
        # For milliseconds, use decimal formatting
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    else:
        # For other metrics, use thousands separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add value labels on bars
    for bar, avg in zip(bars, averages):
        height = bar.get_height()
        if use_milliseconds:
            label = f'{avg:.2f}'
        else:
            label = f'{avg:,.0f}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label,
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save chart
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_path}")
    plt.close()

def create_build_time_chart(build_times, output_dir, filename='construction_time.png'):
    """Create a simple bar chart for build times."""
    if not build_times:
        print("Warning: No build time data found")
        return
    
    approaches = list(build_times.keys())
    times = [build_times[approach] for approach in approaches]
    display_names = [APPROACH_NAMES.get(a, a) for a in approaches]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(approaches)))
    bars = ax.bar(range(len(approaches)), times, color=colors)
    
    ax.set_xlabel('Approach', fontsize=12, fontweight='bold')
    ax.set_ylabel('Construction Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Tree Construction Time Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(approaches)))
    ax.set_xticklabels(display_names, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_path}")
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_charts.py <path_to_json_file>")
        sys.exit(1)
    
    json_file = Path(sys.argv[1])
    
    if not json_file.exists():
        print(f"Error: File not found: {json_file}")
        sys.exit(1)
    
    print(f"Loading data from: {json_file}")
    data = load_json_data(json_file)
    
    # Extract session identifier from filename (e.g., raw_results_20251207_235101.json -> 20251207_235101)
    source_name = json_file.stem  # filename without extension
    if source_name.startswith('raw_results_'):
        session_id = source_name.replace('raw_results_', '')
    else:
        session_id = source_name
    
    # Create session-specific output directory
    output_dir = json_file.parent / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    print("Generating core benchmark charts...")
    
    # 1. Average Gas Cost
    all_gas = extract_all_gas_usage(data)
    create_grouped_bar_chart(
        all_gas,
        'Average Gas Cost Comparison',
        'Average Gas Used',
        'average_gas_cost.png',
        output_dir
    )
    
    # 2. Average Communication Cost
    communication_costs = extract_communication_costs(data)
    create_grouped_bar_chart(
        communication_costs,
        'Average Communication Cost Comparison',
        'Average Communication Cost (Number of Nodes)',
        'average_communication_cost.png',
        output_dir
    )
    
    # 3. Average Verification Time
    verification_times = extract_verification_times(data)
    create_grouped_bar_chart(
        verification_times,
        'Average Verification Time Comparison',
        'Average Verification Time (milliseconds)',
        'average_verification_time.png',
        output_dir,
        use_milliseconds=True
    )
    
    # 4. Construction Time
    build_times = extract_build_times(data)
    create_build_time_chart(build_times, output_dir, filename='construction_time.png')
    
    print(f"\n✓ All charts generated successfully in: {output_dir}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Performance Metrics and Visualization Module

This module provides comprehensive performance analysis and visualization for:
1. Multi-day verification test results
2. Pairs-first Huffman optimization analysis
3. Gas cost analysis
4. Scalability analysis
5. Comparative performance metrics

Features:
- Interactive charts and graphs
- Performance trend analysis
- Optimization effectiveness visualization
- Cost-benefit analysis charts
- Export capabilities for reports
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import os
from collections import defaultdict

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PerformanceVisualizer:
    """
    Comprehensive performance visualizer for hierarchical Merkle tree analysis.
    
    This visualizer creates:
    1. Performance comparison charts
    2. Optimization effectiveness graphs
    3. Gas cost analysis visualizations
    4. Scalability trend charts
    5. Multi-day performance dashboards
    """
    
    def __init__(self, output_dir="performance_reports"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Color schemes for different approaches
        self.color_schemes = {
            'hierarchical': '#228B22',  # Forest Green - FIXED: Use consistent naming
            'traditional_multiproof': "#009480",  # Royal Blue
            'traditional_single_proof': '#DC143C',  # Crimson
            'traditional_huffman': "#FF8C00",  # Dark Orange
            'clustered_flat': "#F747E8",
            'clustered_flat_with_merkle': "#3477C4",
            'baseline': '#808080'  # Gray
        }
        
        # Performance metrics
        self.metrics = {
            'proof_size': 'Proof Size (bytes)',
            'verification_time': 'Verification Time (seconds)',
            'gas_cost': 'Gas Cost (wei)',
            'gas_cost_usd': 'Gas Cost (USD)',
            'build_time': 'Build Time (seconds)',
            'optimization_ratio': 'Optimization Ratio (%)'
        }
    
    def visualize_multi_day_results(self, multi_day_report: Dict[str, Any], selected_approaches: List[str]):
        """Create comprehensive visualizations for multi-day verification results."""
        print("Creating multi-day performance visualizations...")
        
        # Extract data from report first
        verification_results = multi_day_report.get('verification_results', [])
        
        if selected_approaches is None or len(selected_approaches) == 0:
            # Use all available approaches from the data instead of hardcoded list
            available_approaches = set()
            for result in verification_results:
                if 'error' not in result:
                    available_approaches.add(result.get('approach', 'unknown'))
            selected_approaches = list(available_approaches) if available_approaches else ['hierarchical', 'clustered_flat']
        optimization_analysis = multi_day_report.get('optimization_analysis', {})
        traffic_analysis = multi_day_report.get('traffic_analysis', {})
        
        # Create visualizations
        self._create_daily_performance_chart(verification_results, selected_approaches)
        # self._create_optimization_effectiveness_chart(optimization_analysis)
        self._create_traffic_pattern_chart(traffic_analysis)
        self._create_cross_province_analysis_chart(verification_results, selected_approaches)
        self._create_performance_comparison_chart(verification_results)
        # Create consistent proof size comparison using same data as daily trends
        self._create_proof_size_comparison_from_verification_results(verification_results)
        
        print(f"Multi-day visualizations saved to {self.output_dir}/")

    def visualize_pairs_huffman_analysis(self, huffman_report: Dict[str, Any], selected_approaches: List[str]):
        """Create visualizations for Pairs-first Huffman analysis."""
        print("Creating Pairs-first Huffman analysis visualizations...")
        
        if selected_approaches is None or len(selected_approaches) == 0:
            # Use all available approaches from the data instead of hardcoded list
            verification_results = huffman_report.get('verification_results', [])
            available_approaches = set()
            for result in verification_results:
                if 'error' not in result:
                    available_approaches.add(result.get('approach', 'unknown'))
            selected_approaches = list(available_approaches) if available_approaches else ['hierarchical', 'clustered_flat']

        # Extract data from report
        co_verification_analysis = huffman_report.get('co_verification_analysis', {})
        optimization_results = huffman_report.get('optimization_results', {})
        proof_analysis = huffman_report.get('proof_analysis', {})
        verification_results = huffman_report.get('verification_results', [])
        
        # Create visualizations
        self._create_co_verification_pattern_chart(co_verification_analysis)
        # self._create_optimization_scenario_chart(optimization_results)
        
        # Use consistent chart if verification results available, otherwise fallback to old method
        if verification_results:
            self._create_proof_size_comparison_from_verification_results(verification_results)
        else:
            self._create_proof_size_comparison_chart(proof_analysis)
            
        self._create_optimization_potential_chart(co_verification_analysis)
        
        print(f"Pairs-first Huffman visualizations saved to {self.output_dir}/")
    
    def visualize_gas_cost_analysis(self, gas_report: Dict[str, Any], selected_approaches: List[str]):
        """Create visualizations for gas cost analysis from multi-day data."""
        print("Creating gas cost analysis visualizations from multi-day data...")
        
        # Check if this is multi-day data with gas information
        if 'verification_results' in gas_report:
            # Multi-day format - extract gas data directly from verification results
            print("   🔍 Detected multi-day verification results format")
            self._create_multi_day_gas_analysis(gas_report, selected_approaches)
        elif 'gas_results' in gas_report:
            # Legacy gas analysis format - use existing structure
            print("   🔍 Detected legacy gas analysis format")
            gas_results = gas_report.get('gas_results', {})
            optimization_analysis = gas_report.get('optimization_analysis', {})
            scalability_analysis = gas_report.get('scalability_analysis', {})
            
            # Create visualizations
            self._create_gas_cost_comparison_chart(gas_report, selected_approaches)
            self._create_cost_optimization_chart(optimization_analysis)
            self._create_scalability_analysis_chart(scalability_analysis)
            self._create_cost_benefit_analysis_chart(optimization_analysis)
        else:
            print(f"   ⚠️  Unrecognized gas report format. Available keys: {list(gas_report.keys())}")
            return
        
        print(f"Gas cost visualizations saved to {self.output_dir}/")
    
    def _create_multi_day_gas_analysis(self, multi_day_report: Dict[str, Any], selected_approaches: List[str]):
        """Create comprehensive gas cost analysis from multi-day verification results."""
        print("📊 Creating multi-day gas cost analysis...")
        
        # Extract verification results
        verification_results = multi_day_report.get('verification_results', [])
        
        if not verification_results:
            print("⚠️  No verification results found for gas analysis")
            return
        
        # Filter approaches if specified
        if selected_approaches is None:
            # Use all available approaches from the data
            available_approaches = set()
            for result in verification_results:
                if 'error' not in result:
                    available_approaches.add(result.get('approach', 'unknown'))
            selected_approaches = list(available_approaches)
        
        # Extract gas data by approach and day
        gas_data = defaultdict(lambda: defaultdict(list))
        
        for result in verification_results:
            if 'error' not in result and result.get('approach') in selected_approaches:
                approach = result['approach']
                day = result.get('day', 0)
                # Extract gas information
                estimated_gas = result.get('estimated_gas')
                actual_gas = result.get('actual_gas_used')
                
                # Use actual gas if available, otherwise estimated
                gas_cost = actual_gas if actual_gas is not None else estimated_gas
                
                if gas_cost is not None and isinstance(gas_cost, (int, float)) and gas_cost > 0:
                    gas_data[approach][day].append({
                        'gas_cost': gas_cost,
                        'properties_count': result.get('properties_count', 0),
                        'provinces_count': result.get('provinces_count', 0),
                        'cross_province': result.get('cross_province', False),
                        'proof_size': result.get('proof_size_bytes', 0),
                        'verification_time': result.get('verification_time', 0)
                    })
        
        if not gas_data:
            print("⚠️  No valid gas data found in verification results")
            return
        
        # Create gas analysis visualizations
        self._create_multi_day_gas_comparison_chart(gas_data, selected_approaches)
        self._create_multi_day_gas_progression_chart(gas_data, selected_approaches)
        self._create_gas_cost_per_property_chart(gas_data, selected_approaches)
        self._create_cross_province_gas_analysis_chart(gas_data, selected_approaches)
        
        # Calculate and display gas optimization summary
        self._calculate_gas_optimization_summary(gas_data, selected_approaches)
    
    def _create_multi_day_gas_comparison_chart(self, gas_data: Dict[str, Dict[int, List[Dict]]], selected_approaches: List[str]):
        """Create overall gas cost comparison chart from multi-day data."""
        print("   📊 Creating multi-day gas comparison chart...")
        
        # Calculate average gas costs per approach
        approach_gas_averages = {}
        approach_gas_ranges = {}
        
        for approach in selected_approaches:
            if approach in gas_data:
                all_gas_costs = []
                for day_data in gas_data[approach].values():
                    all_gas_costs.extend([event['gas_cost'] for event in day_data])
                
                if all_gas_costs:
                    approach_gas_averages[approach] = np.mean(all_gas_costs)
                    approach_gas_ranges[approach] = {
                        'min': np.min(all_gas_costs),
                        'max': np.max(all_gas_costs),
                        'std': np.std(all_gas_costs),
                        'count': len(all_gas_costs)
                    }
        
        if not approach_gas_averages:
            print("   ⚠️  No gas data available for comparison")
            return
        
        # Create comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Chart 1: Average gas costs with error bars
        approaches = list(approach_gas_averages.keys())
        avg_costs = list(approach_gas_averages.values())
        std_errors = [approach_gas_ranges[app]['std'] for app in approaches]
        colors = [self.color_schemes.get(app, '#808080') for app in approaches]
        
        bars = ax1.bar(approaches, avg_costs, yerr=std_errors, color=colors, alpha=0.7, capsize=5)
        
        # Add value labels
        for bar, avg_cost, app in zip(bars, avg_costs, approaches):
            count = approach_gas_ranges[app]['count']
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(avg_costs) * 0.01,
                    f'{avg_cost:,.0f}\n({count} events)', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
        
        ax1.set_title('Average Gas Cost Comparison\n(Multi-day Data with Error Bars)', fontweight='bold')
        ax1.set_ylabel('Average Gas Cost (wei)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Chart 2: Gas cost distributions (box plot)
        box_data = []
        box_labels = []
        
        for approach in approaches:
            all_gas_costs = []
            for day_data in gas_data[approach].values():
                all_gas_costs.extend([event['gas_cost'] for event in day_data])
            box_data.append(all_gas_costs)
            box_labels.append(approach.replace('_', '\n'))
        
        box_plot = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # Color the boxes
        for patch, approach in zip(box_plot['boxes'], approaches):
            patch.set_facecolor(self.color_schemes.get(approach, '#808080'))
            patch.set_alpha(0.7)
        
        ax2.set_title('Gas Cost Distribution\n(Shows variability across all events)', fontweight='bold')
        ax2.set_ylabel('Gas Cost (wei)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/multi_day_gas_comparison_{self.timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary
        print(f"   ✅ Gas comparison completed:")
        for approach in approaches:
            avg = approach_gas_averages[approach]
            data = approach_gas_ranges[approach]
            print(f"      {approach:25}: {avg:8,.0f} avg ({data['min']:6,.0f}-{data['max']:7,.0f}, {data['count']:3d} events)")
    
    def _create_multi_day_gas_progression_chart(self, gas_data: Dict[str, Dict[int, List[Dict]]], selected_approaches: List[str]):
        """Create daily gas cost progression chart showing optimization over time."""
        print("   📊 Creating daily gas progression chart...")
        
        # Calculate daily averages for each approach
        daily_progression = {}
        max_days = 0
        
        for approach in selected_approaches:
            if approach in gas_data:
                daily_averages = {}
                for day, day_data in gas_data[approach].items():
                    if day_data:
                        daily_averages[day] = np.mean([event['gas_cost'] for event in day_data])
                        max_days = max(max_days, day)
                
                if daily_averages:
                    daily_progression[approach] = daily_averages
        
        if not daily_progression:
            print("   ⚠️  No daily progression data available")
            return
        
        # Create progression chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        days = list(range(max_days))
        
        for approach, progression in daily_progression.items():
            plot_days = []
            plot_costs = []
            
            for day in days:
                if day in progression:
                    plot_days.append(day + 1)  # Display as 1-indexed
                    plot_costs.append(progression[day])
            
            if plot_costs:
                # Calculate optimization trend
                if len(plot_costs) > 1:
                    # Calculate percentage improvement from day 1 to last day
                    improvement = ((plot_costs[0] - plot_costs[-1]) / plot_costs[0]) * 100
                    label = f"{approach.replace('_', ' ').title()} (-{improvement:.1f}%)" if improvement > 0 else f"{approach.replace('_', ' ').title()}"
                else:
                    label = approach.replace('_', ' ').title()
                
                ax.plot(plot_days, plot_costs, marker='o', linewidth=2, markersize=6,
                       label=label, color=self.color_schemes.get(approach, '#808080'))
                
                # Add trend line
                if len(plot_costs) > 2:
                    z = np.polyfit(plot_days, plot_costs, 1)
                    p = np.poly1d(z)
                    ax.plot(plot_days, p(plot_days), "--", alpha=0.5, 
                           color=self.color_schemes.get(approach, '#808080'))
        
        ax.set_title('Daily Gas Cost Progression\n(Showing Adaptive Optimization Over Time)', fontweight='bold')
        ax.set_xlabel('Day')
        ax.set_ylabel('Average Gas Cost (wei)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, max_days + 1))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/multi_day_gas_progression_{self.timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✅ Daily progression chart created for {len(daily_progression)} approaches")
    
    def _create_gas_cost_per_property_chart(self, gas_data: Dict[str, Dict[int, List[Dict]]], selected_approaches: List[str]):
        """Create gas cost per property analysis chart."""
        print("   📊 Creating gas cost per property analysis...")
        
        # Calculate gas cost efficiency (gas per property)
        efficiency_data = {}
        
        for approach in selected_approaches:
            if approach in gas_data:
                gas_per_property = []
                
                for day_data in gas_data[approach].values():
                    for event in day_data:
                        props_count = event.get('properties_count', 1)
                        if props_count > 0:
                            efficiency = event['gas_cost'] / props_count
                            gas_per_property.append({
                                'efficiency': efficiency,
                                'properties': props_count,
                                'total_gas': event['gas_cost'],
                                'cross_province': event.get('cross_province', False)
                            })
                
                if gas_per_property:
                    efficiency_data[approach] = gas_per_property
        
        if not efficiency_data:
            print("   ⚠️  No efficiency data available")
            return
        
        # Create efficiency analysis chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Chart 1: Average gas cost per property
        approaches = list(efficiency_data.keys())
        avg_efficiency = []
        
        for approach in approaches:
            efficiencies = [e['efficiency'] for e in efficiency_data[approach]]
            avg_efficiency.append(np.mean(efficiencies))
        
        colors = [self.color_schemes.get(app, '#808080') for app in approaches]
        bars = ax1.bar(approaches, avg_efficiency, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, eff in zip(bars, avg_efficiency):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(avg_efficiency) * 0.01,
                    f'{eff:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_title('Average Gas Cost per Property\n(Efficiency Comparison)', fontweight='bold')
        ax1.set_ylabel('Gas Cost per Property (wei)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Chart 2: Efficiency vs number of properties (scatter plot)
        for approach in approaches:
            data = efficiency_data[approach]
            properties = [d['properties'] for d in data]
            efficiencies = [d['efficiency'] for d in data]
            
            ax2.scatter(properties, efficiencies, alpha=0.6, s=30,
                       color=self.color_schemes.get(approach, '#808080'),
                       label=approach.replace('_', ' ').title())
        
        ax2.set_title('Gas Efficiency vs Properties Count\n(Scalability Analysis)', fontweight='bold')
        ax2.set_xlabel('Number of Properties')
        ax2.set_ylabel('Gas Cost per Property (wei)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/gas_cost_per_property_{self.timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✅ Gas efficiency analysis completed for {len(approaches)} approaches")
    
    def _create_cross_province_gas_analysis_chart(self, gas_data: Dict[str, Dict[int, List[Dict]]], selected_approaches: List[str]):
        """Create cross-province vs single-province gas cost analysis."""
        print("   📊 Creating cross-province gas analysis...")
        
        # Separate cross-province and single-province events
        cross_province_data = {}
        single_province_data = {}
        
        for approach in selected_approaches:
            if approach in gas_data:
                cross_costs = []
                single_costs = []
                
                for day_data in gas_data[approach].values():
                    for event in day_data:
                        if event.get('cross_province', False):
                            cross_costs.append(event['gas_cost'])
                        else:
                            single_costs.append(event['gas_cost'])
                
                if cross_costs:
                    cross_province_data[approach] = cross_costs
                if single_costs:
                    single_province_data[approach] = single_costs
        
        if not cross_province_data and not single_province_data:
            print("   ⚠️  No province-categorized data available")
            return
        
        # Create comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        approaches = list(set(list(cross_province_data.keys()) + list(single_province_data.keys())))
        
        # Chart 1: Average costs comparison
        cross_averages = []
        single_averages = []
        approach_labels = []
        
        for approach in approaches:
            cross_avg = np.mean(cross_province_data.get(approach, [0])) if approach in cross_province_data else 0
            single_avg = np.mean(single_province_data.get(approach, [0])) if approach in single_province_data else 0
            
            if cross_avg > 0 or single_avg > 0:
                cross_averages.append(cross_avg)
                single_averages.append(single_avg)
                approach_labels.append(approach.replace('_', '\n'))
        
        x = np.arange(len(approach_labels))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, cross_averages, width, label='Cross-Province',
                       color='#FF6B6B', alpha=0.7)
        bars2 = ax1.bar(x + width/2, single_averages, width, label='Single-Province',
                       color='#4ECDC4', alpha=0.7)
        
        ax1.set_title('Gas Cost: Cross-Province vs Single-Province\n(Average Comparison)', fontweight='bold')
        ax1.set_ylabel('Average Gas Cost (wei)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(approach_labels)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Chart 2: Gas cost overhead for cross-province
        overhead_percentages = []
        valid_approaches = []
        
        for i, approach in enumerate(approaches):
            if approach in cross_province_data and approach in single_province_data:
                cross_avg = np.mean(cross_province_data[approach])
                single_avg = np.mean(single_province_data[approach])
                
                if single_avg > 0:
                    overhead = ((cross_avg - single_avg) / single_avg) * 100
                    overhead_percentages.append(overhead)
                    valid_approaches.append(approach.replace('_', '\n'))
        
        if overhead_percentages:
            colors = [self.color_schemes.get(app.replace('\n', '_'), '#808080') for app in valid_approaches]
            bars = ax2.bar(valid_approaches, overhead_percentages, color=colors, alpha=0.7)
            
            # Add value labels
            for bar, overhead in zip(bars, overhead_percentages):
                ax2.text(bar.get_x() + bar.get_width()/2., 
                        bar.get_height() + (max(overhead_percentages) if overhead_percentages else 0) * 0.02,
                        f'{overhead:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax2.set_title('Cross-Province Gas Cost Overhead\n(% Increase vs Single-Province)', fontweight='bold')
            ax2.set_ylabel('Gas Cost Overhead (%)')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cross_province_gas_analysis_{self.timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✅ Cross-province analysis completed for {len(approaches)} approaches")
    
    def _calculate_gas_optimization_summary(self, gas_data: Dict[str, Dict[int, List[Dict]]], selected_approaches: List[str]):
        """Calculate and display gas optimization summary from multi-day data."""
        print("\n📊 Gas Optimization Summary:")
        print("=" * 60)
        
        for approach in selected_approaches:
            if approach not in gas_data:
                continue
            
            # Get all days with data
            days_with_data = sorted([day for day in gas_data[approach].keys() if gas_data[approach][day]])
            
            if len(days_with_data) < 2:
                print(f"   {approach:25}: Insufficient data for optimization analysis")
                continue
            
            # Calculate daily averages
            daily_averages = {}
            for day in days_with_data:
                day_gas_costs = [event['gas_cost'] for event in gas_data[approach][day]]
                daily_averages[day] = np.mean(day_gas_costs)
            
            # Calculate optimization metrics
            first_day_avg = daily_averages[days_with_data[0]]
            last_day_avg = daily_averages[days_with_data[-1]]
            best_day_avg = min(daily_averages.values())
            worst_day_avg = max(daily_averages.values())
            
            # Calculate improvements
            total_improvement = ((first_day_avg - last_day_avg) / first_day_avg * 100) if first_day_avg > 0 else 0
            best_improvement = ((first_day_avg - best_day_avg) / first_day_avg * 100) if first_day_avg > 0 else 0
            
            # Calculate total events and average properties
            total_events = sum(len(gas_data[approach][day]) for day in days_with_data)
            avg_properties = np.mean([
                event.get('properties_count', 0) 
                for day_data in gas_data[approach].values() 
                for event in day_data
            ])
            
            # Calculate efficiency metrics
            overall_avg_gas = np.mean([
                event['gas_cost'] 
                for day_data in gas_data[approach].values() 
                for event in day_data
            ])
            
            gas_per_property = overall_avg_gas / avg_properties if avg_properties > 0 else 0
            
            print(f"   {approach:25}:")
            print(f"      📈 Total Improvement: {total_improvement:6.1f}% (Day 1→{len(days_with_data)})")
            print(f"      🎯 Best Improvement:  {best_improvement:6.1f}% (Day 1→Best)")
            print(f"      💰 Avg Gas Cost:     {overall_avg_gas:8,.0f} wei")
            print(f"      🔧 Gas per Property:  {gas_per_property:8,.0f} wei/prop")
            print(f"      📊 Total Events:      {total_events:8,d} over {len(days_with_data)} days")
            print(f"      📝 Avg Properties:    {avg_properties:8.1f} per event")
            
            # Identify trend
            if total_improvement > 5:
                trend = "📉 Optimizing"
            elif total_improvement < -5:
                trend = "📈 Degrading"
            else:
                trend = "➡️  Stable"
            print(f"      🎭 Trend:            {trend}")
            print()
        
        print("=" * 60)
    
    def create_comprehensive_dashboard(self, reports: Dict[str, Dict[str, Any]]):
        """Create a comprehensive performance dashboard combining all analyses."""
        print("Creating comprehensive performance dashboard...")
        
        # Create dashboard layout
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Comprehensive Performance Dashboard', fontsize=16, fontweight='bold')
        
        # Extract data from all reports
        multi_day_data = reports.get('multi_day', {})
        huffman_data = reports.get('huffman', {})
        gas_data = reports.get('gas', {})
        
        # Dashboard panels
        self._create_dashboard_panel_1(axes[0, 0], multi_day_data)
        self._create_dashboard_panel_2(axes[0, 1], huffman_data)
        self._create_dashboard_panel_3(axes[0, 2], gas_data)
        self._create_dashboard_panel_4(axes[1, 0], multi_day_data)
        self._create_dashboard_panel_5(axes[1, 1], huffman_data)
        self._create_dashboard_panel_6(axes[1, 2], gas_data)
        self._create_dashboard_panel_7(axes[2, 0], multi_day_data)
        self._create_dashboard_panel_8(axes[2, 1], huffman_data)
        self._create_dashboard_panel_9(axes[2, 2], gas_data)
        
        # Adjust layout and save
        plt.tight_layout()
        dashboard_file = f"{self.output_dir}/comprehensive_dashboard_{self.timestamp}.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive dashboard saved to {dashboard_file}")
    
    def _create_daily_performance_chart(self, verification_results: List[Dict[str, Any]], selected_approaches):
        """Create daily performance trend chart."""
        if not verification_results:
            return
        
        # Debug: Log what approaches are actually in the data
        available_approaches = set()
        error_approaches = set()
        for result in verification_results:
            approach = result.get('approach', 'unknown')
            if 'error' in result:
                error_approaches.add(approach)
            else:
                available_approaches.add(approach)
        
        print(f"  📊 Daily Performance Chart - Available approaches: {sorted(available_approaches)}")
        print(f"  ⚠️  Daily Performance Chart - Approaches with errors: {sorted(error_approaches)}")
        
        # Group results by day and approach
        daily_data = defaultdict(lambda: defaultdict(list))
        
        for result in verification_results:
            if 'error' not in result:
                day = result.get('day', 0)
                approach = result.get('approach', 'unknown')
                
                if 'verification_time' in result:
                    daily_data[day][approach].append(result['verification_time'])
                if 'proof_size_bytes' in result:
                    daily_data[day][approach].append(result['proof_size_bytes'])
        
        # Debug: Show data availability per approach per day
        print(f"  📈 Data points per approach per day:")
        for day in sorted(daily_data.keys()):
            day_approaches = {app: len(daily_data[day][app]) for app in daily_data[day] if daily_data[day][app]}
            print(f"    Day {day}: {day_approaches}")
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot verification time trends
        approaches_with_data = []
        for approach in selected_approaches:
            days = []
            avg_times = []
            
            for day in sorted(daily_data.keys()):
                if approach in daily_data[day] and daily_data[day][approach]:
                    days.append(day)
                    avg_times.append(np.mean(daily_data[day][approach]))
            
            if days:
                approaches_with_data.append(approach)
                ax1.plot(days, avg_times, marker='o', label=approach, 
                        color=self.color_schemes.get(approach, '#000000'))
                print(f"    ✅ {approach}: {len(days)} days with data")
            else:
                print(f"    ❌ {approach}: No data available")
        
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Average Verification Time (seconds)')
        ax1.set_title('Daily Verification Time Trends')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot proof size trends
        for approach in selected_approaches:
            days = []
            avg_sizes = []
            
            for day in sorted(daily_data.keys()):
                if approach in daily_data[day] and daily_data[day][approach]:
                    days.append(day)
                    avg_sizes.append(np.mean(daily_data[day][approach]))
            
            if days:
                ax2.plot(days, avg_sizes, marker='s', label=approach,
                        color=self.color_schemes.get(approach, '#000000'))
        
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Average Proof Size (bytes)')
        ax2.set_title('Daily Proof Size Trends')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        print(f"  📊 Daily Performance Chart created with {len(approaches_with_data)} approaches: {approaches_with_data}")
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/daily_performance_trends_{self.timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_optimization_effectiveness_chart(self, optimization_analysis: Dict[str, Any]):
        """Create optimization effectiveness comparison chart."""
        if not optimization_analysis:
            return
        
        # Extract optimization ratios
        optimization_ratios = optimization_analysis.get('optimization_ratios', {})
        
        if not optimization_ratios:
            return
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = []
        values = []
        colors = []
        
        for metric, value in optimization_ratios.items():
            if isinstance(value, (int, float)):
                metrics.append(metric.replace('_', ' ').title())
                values.append(value)
                colors.append(self.color_schemes.get('hierarchical', '#2E8B57'))
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Optimization Effectiveness Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/optimization_effectiveness_{self.timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_traffic_pattern_chart(self, traffic_analysis: Dict[str, Any]):
        """Create traffic pattern analysis chart."""
        if not traffic_analysis:
            return
        
        daily_stats = traffic_analysis.get('daily_stats', {})
        
        if not daily_stats:
            return
        
        # Extract daily data
        days = []
        events = []
        cross_province_events = []
        
        for day_idx, stats in daily_stats.items():
            days.append(f"Day {day_idx + 1}")
            events.append(stats.get('events', 0))
            cross_province_events.append(stats.get('cross_province_events', 0))
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        width = 0.35
        x = np.arange(len(days))
        
        bars1 = ax.bar(x, events, width, label='Total Events', 
                      color=self.color_schemes.get('hierarchical', '#2E8B57'), alpha=0.7)
        bars2 = ax.bar(x, cross_province_events, width, label='Cross-Province Events',
                      color=self.color_schemes.get('traditional_multiproof', '#4169E1'), alpha=0.7)
        
        ax.set_xlabel('Day')
        ax.set_ylabel('Number of Events')
        ax.set_title('Daily Traffic Patterns')
        ax.set_xticks(x)
        ax.set_xticklabels(days)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/traffic_patterns_{self.timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_cross_province_analysis_chart(self, verification_results: List[Dict[str, Any]], selected_approaches):
        """Create cross-province verification analysis chart."""
        if not verification_results:
            return
        
        # Separate cross-province and single-province results
        cross_province_results = [r for r in verification_results if r.get('cross_province', False)]
        single_province_results = [r for r in verification_results if not r.get('cross_province', False)]
        
        # Create comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cross-province performance
        if cross_province_results:
            cross_province_times = []
            cross_province_sizes = []
            
            for approach in selected_approaches:
                approach_results = [r for r in cross_province_results if r.get('approach') == approach]
                if approach_results:
                    times = [r.get('verification_time', 0) for r in approach_results if 'verification_time' in r]
                    sizes = [r.get('proof_size_bytes', 0) for r in approach_results if 'proof_size_bytes' in r]
                    
                    cross_province_times.append(np.mean(times) if times else 0)
                    cross_province_sizes.append(np.mean(sizes) if sizes else 0)
                else:
                    cross_province_times.append(0)
                    cross_province_sizes.append(0)
            
            x = np.arange(len(selected_approaches))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, cross_province_times, width, label='Verification Time',
                           color=self.color_schemes.get('hierarchical', '#2E8B57'), alpha=0.7)
            bars2 = ax1.bar(x + width/2, [s/1000 for s in cross_province_sizes], width, label='Proof Size (KB)',
                           color=self.color_schemes.get('traditional_multiproof', '#4169E1'), alpha=0.7)
            
            ax1.set_xlabel('Approach')
            ax1.set_ylabel('Performance Metric')
            ax1.set_title('Cross-Province Verification Performance')
            ax1.set_xticks(x)
            ax1.set_xticklabels([a.replace('_', '\n') for a in selected_approaches])
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
        
        # Single-province performance
        if single_province_results:
            single_province_times = []
            single_province_sizes = []
            
            for approach in selected_approaches:
                approach_results = [r for r in single_province_results if r.get('approach') == approach]
                if approach_results:
                    times = [r.get('verification_time', 0) for r in approach_results if 'verification_time' in r]
                    sizes = [r.get('proof_size_bytes', 0) for r in approach_results if 'proof_size_bytes' in r]
                    
                    single_province_times.append(np.mean(times) if times else 0)
                    single_province_sizes.append(np.mean(sizes) if sizes else 0)
                else:
                    single_province_times.append(0)
                    single_province_sizes.append(0)
            
            x = np.arange(len(selected_approaches))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, single_province_times, width, label='Verification Time',
                           color=self.color_schemes.get('hierarchical', '#2E8B57'), alpha=0.7)
            bars2 = ax2.bar(x + width/2, [s/1000 for s in single_province_sizes], width, label='Proof Size (KB)',
                           color=self.color_schemes.get('traditional_multiproof', '#4169E1'), alpha=0.7)
            
            ax2.set_xlabel('Approach')
            ax2.set_ylabel('Performance Metric')
            ax2.set_title('Single-Province Verification Performance')
            ax2.set_xticks(x)
            ax2.set_xticklabels([a.replace('_', '\n') for a in selected_approaches])
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cross_province_analysis_{self.timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_comparison_chart(self, verification_results: List[Dict[str, Any]]):
        """Create overall performance comparison chart."""
        if not verification_results:
            return
        
        # Group results by approach
        approach_data = defaultdict(list)
        
        for result in verification_results:
            if 'error' not in result:
                approach = result.get('approach', 'unknown')
                if 'verification_time' in result:
                    approach_data[approach].append({
                        'verification_time': result['verification_time'],
                        'proof_size_bytes': result.get('proof_size_bytes', 0),
                        'properties_count': result.get('properties_count', 0),
                        'provinces_count': result.get('provinces_count', 0)
                    })
        
        # Create performance comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        approaches = list(approach_data.keys())
        colors = [self.color_schemes.get(approach, '#808080') for approach in approaches]
        
        # Verification time comparison
        times = [np.mean([r['verification_time'] for r in data]) for data in approach_data.values()]
        ax1.bar(approaches, times, color=colors, alpha=0.7)
        ax1.set_ylabel('Average Verification Time (seconds)')
        ax1.set_title('Verification Time Comparison')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Proof size comparison
        sizes = [np.mean([r['proof_size_bytes'] for r in data]) for data in approach_data.values()]
        ax2.bar(approaches, [s/1000 for s in sizes], color=colors, alpha=0.7)
        ax2.set_ylabel('Average Proof Size (KB)')
        ax2.set_title('Proof Size Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Properties per verification
        props = [np.mean([r['properties_count'] for r in data]) for data in approach_data.values()]
        ax3.bar(approaches, props, color=colors, alpha=0.7)
        ax3.set_ylabel('Average Properties per Verification')
        ax3.set_title('Verification Scale Comparison')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Provinces per verification
        provinces = [np.mean([r['provinces_count'] for r in data]) for data in approach_data.values()]
        ax4.bar(approaches, provinces, color=colors, alpha=0.7)
        ax4.set_ylabel('Average Provinces per Verification')
        ax4.set_title('Cross-Province Scale Comparison')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/performance_comparison_{self.timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_co_verification_pattern_chart(self, co_verification_analysis: Dict[str, Any]):
        """Create co-verification pattern analysis chart."""
        if not co_verification_analysis:
            return
        
        # Extract top co-verification pairs
        top_pairs = co_verification_analysis.get('top_co_verification_pairs', [])
        
        if not top_pairs:
            return
        
        # Create horizontal bar chart for top pairs
        fig, ax = plt.subplots(figsize=(12, 8))
        
        pairs = [f"{pair[0][:20]}...\n{pair[1][:20]}..." for pair, freq in top_pairs[:15]]
        frequencies = [freq for pair, freq in top_pairs[:15]]
        
        bars = ax.barh(pairs, frequencies, color=self.color_schemes.get('hierarchical', '#2E8B57'), alpha=0.7)
        
        # Add frequency labels
        for bar, freq in zip(bars, frequencies):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{freq}', ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel('Co-verification Frequency')
        ax.set_title('Top Property Co-verification Pairs')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/co_verification_patterns_{self.timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_optimization_scenario_chart(self, optimization_results: Dict[str, Any]):
        """Create optimization scenario comparison chart."""
        if not optimization_results:
            return
        
        # Extract scenario data
        scenarios = []
        hierarchical_performance = []
        traditional_performance = []
        
        for scenario_name, scenario_data in optimization_results.items():
            scenarios.append(scenario_name.replace('_', ' ').title())
            
            approach_results = scenario_data.get('approach_results', {})
            
            if 'hierarchical' in approach_results:  # FIXED: Use consistent naming
                hierarchical_performance.append(
                    approach_results['hierarchical'].get('avg_proof_size', 0)
                )
            else:
                hierarchical_performance.append(0)
            
            if 'traditional_multiproof' in approach_results:
                traditional_performance.append(
                    approach_results['traditional_multiproof'].get('avg_proof_size', 0)
                )
            else:
                traditional_performance.append(0)
        
        # Create comparison chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, [p/1000 for p in hierarchical_performance], width,
                      label='Optimized Hierarchical', color=self.color_schemes.get('hierarchical', '#2E8B57'), alpha=0.7)
        bars2 = ax.bar(x + width/2, [p/1000 for p in traditional_performance], width,
                      label='Traditional Multiproof', color=self.color_schemes.get('traditional_multiproof', '#4169E1'), alpha=0.7)
        
        ax.set_xlabel('Test Scenario')
        ax.set_ylabel('Average Proof Size (KB)')
        ax.set_title('Optimization Effectiveness by Scenario')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/optimization_scenarios_{self.timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_proof_size_comparison_chart(self, proof_analysis: Dict[str, Any], verification_results: List[Dict[str, Any]] = None):
        """Create proof size comparison chart using consistent data source."""
        
        # PRIORITY 1: Use verification_results if available (same as daily trends)
        if verification_results:
            return self._create_proof_size_comparison_from_verification_results(verification_results)
        
        # PRIORITY 2: Fallback to proof_analysis data
        if not proof_analysis:
            return
        
        approach_comparison = proof_analysis.get('approach_comparison', {})
        
        if not approach_comparison:
            return
        
        # Create box plot for proof sizes
        fig, ax = plt.subplots(figsize=(10, 6))
        
        approaches = list(approach_comparison.keys())
        proof_sizes = []
        
        for approach in approaches:
            avg_size = approach_comparison[approach].get('avg_proof_size', 0)
            proof_sizes.append(avg_size / 1000)  # Convert to KB
        
        colors = [self.color_schemes.get(approach, '#808080') for approach in approaches]
        bars = ax.bar(approaches, proof_sizes, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, size in zip(bars, proof_sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{size:.1f} KB', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Average Proof Size (KB)')
        ax.set_title('Proof Size Comparison Across Approaches')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/proof_size_comparison_{self.timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_proof_size_comparison_from_verification_results(self, verification_results: List[Dict[str, Any]]):
        """Create proof size comparison chart using the same data as daily trends."""
        
        # Group proof sizes by approach (same logic as daily performance chart)
        approach_data = defaultdict(list)
        
        for result in verification_results:
            if 'error' not in result and 'proof_size_bytes' in result:
                approach = result.get('approach', 'unknown')
                proof_size = result['proof_size_bytes']
                approach_data[approach].append(proof_size)
        
        if not approach_data:
            return
        
        # Create comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Chart 1: Average proof sizes (matches daily trends)
        approaches = list(approach_data.keys())
        avg_sizes = [np.mean(sizes) / 1000 for sizes in approach_data.values()]  # Convert to KB
        
        colors = [self.color_schemes.get(approach, '#808080') for approach in approaches]
        bars = ax1.bar(approaches, avg_sizes, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, size in zip(bars, avg_sizes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{size:.1f} KB', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_ylabel('Average Proof Size (KB)')
        ax1.set_title('Average Proof Size Comparison\n(Same data as Daily Trends)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Chart 2: Distribution box plots
        box_data = [sizes for sizes in approach_data.values()]
        box_labels = list(approaches)
        
        # Convert to KB for display
        box_data_kb = [[size / 1000 for size in sizes] for sizes in box_data]
        
        box_plot = ax2.boxplot(box_data_kb, labels=box_labels, patch_artist=True)
        
        # Color the boxes
        for patch, approach in zip(box_plot['boxes'], approaches):
            patch.set_facecolor(self.color_schemes.get(approach, '#808080'))
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Proof Size Distribution (KB)')
        ax2.set_title('Proof Size Distribution by Approach\n(Shows variability across events)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/proof_size_comparison_consistent_{self.timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary for debugging
        print(f"\n📊 Proof Size Comparison (Consistent Data):")
        for approach, sizes in approach_data.items():
            avg_kb = np.mean(sizes) / 1000
            min_kb = np.min(sizes) / 1000
            max_kb = np.max(sizes) / 1000
            count = len(sizes)
            print(f"   {approach:25}: {avg_kb:6.1f} KB avg ({min_kb:5.1f}-{max_kb:6.1f} KB, {count:3d} events)")
    
    def _create_optimization_potential_chart(self, co_verification_analysis: Dict[str, Any]):
        """Create optimization potential analysis chart."""
        if not co_verification_analysis:
            return
        
        optimization_potential = co_verification_analysis.get('optimization_potential', {})
        
        if not optimization_potential:
            return
        
        # Create pie chart for optimization potential
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = ['Frequent Pairs', 'High Frequency Pairs', 'Other Pairs']
        sizes = [
            optimization_potential.get('frequent_pairs_ratio', 0) * 100,
            optimization_potential.get('high_frequency_pairs_ratio', 0) * 100,
            100 - (optimization_potential.get('frequent_pairs_ratio', 0) * 100)
        ]
        
        colors = [self.color_schemes.get('hierarchical', '#2E8B57'),
                 self.color_schemes.get('traditional_multiproof', '#4169E1'),
                 self.color_schemes.get('baseline', '#808080')]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                         startangle=90)
        
        ax.set_title('Property Co-verification Optimization Potential')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/optimization_potential_{self.timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_gas_cost_comparison_chart(self, gas_results: Dict[str, Any], selected_approaches: List[str] = None):
        """Create gas cost comparison chart showing adaptive optimization benefits."""
        if not gas_results:
            return
        
        print("📊 Creating adaptive gas cost comparison chart...")
        
        # Check if this is multi-day adaptive data
        is_adaptive = gas_results.get('test_metadata', {}).get('adaptive_optimization', False)
        
        if is_adaptive:
            self._create_adaptive_gas_comparison_chart(gas_results, selected_approaches)
        else:
            self._create_traditional_gas_comparison_chart(gas_results, selected_approaches)
    
    def _create_adaptive_gas_comparison_chart(self, gas_results: Dict[str, Any], selected_approaches: List[str] = None):
        """Create gas cost comparison chart highlighting adaptive optimization benefits."""
        gas_data = gas_results.get('gas_results', {})
        optimization_analysis = gas_results.get('optimization_analysis', {})
        
        # Debug: Print data structure to identify the mismatch
        print(f"🔍 Gas data debug:")
        for approach, data in gas_data.items():
            daily_prog = data.get('daily_progression', [])
            valid_days = len([d for d in daily_prog if d and d.get('average_verification_gas', 0) > 0])
            print(f"   {approach}: {len(daily_prog)} total days, {valid_days} with gas data")
        
        if not gas_data:
            print("⚠️  No gas data found for adaptive comparison")
            return
        
        # Filter approaches
        if selected_approaches is None:
            selected_approaches = list(gas_data.keys())
        else:
            # Filter to only include approaches we have data for
            selected_approaches = [app for app in selected_approaches if app in gas_data]
        
        if not selected_approaches:
            print("⚠️  No valid approaches found for gas comparison")
            return
        
        # Create figure with subplots for comparison and daily progression
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Subplot 1: Final optimized gas costs comparison
        approach_names = []
        final_gas_costs = []
        improvements = []
        
        for approach in selected_approaches:
            approach_data = gas_data[approach]
            final_results = approach_data.get('approach_results', {})
            
            if 'average_verification_gas' in final_results:
                approach_names.append(approach.replace('_', ' ').title())
                final_gas_costs.append(final_results['average_verification_gas'])
                
                # Get improvement percentage
                daily_improvements = optimization_analysis.get('daily_improvements', {}).get(approach, {})
                improvement_pct = daily_improvements.get('gas_reduction_percent', 0)
                improvements.append(improvement_pct)
        
        if approach_names and final_gas_costs:
            bars1 = ax1.bar(approach_names, final_gas_costs, 
                           color=[self.color_schemes.get(app.lower().replace(' ', '_'), '#808080') 
                                 for app in approach_names], alpha=0.7)
            
            # Add improvement percentages as text on bars
            for bar, improvement in zip(bars1, improvements):
                if improvement > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_gas_costs) * 0.01,
                            f'-{improvement:.1f}%', ha='center', va='bottom', 
                            fontweight='bold', color='green', fontsize=10)
            
            ax1.set_title('Final Optimized Gas Costs\n(After Daily Learning)', fontweight='bold')
            ax1.set_ylabel('Average Gas per Verification')
            ax1.grid(True, alpha=0.3, axis='y')
        
        # Subplot 2: Daily progression for approaches with multi-day data
        daily_progression_data = {}
        max_days = 0
        
        for approach in selected_approaches:
            approach_data = gas_data[approach]
            daily_progression = approach_data.get('daily_progression', [])
            
            if len(daily_progression) > 1:  # Only show if we have multi-day data
                gas_progression = []
                for day_data in daily_progression:
                    gas_value = day_data.get('average_verification_gas', 0) if day_data else 0
                    gas_progression.append(gas_value)
                
                if any(g > 0 for g in gas_progression):  # Only include if we have actual data
                    daily_progression_data[approach] = gas_progression
                    max_days = max(max_days, len(gas_progression))
        
        if daily_progression_data and max_days > 1:
            days = list(range(1, max_days + 1))
            
            for approach, progression in daily_progression_data.items():
                # Pad progression to match max_days if needed
                while len(progression) < max_days:
                    progression.append(0)
                
                # Show all days, but use None for missing/zero values to create gaps
                plot_days = []
                plot_values = []
                
                for i, val in enumerate(progression):
                    day_label = days[i]
                    if val > 0:
                        plot_days.append(day_label)
                        plot_values.append(val)
                    else:
                        # Add None to create gaps in the line for missing data
                        if plot_days:  # Only add gap if we have previous data
                            plot_days.append(day_label)
                            plot_values.append(None)
                
                if plot_values and any(v is not None for v in plot_values):
                    # Filter out None values for matplotlib (but keep day alignment)
                    clean_days = []
                    clean_values = []
                    for day, val in zip(plot_days, plot_values):
                        if val is not None:
                            clean_days.append(day)
                            clean_values.append(val)
                    
                    if clean_values:
                        ax2.plot(clean_days, clean_values, 
                                marker='o', linewidth=2, markersize=6,
                                label=approach.replace('_', ' ').title(),
                                color=self.color_schemes.get(approach, '#808080'))
            
            ax2.set_title('Daily Gas Cost Optimization\n(Adaptive Learning Progress)', fontweight='bold')
            ax2.set_xlabel('Day')
            ax2.set_ylabel('Average Gas per Verification')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Ensure x-axis shows all days consistently with traffic pattern chart
            # Try to get the number of days from the actual test configuration
            total_days = max_days
            try:
                # Check if we can get the actual number of days from the data structure
                if hasattr(self, 'expected_days'):
                    total_days = self.expected_days
                elif any('7' in str(k) for k in daily_progression_data.keys()):
                    total_days = 7  # Common test configuration
            except:
                pass
            
            ax2.set_xticks(range(1, total_days + 1))
            ax2.set_xlim(0.5, total_days + 0.5)
        else:
            # No multi-day data available
            ax2.text(0.5, 0.5, 'No multi-day progression data\navailable for comparison', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=12, style='italic')
            ax2.set_title('Daily Progression (No Data)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/adaptive_gas_cost_comparison_{self.timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Adaptive gas comparison chart saved")
    
    def _create_traditional_gas_comparison_chart(self, gas_results: Dict[str, Any], selected_approaches: List[str] = None):
        """Create traditional gas cost comparison chart (fallback for non-adaptive data)."""
        if selected_approaches is None or len(selected_approaches) == 0:
            # Use all available approaches from the gas results data
            available_approaches = set()
            for scenario_data in gas_results.values():
                if isinstance(scenario_data, dict) and 'approach_results' in scenario_data:
                    available_approaches.update(scenario_data['approach_results'].keys())
            selected_approaches = list(available_approaches) if available_approaches else ['hierarchical', 'clustered_flat']
        
        # Extract gas cost data by scenario
        scenarios = []
        approach_costs = {approach: [] for approach in selected_approaches}
        
        for scenario_name, scenario_data in gas_results.items():
            scenarios.append(scenario_name.replace('_', ' ').title())
            
            approach_results = scenario_data.get('approach_results', {})
            
            for approach in selected_approaches:
                cost = approach_results.get(approach, {}).get('avg_gas_cost_usd', 0)
                approach_costs[approach].append(cost)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(scenarios))
        width = 0.8 / len(selected_approaches)  # Adjust width based on number of approaches
        
        bars = []
        for i, approach in enumerate(selected_approaches):
            offset = (i - len(selected_approaches)/2 + 0.5) * width
            bars.append(ax.bar(x + offset, approach_costs[approach], width, 
                              label=approach.replace('_', ' ').title(),
                              color=self.color_schemes.get(approach, '#808080'), alpha=0.7))
        
        ax.set_xlabel('Test Scenario')
        ax.set_ylabel('Average Gas Cost (USD)')
        ax.set_title('Gas Cost Comparison by Scenario')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/gas_cost_comparison_{self.timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_cost_optimization_chart(self, optimization_analysis: Dict[str, Any]):
        """Create cost optimization analysis chart."""
        if not optimization_analysis:
            return
        
        scenario_comparisons = optimization_analysis.get('scenario_comparisons', {})
        
        if not scenario_comparisons:
            return
        
        # Extract optimization data
        scenarios = []
        gas_reductions = []
        cost_reductions = []
        
        for scenario_name, comparison in scenario_comparisons.items():
            scenarios.append(scenario_name.replace('_', ' ').title())
            gas_reductions.append(comparison.get('gas_reduction_percent', 0))
            cost_reductions.append(comparison.get('cost_reduction_percent', 0))
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, gas_reductions, width, label='Gas Reduction (%)',
                      color=self.color_schemes.get('hierarchical', '#2E8B57'), alpha=0.7)
        bars2 = ax.bar(x + width/2, cost_reductions, width, label='Cost Reduction (%)',
                      color=self.color_schemes.get('traditional_multiproof', '#4169E1'), alpha=0.7)
        
        ax.set_xlabel('Test Scenario')
        ax.set_ylabel('Reduction (%)')
        ax.set_title('Cost Optimization by Scenario')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cost_optimization_{self.timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_scalability_analysis_chart(self, scalability_analysis: Dict[str, Any]):
        """Create scalability analysis chart."""
        if not scalability_analysis:
            return
        
        # Extract scalability data
        scales = sorted(scalability_analysis.keys())
        hierarchical_costs = []
        traditional_costs = []
        single_proof_costs = []
        
        for scale in scales:
            scale_data = scalability_analysis[scale]
            
            hierarchical_costs.append(
                scale_data.get('hierarchical', {}).get('avg_cost_per_event', 0)
            )
            traditional_costs.append(
                scale_data.get('traditional_multiproof', {}).get('avg_cost_per_event', 0)
            )
            single_proof_costs.append(
                scale_data.get('traditional_single_proof', {}).get('avg_cost_per_event', 0)
            )
        
        # Create line chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(scales, hierarchical_costs, marker='o', label='Hierarchical',
               color=self.color_schemes.get('hierarchical', '#2E8B57'), linewidth=2)
        ax.plot(scales, traditional_costs, marker='s', label='Traditional Multiproof',
               color=self.color_schemes.get('traditional_multiproof', '#4169E1'), linewidth=2)
        ax.plot(scales, single_proof_costs, marker='^', label='Traditional Single Proof',
               color=self.color_schemes.get('traditional_single_proof', '#DC143C'), linewidth=2)
        
        ax.set_xlabel('Number of Events')
        ax.set_ylabel('Average Cost per Event (USD)')
        ax.set_title('Scalability Analysis: Cost per Event')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/scalability_analysis_{self.timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_cost_benefit_analysis_chart(self, optimization_analysis: Dict[str, Any]):
        """Create cost-benefit analysis chart."""
        if not optimization_analysis:
            return
        
        overall_optimization = optimization_analysis.get('overall_optimization', {})
        
        if not overall_optimization:
            return
        
        # Create cost-benefit matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract metrics
        gas_reduction = overall_optimization.get('avg_gas_reduction_percent', 0)
        cost_reduction = overall_optimization.get('avg_cost_reduction_percent', 0)
        max_optimization = overall_optimization.get('max_gas_reduction_percent', 0)
        
        # Create bubble chart
        categories = ['Average Gas\nReduction', 'Average Cost\nReduction', 'Maximum\nOptimization']
        values = [gas_reduction, cost_reduction, max_optimization]
        colors = [self.color_schemes.get('hierarchical', '#2E8B57'),
                 self.color_schemes.get('traditional_multiproof', '#4169E1'),
                 self.color_schemes.get('traditional_single_proof', '#DC143C')]
        
        # Create bars with different heights
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_ylabel('Optimization Percentage (%)')
        ax.set_title('Cost-Benefit Analysis: Optimization Effectiveness')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add threshold lines
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='High Impact Threshold')
        ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Medium Impact Threshold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cost_benefit_analysis_{self.timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Dashboard panel methods
    def _create_dashboard_panel_1(self, ax, data):
        """Dashboard panel 1: Daily performance trends."""
        ax.set_title('Daily Performance Trends', fontweight='bold')
        ax.text(0.5, 0.5, 'Daily Performance\nTrends Chart', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _create_dashboard_panel_2(self, ax, data):
        """Dashboard panel 2: Optimization effectiveness."""
        ax.set_title('Optimization Effectiveness', fontweight='bold')
        ax.text(0.5, 0.5, 'Optimization\nEffectiveness Chart', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _create_dashboard_panel_3(self, ax, data):
        """Dashboard panel 3: Gas cost comparison."""
        ax.set_title('Gas Cost Comparison', fontweight='bold')
        ax.text(0.5, 0.5, 'Gas Cost\nComparison Chart', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _create_dashboard_panel_4(self, ax, data):
        """Dashboard panel 4: Traffic patterns."""
        ax.set_title('Traffic Patterns', fontweight='bold')
        ax.text(0.5, 0.5, 'Traffic Patterns\nChart', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _create_dashboard_panel_5(self, ax, data):
        """Dashboard panel 5: Co-verification patterns."""
        ax.set_title('Co-verification Patterns', fontweight='bold')
        ax.text(0.5, 0.5, 'Co-verification\nPatterns Chart', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _create_dashboard_panel_6(self, ax, data):
        """Dashboard panel 6: Cost optimization."""
        ax.set_title('Cost Optimization', fontweight='bold')
        ax.text(0.5, 0.5, 'Cost Optimization\nChart', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _create_dashboard_panel_7(self, ax, data):
        """Dashboard panel 7: Cross-province analysis."""
        ax.set_title('Cross-province Analysis', fontweight='bold')
        ax.text(0.5, 0.5, 'Cross-province\nAnalysis Chart', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _create_dashboard_panel_8(self, ax, data):
        """Dashboard panel 8: Optimization scenarios."""
        ax.set_title('Optimization Scenarios', fontweight='bold')
        ax.text(0.5, 0.5, 'Optimization\nScenarios Chart', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _create_dashboard_panel_9(self, ax, data):
        """Dashboard panel 9: Scalability analysis."""
        ax.set_title('Scalability Analysis', fontweight='bold')
        ax.text(0.5, 0.5, 'Scalability\nAnalysis Chart', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

def main():
    """Main execution function for performance visualization."""
    print("=== PERFORMANCE VISUALIZATION MODULE ===")
    
    # Create visualizer
    visualizer = PerformanceVisualizer()
    
    # Example usage - you would load actual reports here
    print("Performance visualizer ready!")
    print("Use the following methods to create visualizations:")
    print("  - visualize_multi_day_results(multi_day_report)")
    print("  - visualize_pairs_huffman_analysis(huffman_report)")
    print("  - visualize_gas_cost_analysis(gas_report)")
    print("  - create_comprehensive_dashboard(reports_dict)")

if __name__ == "__main__":
    main()

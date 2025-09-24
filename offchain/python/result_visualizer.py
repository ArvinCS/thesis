#!/usr/bin/env python3
"""
Cross-Province Test Results Visualizer

This module provides comprehensive visualization and analysis for cross-province
test results, generating graphs and metrics for:
- Gas usage analysis (estimated vs actual)
- Verification time comparison
- Proof size analysis
- Performance trends over time
- Approach comparison
- Day-by-day analysis
"""

import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import glob
import argparse
import sys

class CrossProvinceResultsVisualizer:
    """
    Comprehensive visualizer for cross-province test results.
    """
    
    def __init__(self, results_directory: str = "report", single_file: str = None):
        self.results_directory = Path(results_directory)
        self.single_file = single_file
        
        # Create organized output directory structure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_directory = Path("visualization_outputs") / f"analysis_{timestamp}"
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different types of outputs
        self.charts_dir = self.output_directory / "charts"
        self.reports_dir = self.output_directory / "reports"  
        self.data_dir = self.output_directory / "data"
        
        for dir_path in [self.charts_dir, self.reports_dir, self.data_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Set up matplotlib style for better looking charts
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Configure matplotlib for high-quality outputs
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['font.size'] = 10
        
    def load_results(self) -> List[Dict]:
        """Load cross-province test results - either single file or all from directory."""
        results = []
        
        if self.single_file:
            # Load single specified file
            file_path = Path(self.single_file)
            if not file_path.exists():
                print(f"❌ File not found: {file_path}")
                return results
                
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['file_path'] = str(file_path)
                    data['file_name'] = file_path.name
                    results.append(data)
                    print(f"✅ Loaded single file: {file_path}")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                
        else:
            # Load all files from directory (original behavior)
            pattern = str(self.results_directory / "**" / "*cross_province_massive_test*.json")
            
            for file_path in glob.glob(pattern, recursive=True):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        data['file_path'] = file_path
                        data['file_name'] = os.path.basename(file_path)
                        results.append(data)
                        print(f"✅ Loaded: {file_path}")
                except Exception as e:
                    print(f"❌ Error loading {file_path}: {e}")
        
        print(f"📊 Total results loaded: {len(results)}")
        return results
    
    def list_available_reports(self) -> List[str]:
        """List all available cross-province test reports."""
        pattern = str(self.results_directory / "**" / "*cross_province_massive_test*.json")
        reports = []
        
        print("📋 Available Cross-Province Test Reports:")
        print("=" * 60)
        
        for i, file_path in enumerate(glob.glob(pattern, recursive=True), 1):
            file_path = Path(file_path)
            try:
                # Extract timestamp from filename
                filename = file_path.name
                if "cross_province_massive_test_" in filename:
                    timestamp_part = filename.replace("cross_province_massive_test_", "").replace(".json", "")
                    try:
                        # Parse timestamp
                        timestamp = datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
                        date_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        date_str = timestamp_part
                else:
                    date_str = "Unknown"
                
                # Get file size
                size_kb = file_path.stat().st_size / 1024
                
                print(f"{i:2d}. {file_path}")
                print(f"    📅 Date: {date_str}")
                print(f"    📁 Size: {size_kb:.1f} KB")
                print()
                
                reports.append(str(file_path))
                
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
        
        if not reports:
            print("No cross-province test reports found.")
        else:
            print(f"Total: {len(reports)} reports found")
            
        return reports
    
    def extract_verification_data(self, results: List[Dict]) -> pd.DataFrame:
        """Extract verification results into a pandas DataFrame for analysis."""
        rows = []
        
        for result in results:
            test_metadata = result.get('test_metadata', {})
            execution_time = test_metadata.get('execution_time', '')
            document_count = test_metadata.get('document_count', 0)
            traffic_events = test_metadata.get('traffic_events', 0)
            
            # Extract verification results from multi_day_results
            verification_results = result.get('multi_day_results', {}).get('verification_results', [])
            
            for verification in verification_results:
                row = {
                    'test_date': execution_time,
                    'document_count': document_count,
                    'traffic_events': traffic_events,
                    'approach': verification.get('approach', ''),
                    'event_id': verification.get('event', 0),
                    'properties_count': verification.get('properties_count', 0),
                    'provinces_count': verification.get('provinces_count', 0),
                    'is_4plus_province': verification.get('is_4plus_province', False),
                    'verification_time': verification.get('verification_time', 0),
                    'proof_size_bytes': verification.get('proof_size_bytes', 0),
                    'estimated_gas': verification.get('estimated_gas', 0),
                    'actual_gas_used': verification.get('actual_gas_used', 0),
                    'local_verification_passed': verification.get('local_verification_passed', False),
                    'provinces_list': ','.join(verification.get('provinces_list', [])),
                    'file_name': result.get('file_name', '')
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        if not df.empty and 'test_date' in df.columns:
            df['test_date'] = pd.to_datetime(df['test_date'])
        
        return df
    
    def create_gas_analysis_charts(self, df: pd.DataFrame):
        """Create comprehensive gas usage analysis charts."""
        if df.empty:
            print("❌ No data available for gas analysis")
            return
        
        # 1. Gas Usage Comparison by Approach
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Gas Usage Analysis by Approach', fontsize=16, fontweight='bold')
        
        # Estimated vs Actual Gas
        approach_gas = df.groupby('approach').agg({
            'estimated_gas': 'mean',
            'actual_gas_used': 'mean'
        }).reset_index()
        
        x = np.arange(len(approach_gas))
        width = 0.35
        
        axes[0,0].bar(x - width/2, approach_gas['estimated_gas'], width, 
                     label='Estimated Gas', alpha=0.8)
        axes[0,0].bar(x + width/2, approach_gas['actual_gas_used'], width, 
                     label='Actual Gas Used', alpha=0.8)
        axes[0,0].set_xlabel('Approach')
        axes[0,0].set_ylabel('Gas Units')
        axes[0,0].set_title('Average Gas Usage: Estimated vs Actual')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(approach_gas['approach'], rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Gas efficiency by province count
        if 'provinces_count' in df.columns:
            gas_by_provinces = df.groupby(['approach', 'provinces_count'])['actual_gas_used'].mean().unstack()
            gas_by_provinces.plot(kind='bar', ax=axes[0,1], width=0.8)
            axes[0,1].set_title('Gas Usage by Province Count')
            axes[0,1].set_xlabel('Approach')
            axes[0,1].set_ylabel('Average Gas Used')
            axes[0,1].legend(title='Province Count')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Gas vs Properties Count
        for approach in df['approach'].unique():
            approach_data = df[df['approach'] == approach]
            axes[1,0].scatter(approach_data['properties_count'], 
                            approach_data['actual_gas_used'], 
                            label=approach, alpha=0.7)
        axes[1,0].set_xlabel('Properties Count')
        axes[1,0].set_ylabel('Actual Gas Used')
        axes[1,0].set_title('Gas Usage vs Properties Count')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Gas efficiency ranking
        gas_efficiency = df.groupby('approach')['actual_gas_used'].mean().sort_values()
        axes[1,1].barh(range(len(gas_efficiency)), gas_efficiency.values)
        axes[1,1].set_yticks(range(len(gas_efficiency)))
        axes[1,1].set_yticklabels(gas_efficiency.index)
        axes[1,1].set_xlabel('Average Gas Used')
        axes[1,1].set_title('Gas Efficiency Ranking (Lower is Better)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = self.charts_dir / 'gas_analysis.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved: {chart_path}")
        plt.close()  # Close to free memory
        
    def create_performance_comparison_charts(self, df: pd.DataFrame):
        """Create performance comparison charts."""
        if df.empty:
            print("❌ No data available for performance analysis")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. Verification Time by Approach
        df.boxplot(column='verification_time', by='approach', ax=axes[0,0])
        axes[0,0].set_title('Verification Time Distribution by Approach')
        axes[0,0].set_xlabel('Approach')
        axes[0,0].set_ylabel('Verification Time (seconds)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Proof Size by Approach
        df.boxplot(column='proof_size_bytes', by='approach', ax=axes[0,1])
        axes[0,1].set_title('Proof Size Distribution by Approach')
        axes[0,1].set_xlabel('Approach')
        axes[0,1].set_ylabel('Proof Size (bytes)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Performance vs Scale (properties count)
        for approach in df['approach'].unique():
            approach_data = df[df['approach'] == approach]
            axes[1,0].scatter(approach_data['properties_count'], 
                            approach_data['verification_time'], 
                            label=approach, alpha=0.7)
        axes[1,0].set_xlabel('Properties Count')
        axes[1,0].set_ylabel('Verification Time (seconds)')
        axes[1,0].set_title('Verification Time vs Scale')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Efficiency Matrix (Gas vs Time)
        for approach in df['approach'].unique():
            approach_data = df[df['approach'] == approach]
            axes[1,1].scatter(approach_data['verification_time'], 
                            approach_data['actual_gas_used'], 
                            label=approach, alpha=0.7, s=50)
        axes[1,1].set_xlabel('Verification Time (seconds)')
        axes[1,1].set_ylabel('Actual Gas Used')
        axes[1,1].set_title('Efficiency Matrix: Time vs Gas')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = self.charts_dir / 'performance_comparison.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved: {chart_path}")
        plt.close()  # Close to free memory
        
    def create_time_series_analysis(self, df: pd.DataFrame):
        """Create time series analysis for day-by-day trends."""
        if df.empty or 'test_date' not in df.columns:
            print("❌ No time series data available")
            return
            
        # Group by date and approach
        daily_metrics = df.groupby([df['test_date'].dt.date, 'approach']).agg({
            'actual_gas_used': 'mean',
            'verification_time': 'mean',
            'proof_size_bytes': 'mean',
            'properties_count': 'mean'
        }).reset_index()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Time Series Analysis - Daily Trends', fontsize=16, fontweight='bold')
        
        # 1. Daily Gas Usage Trends
        for approach in daily_metrics['approach'].unique():
            approach_data = daily_metrics[daily_metrics['approach'] == approach]
            axes[0,0].plot(approach_data['test_date'], approach_data['actual_gas_used'], 
                          marker='o', label=approach, linewidth=2)
        axes[0,0].set_title('Daily Average Gas Usage')
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('Average Gas Used')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Daily Verification Time Trends
        for approach in daily_metrics['approach'].unique():
            approach_data = daily_metrics[daily_metrics['approach'] == approach]
            axes[0,1].plot(approach_data['test_date'], approach_data['verification_time'], 
                          marker='s', label=approach, linewidth=2)
        axes[0,1].set_title('Daily Average Verification Time')
        axes[0,1].set_xlabel('Date')
        axes[0,1].set_ylabel('Average Verification Time (s)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Daily Proof Size Trends
        for approach in daily_metrics['approach'].unique():
            approach_data = daily_metrics[daily_metrics['approach'] == approach]
            axes[1,0].plot(approach_data['test_date'], approach_data['proof_size_bytes'], 
                          marker='^', label=approach, linewidth=2)
        axes[1,0].set_title('Daily Average Proof Size')
        axes[1,0].set_xlabel('Date')
        axes[1,0].set_ylabel('Average Proof Size (bytes)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Daily Test Scale
        test_scale = df.groupby(df['test_date'].dt.date).agg({
            'document_count': 'first',
            'traffic_events': 'first'
        }).reset_index()
        
        ax2 = axes[1,1].twinx()
        bars1 = axes[1,1].bar(test_scale['test_date'], test_scale['document_count'], 
                             alpha=0.7, label='Documents', color='skyblue')
        bars2 = ax2.bar(test_scale['test_date'], test_scale['traffic_events'], 
                       alpha=0.7, label='Traffic Events', color='lightcoral')
        
        axes[1,1].set_title('Daily Test Scale')
        axes[1,1].set_xlabel('Date')
        axes[1,1].set_ylabel('Document Count', color='skyblue')
        ax2.set_ylabel('Traffic Events', color='lightcoral')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # Add legends
        axes[1,1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        chart_path = self.charts_dir / 'time_series_analysis.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved: {chart_path}")
        plt.close()  # Close to free memory
        
    def create_approach_comparison_heatmap(self, df: pd.DataFrame):
        """Create a heatmap comparing approaches across different metrics."""
        if df.empty:
            print("❌ No data available for heatmap")
            return
            
        # Calculate normalized metrics for each approach
        metrics = df.groupby('approach').agg({
            'actual_gas_used': 'mean',
            'verification_time': 'mean',
            'proof_size_bytes': 'mean',
            'properties_count': 'mean'
        })
        
        # Normalize metrics (lower is better for gas, time, and proof size)
        normalized_metrics = metrics.copy()
        for col in ['actual_gas_used', 'verification_time', 'proof_size_bytes']:
            if col in normalized_metrics.columns:
                # Invert so that lower values (better performance) get higher scores
                normalized_metrics[col] = 1 / (normalized_metrics[col] + 1)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(normalized_metrics.T, annot=True, cmap='RdYlGn', 
                   cbar_kws={'label': 'Performance Score (Higher is Better)'})
        plt.title('Approach Performance Comparison Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Approach')
        plt.ylabel('Metrics')
        plt.tight_layout()
        chart_path = self.charts_dir / 'approach_comparison_heatmap.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved: {chart_path}")
        plt.close()  # Close to free memory
        
    def generate_metrics_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive metrics report."""
        if df.empty:
            return "❌ No data available for metrics report"
            
        report = []
        report.append("=" * 80)
        report.append("CROSS-PROVINCE TEST RESULTS - COMPREHENSIVE METRICS REPORT")
        report.append("=" * 80)
        report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total test records: {len(df)}")
        report.append(f"Date range: {df['test_date'].min()} to {df['test_date'].max()}")
        report.append("")
        
        # Overall Statistics
        report.append("OVERALL STATISTICS")
        report.append("-" * 40)
        report.append(f"Total approaches tested: {df['approach'].nunique()}")
        report.append(f"Approaches: {', '.join(df['approach'].unique())}")
        report.append(f"Total test events: {df['event_id'].nunique()}")
        report.append(f"Average properties per event: {df['properties_count'].mean():.2f}")
        report.append(f"Average provinces per event: {df['provinces_count'].mean():.2f}")
        report.append("")
        
        # Performance Metrics by Approach
        report.append("PERFORMANCE METRICS BY APPROACH")
        report.append("-" * 40)
        
        approach_metrics = df.groupby('approach').agg({
            'actual_gas_used': ['mean', 'std', 'min', 'max'],
            'verification_time': ['mean', 'std', 'min', 'max'],
            'proof_size_bytes': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        for approach in df['approach'].unique():
            approach_data = df[df['approach'] == approach]
            report.append(f"\n{approach.upper()}:")
            report.append(f"  Gas Usage:")
            report.append(f"    Average: {approach_data['actual_gas_used'].mean():.0f} gas units")
            report.append(f"    Std Dev: {approach_data['actual_gas_used'].std():.0f}")
            report.append(f"    Range: {approach_data['actual_gas_used'].min():.0f} - {approach_data['actual_gas_used'].max():.0f}")
            
            report.append(f"  Verification Time:")
            report.append(f"    Average: {approach_data['verification_time'].mean():.4f} seconds")
            report.append(f"    Std Dev: {approach_data['verification_time'].std():.4f}")
            
            report.append(f"  Proof Size:")
            report.append(f"    Average: {approach_data['proof_size_bytes'].mean():.0f} bytes")
            report.append(f"    Std Dev: {approach_data['proof_size_bytes'].std():.0f}")
            
        # Best Performance Analysis
        report.append("\nBEST PERFORMANCE ANALYSIS")
        report.append("-" * 40)
        
        best_gas = df.loc[df.groupby('event_id')['actual_gas_used'].idxmin()]
        best_time = df.loc[df.groupby('event_id')['verification_time'].idxmin()]
        best_size = df.loc[df.groupby('event_id')['proof_size_bytes'].idxmin()]
        
        report.append("Gas Efficiency Winners:")
        gas_winners = best_gas['approach'].value_counts()
        for approach, count in gas_winners.items():
            percentage = (count / len(best_gas)) * 100
            report.append(f"  {approach}: {count} events ({percentage:.1f}%)")
            
        report.append("\nVerification Speed Winners:")
        time_winners = best_time['approach'].value_counts()
        for approach, count in time_winners.items():
            percentage = (count / len(best_time)) * 100
            report.append(f"  {approach}: {count} events ({percentage:.1f}%)")
            
        report.append("\nProof Size Winners:")
        size_winners = best_size['approach'].value_counts()
        for approach, count in size_winners.items():
            percentage = (count / len(best_size)) * 100
            report.append(f"  {approach}: {count} events ({percentage:.1f}%)")
        
        # Save report
        report_text = "\n".join(report)
        report_file = self.reports_dir / f"metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
            
        print(f"📊 Metrics report saved to: {report_file}")
        return report_text
    
    def save_processed_data(self, df: pd.DataFrame):
        """Save the processed DataFrame for further analysis."""
        if df.empty:
            return
            
        # Save as CSV for easy analysis
        csv_file = self.data_dir / "processed_verification_data.csv"
        df.to_csv(csv_file, index=False)
        print(f"💾 Processed data saved to: {csv_file}")
        
        # Save summary statistics
        summary_stats = df.groupby('approach').agg({
            'actual_gas_used': ['count', 'mean', 'std', 'min', 'max'],
            'verification_time': ['mean', 'std', 'min', 'max'],
            'proof_size_bytes': ['mean', 'std', 'min', 'max'],
            'properties_count': ['mean', 'std'],
            'provinces_count': ['mean', 'std']
        }).round(4)
        
        summary_file = self.data_dir / "summary_statistics.csv"
        summary_stats.to_csv(summary_file)
        print(f"💾 Summary statistics saved to: {summary_file}")
        
    def create_analysis_index(self, df: pd.DataFrame):
        """Create an HTML index file for easy navigation of results."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cross-Province Analysis Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .chart-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
        .chart-item {{ border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
        .metrics {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Cross-Province Test Results Analysis</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Total Records:</strong> {len(df):,}</p>
        <p><strong>Approaches:</strong> {', '.join(df['approach'].unique())}</p>
        <p><strong>Date Range:</strong> {df['test_date'].min()} to {df['test_date'].max()}</p>
    </div>
    
    <div class="section">
        <h2>📊 Charts</h2>
        <div class="chart-grid">
            <div class="chart-item">
                <h3>Gas Analysis</h3>
                <img src="charts/gas_analysis.png" alt="Gas Analysis">
                <p>Comparison of gas usage across approaches</p>
            </div>
            <div class="chart-item">
                <h3>Performance Comparison</h3>
                <img src="charts/performance_comparison.png" alt="Performance Comparison">
                <p>Verification time and proof size analysis</p>
            </div>
            <div class="chart-item">
                <h3>Time Series Analysis</h3>
                <img src="charts/time_series_analysis.png" alt="Time Series Analysis">
                <p>Daily trends and patterns</p>
            </div>
            <div class="chart-item">
                <h3>Approach Comparison Heatmap</h3>
                <img src="charts/approach_comparison_heatmap.png" alt="Approach Comparison">
                <p>Performance heatmap across metrics</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>📋 Reports & Data</h2>
        <ul>
            <li><a href="reports/">📄 Detailed Metrics Reports</a></li>
            <li><a href="data/processed_verification_data.csv">📊 Raw Data (CSV)</a></li>
            <li><a href="data/summary_statistics.csv">📈 Summary Statistics (CSV)</a></li>
        </ul>
    </div>
    
    <div class="section metrics">
        <h2>🔍 Quick Metrics</h2>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
        """
        
        for approach in df['approach'].unique():
            approach_data = df[df['approach'] == approach]
            avg_gas = approach_data['actual_gas_used'].mean()
            avg_time = approach_data['verification_time'].mean()
            avg_size = approach_data['proof_size_bytes'].mean()
            
            html_content += f"""
            <div style="text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                <h4>{approach.replace('_', ' ').title()}</h4>
                <p><strong>Avg Gas:</strong> {avg_gas:,.0f}</p>
                <p><strong>Avg Time:</strong> {avg_time:.3f}s</p>
                <p><strong>Avg Size:</strong> {avg_size:,.0f}b</p>
            </div>
            """
        
        html_content += """
        </div>
    </div>
</body>
</html>
"""
        
        index_file = self.output_directory / "index.html"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"🌐 Analysis index created: {index_file}")
        
    def run_complete_analysis(self):
        """Run complete visualization and analysis pipeline."""
        print("🚀 Starting Cross-Province Results Visualization")
        print("=" * 60)
        
        # Load results (single file or all)
        results = self.load_results()
        if not results:
            print("❌ No results found to visualize")
            return
            
        # Extract data
        df = self.extract_verification_data(results)
        if df.empty:
            print("❌ No verification data found")
            return
            
        print(f"📊 Extracted {len(df)} verification records")
        
        # Create visualizations
        print("\n📈 Creating gas analysis charts...")
        self.create_gas_analysis_charts(df)
        
        print("\n⚡ Creating performance comparison charts...")
        self.create_performance_comparison_charts(df)
        
        print("\n📅 Creating time series analysis...")
        self.create_time_series_analysis(df)
        
        print("\n🔥 Creating approach comparison heatmap...")
        self.create_approach_comparison_heatmap(df)
        
        print("\n📋 Generating metrics report...")
        report = self.generate_metrics_report(df)
        
        print("\n💾 Saving processed data...")
        self.save_processed_data(df)
        
        print("\n🌐 Creating analysis index...")
        self.create_analysis_index(df)
        
        print("\n" + "=" * 60)
        print("✅ VISUALIZATION COMPLETE!")
        print(f"📁 All outputs saved to: {self.output_directory}")
        print("\nGenerated files:")
        
        # List all generated files organized by folder
        print("📊 Charts:")
        for file in self.charts_dir.glob("*.png"):
            print(f"  - {file.name}")
            
        print("📋 Reports:")
        for file in self.reports_dir.glob("*"):
            print(f"  - {file.name}")
            
        print("💾 Data:")
        for file in self.data_dir.glob("*"):
            print(f"  - {file.name}")
            
        print("🌐 Index:")
        print(f"  - index.html")
        
        print(f"\n🔗 Open {self.output_directory / 'index.html'} in your browser to view all results!")
            
        return df, report

def main():
    """Main function to run the visualizer with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cross-Province Test Results Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python result_visualizer.py                                    # Analyze all reports
  python result_visualizer.py --list                            # List available reports  
  python result_visualizer.py --file report.json               # Analyze single report
  python result_visualizer.py --file "path/to/specific.json"    # Analyze specific file
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Path to a specific cross-province test result JSON file'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available cross-province test reports and exit'
    )
    
    parser.add_argument(
        '--results-dir', '-d',
        type=str,
        default='report',
        help='Directory to search for result files (default: report)'
    )
    
    args = parser.parse_args()
    
    # Create visualizer instance
    visualizer = CrossProvinceResultsVisualizer(
        results_directory=args.results_dir,
        single_file=args.file
    )
    
    # Handle list command
    if args.list:
        reports = visualizer.list_available_reports()
        if reports:
            print("\n💡 To analyze a specific report, use:")
            print('python result_visualizer.py --file "path/to/report.json"')
        return
    
    # Validate single file if specified
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ Error: File not found: {file_path}")
            print("\n💡 Use --list to see available reports")
            return
        
        print(f"🎯 Analyzing single file: {file_path}")
    else:
        print("🔍 Analyzing all cross-province test reports...")
    
    # Run analysis
    df, report = visualizer.run_complete_analysis()
    
    if df is not None and not df.empty:
        print("\n" + "=" * 60)
        print("QUICK METRICS SUMMARY")
        print("=" * 60)
        print(report[:1000] + "..." if len(report) > 1000 else report)
    else:
        print("❌ No data available for analysis")

if __name__ == "__main__":
    main()

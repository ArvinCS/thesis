#!/usr/bin/env python3
"""
Cross-Province Massive Scale Test

This script runs massive scale tests specifically focused on 4+ cross-province scenarios
to determine if hierarchical approach wins in complex cross-province verifications.

Key Features:
1. Heavily biased towards 4+ cross-province verifications (80% of events)
2. Massive scale testing (100k+ documents)
3. Detailed analysis of hierarchical vs traditional performance
4. Focus on gas cost optimization in complex scenarios
"""

import argparse
import json
import time
import os
import sys
import random
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict

# Import existing modules
from multi_day_verification_suite import MultiDayVerificationSuite
from gas_cost_analyzer import GasCostAnalyzer
from large_scale_generator import LargeScaleDocumentGenerator
from report_organizer import save_organized_file
from web3 import Web3

class CrossProvinceTrafficGenerator:
    """
    Specialized traffic generator focused on 4+ cross-province scenarios.
    """
    
    def __init__(self, documents, properties_by_province, seed=42):
        self.documents = documents
        self.properties_by_province = properties_by_province
        self.seed = seed
        random.seed(seed)
        
        # HEAVILY biased towards 4+ cross-province scenarios
        self.traffic_patterns = {
            'intra_province': 0.10,      # Only 10% single province
            'cross_province_2': 0.05,    # 5% across 2 provinces  
            'cross_province_3': 0.05,    # 5% across 3 provinces
            'cross_province_4plus': 0.80  # 80% across 4+ provinces (FOCUS!)
        }
        
        print(f"🎯 Cross-Province Traffic Generator initialized")
        print(f"   Target distribution: 80% are 4+ cross-province scenarios")
    
    def generate_cross_province_traffic_logs(self, num_events=5000):
        """Generate traffic logs heavily biased towards 4+ cross-province scenarios."""
        print(f"Generating {num_events} traffic events (80% cross-province 4+)...")
        
        traffic_logs = []
        province_names = list(self.properties_by_province.keys())
        
        # Ensure we have enough provinces for 4+ cross-province scenarios
        if len(province_names) < 4:
            raise Exception(f"Need at least 4 provinces for cross-province testing, found {len(province_names)}")
        
        cross_province_4plus_count = 0
        
        for event_idx in range(num_events):
            pattern = self._choose_verification_pattern()
            
            if pattern == 'intra_province':
                # Single province verification (rare)
                event = self._generate_single_province_event(province_names)
            elif pattern == 'cross_province_2':
                # 2 provinces (rare)
                event = self._generate_2_province_event(province_names)
            elif pattern == 'cross_province_3':
                # 3 provinces (rare)
                event = self._generate_3_province_event(province_names)
            else:  # cross_province_4plus (MAIN FOCUS)
                # 4+ provinces (80% of events)
                event = self._generate_4plus_province_event(province_names)
                cross_province_4plus_count += 1
            
            if event:
                traffic_logs.append(event)
        
        actual_4plus_percentage = (cross_province_4plus_count / len(traffic_logs)) * 100
        print(f"✅ Generated {len(traffic_logs)} events")
        print(f"   4+ cross-province events: {cross_province_4plus_count} ({actual_4plus_percentage:.1f}%)")
        
        return traffic_logs
    
    def _generate_single_province_event(self, province_names):
        """Generate single province event."""
        province = random.choice(province_names)
        properties = self.properties_by_province[province]
        
        if properties:
            num_props = min(random.randint(1, 3), len(properties))
            selected_props = random.sample(properties, num_props)
            return [prop['full_id'] for prop in selected_props]
        return []
    
    def _generate_2_province_event(self, province_names):
        """Generate 2-province event."""
        selected_provinces = random.sample(province_names, 2)
        event = []
        
        for province in selected_provinces:
            properties = self.properties_by_province[province]
            if properties:
                num_props = random.randint(1, 2)
                selected = random.sample(properties, min(num_props, len(properties)))
                event.extend([prop['full_id'] for prop in selected])
        
        return event
    
    def _generate_3_province_event(self, province_names):
        """Generate 3-province event."""
        selected_provinces = random.sample(province_names, min(3, len(province_names)))
        event = []
        
        for province in selected_provinces:
            properties = self.properties_by_province[province]
            if properties:
                num_props = random.randint(1, 2)
                selected = random.sample(properties, min(num_props, len(properties)))
                event.extend([prop['full_id'] for prop in selected])
        
        return event
    
    def _generate_4plus_province_event(self, province_names):
        """Generate 4+ province event (MAIN FOCUS)."""
        # Choose 4-8 provinces for maximum complexity
        max_provinces = min(8, len(province_names))
        num_provinces = random.randint(4, max_provinces)
        selected_provinces = random.sample(province_names, num_provinces)
        event = []
        
        for province in selected_provinces:
            properties = self.properties_by_province[province]
            if properties:
                # 1-3 properties per province for complex but manageable verification
                num_props = random.randint(1, 3)
                selected = random.sample(properties, min(num_props, len(properties)))
                event.extend([prop['full_id'] for prop in selected])
        
        return event
    
    def _choose_verification_pattern(self):
        """Choose verification pattern based on cross-province focused probabilities."""
        rand = random.random()
        cumulative = 0
        
        for pattern, probability in self.traffic_patterns.items():
            cumulative += probability
            if rand <= cumulative:
                return pattern
        
        return 'cross_province_4plus'  # fallback to our focus area
    
    def analyze_cross_province_patterns(self, traffic_logs):
        """Analyze cross-province patterns in detail."""
        stats = {
            'total_events': len(traffic_logs),
            'cross_province_4plus_events': 0,
            'cross_province_4plus_percentage': 0,
            'province_distribution': defaultdict(int),
            'property_count_distribution': defaultdict(int),
            'complex_events': []  # Events with 4+ provinces for detailed analysis
        }
        
        for event in traffic_logs:
            # Count properties
            stats['property_count_distribution'][len(event)] += 1
            
            # Analyze provinces
            provinces_in_event = set()
            for prop_id in event:
                if '.' in prop_id:
                    province = prop_id.split('.')[0]
                    provinces_in_event.add(province)
            
            num_provinces = len(provinces_in_event)
            stats['province_distribution'][num_provinces] += 1
            
            # Track 4+ province events
            if num_provinces >= 4:
                stats['cross_province_4plus_events'] += 1
                stats['complex_events'].append({
                    'provinces': list(provinces_in_event),
                    'province_count': num_provinces,
                    'property_count': len(event),
                    'event': event
                })
        
        if stats['total_events'] > 0:
            stats['cross_province_4plus_percentage'] = (stats['cross_province_4plus_events'] / stats['total_events']) * 100
        
        return stats

class CrossProvinceMassiveTestRunner:
    """
    Massive scale test runner focused on 4+ cross-province scenarios.
    """
    
    def __init__(self, web3_instance=None, approaches=None):
        self.web3 = web3_instance
        self.results = {}
        # Set default approaches or use provided ones
        if approaches is None or 'all' in approaches:
            self.approaches = ['hierarchical', 'traditional_multiproof', 'traditional_single_proof', 'traditional_huffman', 'clustered_flat', 'clustered_flat_with_merkle']
        else:
            self.approaches = approaches
    
    def _filter_results_by_approaches(self, results):
        """Filter test results to only include selected approaches."""
        if 'verification_results' in results:
            # Filter verification results to only include selected approaches
            filtered_verification = []
            for result in results['verification_results']:
                if result is not None and result.get('approach') in self.approaches:
                    filtered_verification.append(result)
            results['verification_results'] = filtered_verification
        
        if 'tree_systems' in results:
            # Filter tree systems to only include selected approaches
            filtered_tree_systems = {}
            for approach, system in results['tree_systems'].items():
                if approach in self.approaches:
                    # Make system JSON-serializable by removing non-serializable objects
                    serializable_system = {}
                    for key, value in system.items():
                        if key not in ['manager', 'builder', 'adapter', 'system']:  # Skip non-serializable objects
                            serializable_system[key] = value
                    filtered_tree_systems[approach] = serializable_system
            results['tree_systems'] = filtered_tree_systems
        
        return results
        
    def run_cross_province_massive_test(self, document_count=50000, traffic_events=5000):
        """
        Run massive scale test focused on 4+ cross-province scenarios.
        
        Args:
            document_count: Number of documents to generate (default 50k)
            traffic_events: Number of traffic events (80% will be 4+ cross-province)
            
        Returns:
            Comprehensive test results focused on cross-province analysis
        """
        print(f"{'='*80}")
        print(f"CROSS-PROVINCE MASSIVE SCALE TEST")
        print(f"{'='*80}")
        print(f"Documents: {document_count:,}")
        print(f"Traffic Events: {traffic_events}")
        print(f"Focus: 80% of events are 4+ cross-province scenarios")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        start_time = time.time()
        
        # Step 1: Generate massive scale documents
        print(f"\n--- STEP 1: GENERATING MASSIVE SCALE DOCUMENTS ---")
        doc_generator = LargeScaleDocumentGenerator(target_document_count=document_count)
        documents = doc_generator.generate_documents()
        
        generation_stats = doc_generator.get_generation_stats()
        print(f"✅ Generated {generation_stats['total_documents']:,} documents across {generation_stats['provinces']} provinces")
        
        # Step 2: Generate cross-province focused traffic
        print(f"\n--- STEP 2: GENERATING CROSS-PROVINCE TRAFFIC ---")
        traffic_generator = CrossProvinceTrafficGenerator(documents, doc_generator.properties_by_province)
        traffic_logs = traffic_generator.generate_cross_province_traffic_logs(traffic_events)
        
        # Analyze traffic patterns
        traffic_stats = traffic_generator.analyze_cross_province_patterns(traffic_logs)
        print(f"✅ 4+ cross-province events: {traffic_stats['cross_province_4plus_events']} ({traffic_stats['cross_province_4plus_percentage']:.1f}%)")
        
        # Step 3: Run multi-day verification test
        print(f"\n--- STEP 3: RUNNING CROSS-PROVINCE VERIFICATION TEST ---")
        
        multi_day_suite = MultiDayVerificationSuite(self.web3)
        
        # Use the cross-province focused traffic for testing with all approaches
        # First build all tree systems manually to ensure we get all 6 approaches (including clustered_flat)
        print("Building all tree systems (5 approaches)...")
        tree_systems = multi_day_suite._build_all_tree_systems(documents, {'traffic_logs': traffic_logs})
        
        # Now run the cross-province test with pre-built systems
        multi_day_results = multi_day_suite.run_cross_province_focused_test(
            documents=documents,
            properties_by_province=doc_generator.properties_by_province,
            traffic_logs=traffic_logs,
            force_onchain_verification=(self.web3 is not None)
        )
        
        # Override the tree_systems in results to ensure all 5 approaches are included
        multi_day_results['tree_systems'] = tree_systems
        
        # Step 4: Run gas cost analysis on 4+ province scenarios
        print(f"\n--- STEP 4: ANALYZING GAS COSTS FOR 4+ PROVINCE SCENARIOS ---")
        gas_analyzer = GasCostAnalyzer(self.web3)
        
        # Filter to only 4+ province events for gas analysis
        complex_events = [event['event'] for event in traffic_stats['complex_events'][:100]]  # Limit for gas analysis
        
        gas_results = gas_analyzer.run_cross_province_gas_analysis(
            documents=documents,
            properties_by_province=doc_generator.properties_by_province,
            complex_events=complex_events
        )
        
        # Step 5: Compile comprehensive results
        total_time = time.time() - start_time
        
        # Filter results to only include selected approaches
        filtered_multi_day_results = self._filter_results_by_approaches(multi_day_results)
        filtered_gas_results = self._filter_results_by_approaches(gas_results)
        
        comprehensive_results = {
            'test_metadata': {
                'test_type': 'cross_province_massive',
                'execution_time': datetime.now().isoformat(),
                'total_duration_seconds': total_time,
                'document_count': document_count,
                'traffic_events': traffic_events,
                'selected_approaches': self.approaches
            },
            'document_generation': generation_stats,
            'traffic_analysis': traffic_stats,
            'multi_day_results': filtered_multi_day_results,
            'gas_analysis': filtered_gas_results,
            'cross_province_insights': self._analyze_cross_province_performance(filtered_multi_day_results, filtered_gas_results)
        }
        
        # Save results
        self._save_results(comprehensive_results)
        
        print(f"\n{'='*80}")
        print(f"CROSS-PROVINCE MASSIVE TEST COMPLETED")
        print(f"{'='*80}")
        print(f"Total Duration: {total_time:.1f} seconds")
        print(f"4+ Cross-Province Events: {traffic_stats['cross_province_4plus_events']}")
        print(f"Results saved to: cross_province_massive_test_results.json")
        
        return comprehensive_results
    
    def _analyze_cross_province_performance(self, multi_day_results, gas_results):
        """Analyze performance across all approaches in cross-province scenarios."""
        # Extract analysis from multi_day_results if available
        cross_province_analysis = multi_day_results.get('cross_province_analysis', {})
        
        # Get approach performance data
        approach_performance = cross_province_analysis.get('approach_performance', {})
        gas_rankings = cross_province_analysis.get('gas_cost_rankings', {})
        best_approach = cross_province_analysis.get('overall_best_approach', 'unknown')
        
        # Legacy values for backward compatibility
        hierarchical_wins = cross_province_analysis.get('hierarchical_wins_4plus', 0)
        traditional_wins = cross_province_analysis.get('traditional_wins_4plus', 0)
        hierarchical_win_rate = cross_province_analysis.get('hierarchical_win_rate_4plus', 0)
        hierarchical_savings = cross_province_analysis.get('avg_gas_savings_4plus_when_hierarchical_wins', 0)
        
        # Create comprehensive summary
        summary_parts = []
        
        if gas_rankings and 'by_average_gas' in gas_rankings:
            rankings = gas_rankings['by_average_gas']
            if rankings:
                best_efficiency = rankings[0][0]
                best_gas = rankings[0][1]
                worst_efficiency = rankings[-1][0]
                worst_gas = rankings[-1][1]
                
                summary_parts.append(f"Most efficient approach: {best_efficiency} (avg: {best_gas:.0f} gas)")
                summary_parts.append(f"Least efficient approach: {worst_efficiency} (avg: {worst_gas:.0f} gas)")
                
                if len(rankings) > 1:
                    savings_pct = ((worst_gas - best_gas) / worst_gas) * 100
                    summary_parts.append(f"Gas savings potential: {savings_pct:.1f}%")
        
        if approach_performance:
            total_events = sum(perf.get('wins', 0) for perf in approach_performance.values())
            if total_events > 0:
                summary_parts.append(f"Total events analyzed: {total_events}")
                
                # Sort approaches by win rate
                sorted_by_wins = sorted(approach_performance.items(), 
                                      key=lambda x: x[1].get('win_rate', 0), reverse=True)
                
                if sorted_by_wins:
                    best_performer = sorted_by_wins[0]
                    summary_parts.append(f"Best overall performer: {best_performer[0]} ({best_performer[1].get('win_rate', 0):.1f}% win rate)")
        
        insights = {
            'hierarchical_wins_4plus_provinces': hierarchical_wins,
            'traditional_wins_4plus_provinces': traditional_wins,
            'hierarchical_win_percentage': hierarchical_win_rate,
            'avg_gas_savings_when_hierarchical_wins': hierarchical_savings,
            'best_hierarchical_scenarios': cross_province_analysis.get('best_hierarchical_cases_4plus', []),
            'summary': ". ".join(summary_parts) if summary_parts else "No sufficient data for analysis",
            'approach_performance': approach_performance,
            'gas_cost_rankings': gas_rankings,
            'overall_best_approach': best_approach,
            'best_approach_per_event': cross_province_analysis.get('best_approach_per_event', [])
        }
        
        return insights
    
    def _save_results(self, results):
        """Save comprehensive results to organized file structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cross_province_massive_test_{timestamp}.json"
        
        saved_file = save_organized_file(results, filename, "cross_province_tests")
        print(f"✅ Results saved to: {saved_file}")
        return saved_file

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Cross-Province Massive Scale Test")
    parser.add_argument('--documents', type=int, default=50000, help='Number of documents to generate')
    parser.add_argument('--events', type=int, default=2000, help='Number of traffic events')
    parser.add_argument('--approaches', nargs='+', 
                        choices=['hierarchical', 'traditional_multiproof', 'traditional_single_proof', 'traditional_huffman', 'clustered_flat', 'clustered_flat_with_merkle', 'all'],
                        default=['all'],
                        help='Which approaches to test (default: all). Options: hierarchical, traditional_multiproof, traditional_single_proof, traditional_huffman, clustered_flat, clustered_flat_with_merkle, all')
    
    args = parser.parse_args()
    
    # Setup Web3 connection
    web3_instance = None
    try:
        web3_instance = Web3(Web3.HTTPProvider('http://localhost:8545'))
        if web3_instance.is_connected():
            print("✅ Web3 connected to local Hardhat network")
        else:
            print("⚠️  Web3 connection failed, running estimation-only")
            web3_instance = None
    except Exception as e:
        print(f"⚠️  Web3 setup failed: {e}")
        web3_instance = None
    
    # Run cross-province massive test
    test_runner = CrossProvinceMassiveTestRunner(web3_instance, approaches=args.approaches)
    results = test_runner.run_cross_province_massive_test(
        document_count=args.documents,
        traffic_events=args.events
    )
    
    print(f"\n🎯 CROSS-PROVINCE TEST SUMMARY:")
    print(f"   Generated: {args.documents:,} documents")
    print(f"   Traffic Events: {args.events}")
    print(f"   4+ Province Focus: 80% of verification events")
    print(f"   Selected Approaches: {', '.join(args.approaches)}")
    print(f"   Results available for analysis")

if __name__ == "__main__":
    main()

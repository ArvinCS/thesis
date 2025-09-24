#!/usr/bin/env python3
"""
Intra-Province Massive Scale Test

This script runs massive scale tests specifically focused on intra-province scenarios
to determine if traditional approaches win in simple intra-province verifications.

Key Features:
1. Heavily biased towards intra-province verifications (80% of events)
2. Massive scale testing (100k+ documents)
3. Detailed analysis of traditional vs hierarchical performance
4. Focus on gas cost optimization in simple scenarios where traditional might excel
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

class IntraProvinceTrafficGenerator:
    """
    Specialized traffic generator focused on intra-province scenarios.
    """
    
    def __init__(self, documents, properties_by_province, seed=42):
        self.documents = documents
        self.properties_by_province = properties_by_province
        self.seed = seed
        random.seed(seed)
        
        # HEAVILY biased towards intra-province scenarios
        self.traffic_patterns = {
            'intra_province': 0.80,      # 80% single province (FOCUS!)
            'cross_province_2': 0.15,    # 15% across 2 provinces  
            'cross_province_3': 0.05,    # 5% across 3 provinces
            'cross_province_4plus': 0.0  # 0% across 4+ provinces
        }
        
        print(f"🎯 Intra-Province Traffic Generator initialized")
        print(f"   Target distribution: 80% are intra-province scenarios")
    
    def generate_intra_province_traffic_logs(self, num_events=5000):
        """Generate traffic logs heavily biased towards intra-province scenarios."""
        print(f"Generating {num_events} traffic events (80% intra-province)...")
        
        traffic_logs = []
        province_names = list(self.properties_by_province.keys())
        
        # Ensure we have at least some provinces
        if len(province_names) < 1:
            raise Exception(f"Need at least 1 province for intra-province testing, found {len(province_names)}")
        
        intra_province_count = 0
        
        for event_idx in range(num_events):
            pattern = self._choose_verification_pattern()
            
            if pattern == 'intra_province':
                # Single province verification (MAIN FOCUS - 80%)
                event = self._generate_single_province_event(province_names)
                intra_province_count += 1
            elif pattern == 'cross_province_2':
                # 2 provinces (15%)
                event = self._generate_2_province_event(province_names)
            elif pattern == 'cross_province_3':
                # 3 provinces (5%)
                event = self._generate_3_province_event(province_names)
            else:  # cross_province_4plus (0%)
                # This shouldn't happen with our traffic patterns, but fallback to intra-province
                event = self._generate_single_province_event(province_names)
                intra_province_count += 1
            
            if event:
                traffic_logs.append(event)
        
        actual_intra_percentage = (intra_province_count / len(traffic_logs)) * 100
        print(f"✅ Generated {len(traffic_logs)} events")
        print(f"   Intra-province events: {intra_province_count} ({actual_intra_percentage:.1f}%)")
        
        return traffic_logs
    
    def _generate_single_province_event(self, province_names):
        """Generate single province event with varying complexity."""
        province = random.choice(province_names)
        properties = self.properties_by_province[province]
        
        if properties:
            # For intra-province focus, vary the number of properties more
            # Small events (1-5 props): 60%
            # Medium events (6-15 props): 30% 
            # Large events (16+ props): 10%
            rand = random.random()
            if rand < 0.6:
                num_props = min(random.randint(1, 5), len(properties))
            elif rand < 0.9:
                num_props = min(random.randint(6, 15), len(properties))
            else:
                num_props = min(random.randint(16, min(50, len(properties))), len(properties))
            
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
        """Generate 4+ province event (not used in intra-province focus, fallback to intra)."""
        # For intra-province testing, we don't want 4+ province events
        # Fallback to single province event
        return self._generate_single_province_event(province_names)
    
    def _choose_verification_pattern(self):
        """Choose verification pattern based on cross-province focused probabilities."""
        rand = random.random()
        cumulative = 0
        
        for pattern, probability in self.traffic_patterns.items():
            cumulative += probability
            if rand <= cumulative:
                return pattern
        
        return 'intra_province'  # fallback to our focus area
    
    def analyze_intra_province_patterns(self, traffic_logs):
        """Analyze intra-province patterns in detail."""
        stats = {
            'total_events': len(traffic_logs),
            'intra_province_events': 0,
            'intra_province_percentage': 0,
            'province_distribution': defaultdict(int),
            'property_count_distribution': defaultdict(int),
            'large_intra_events': []  # Large intra-province events for detailed analysis
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
            
            # Track intra-province events (single province)
            if num_provinces == 1:
                stats['intra_province_events'] += 1
                # Track large intra-province events (10+ properties)
                if len(event) >= 10:
                    stats['large_intra_events'].append({
                        'province': list(provinces_in_event)[0],
                        'property_count': len(event),
                        'event': event
                    })
        
        if stats['total_events'] > 0:
            stats['intra_province_percentage'] = (stats['intra_province_events'] / stats['total_events']) * 100
        
        return stats

class IntraProvinceMassiveTestRunner:
    """
    Massive scale test runner focused on intra-province scenarios.
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
        
    def run_intra_province_massive_test(self, document_count=50000, traffic_events=5000):
        """
        Run massive scale test focused on intra-province scenarios.
        
        Args:
            document_count: Number of documents to generate (default 50k)
            traffic_events: Number of traffic events (80% will be intra-province)
            
        Returns:
            Comprehensive test results focused on intra-province analysis
        """
        print(f"{'='*80}")
        print(f"INTRA-PROVINCE MASSIVE SCALE TEST")
        print(f"{'='*80}")
        print(f"Documents: {document_count:,}")
        print(f"Traffic Events: {traffic_events}")
        print(f"Focus: 80% of events are intra-province scenarios")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        start_time = time.time()
        
        # Step 1: Generate massive scale documents
        print(f"\n--- STEP 1: GENERATING MASSIVE SCALE DOCUMENTS ---")
        doc_generator = LargeScaleDocumentGenerator(target_document_count=document_count)
        documents = doc_generator.generate_documents()
        
        generation_stats = doc_generator.get_generation_stats()
        print(f"✅ Generated {generation_stats['total_documents']:,} documents across {generation_stats['provinces']} provinces")
        
        # Step 2: Generate intra-province focused traffic
        print(f"\n--- STEP 2: GENERATING INTRA-PROVINCE TRAFFIC ---")
        traffic_generator = IntraProvinceTrafficGenerator(documents, doc_generator.properties_by_province)
        traffic_logs = traffic_generator.generate_intra_province_traffic_logs(traffic_events)
        
        # Analyze traffic patterns
        traffic_stats = traffic_generator.analyze_intra_province_patterns(traffic_logs)
        print(f"✅ Intra-province events: {traffic_stats['intra_province_events']} ({traffic_stats['intra_province_percentage']:.1f}%)")
        
        # Step 3: Run multi-day verification test
        print(f"\n--- STEP 3: RUNNING INTRA-PROVINCE VERIFICATION TEST ---")
        
        multi_day_suite = MultiDayVerificationSuite(self.web3)
        
        # Use the intra-province focused traffic for testing with all approaches
        # First build all tree systems manually to ensure we get all 5 approaches (including clustered_flat)
        print("Building all tree systems (5 approaches)...")
        tree_systems = multi_day_suite._build_all_tree_systems(documents, {'traffic_logs': traffic_logs})
        
        # Now run the cross-province test with pre-built systems (will work for intra-province too)
        multi_day_results = multi_day_suite.run_cross_province_focused_test(
            documents=documents,
            properties_by_province=doc_generator.properties_by_province,
            traffic_logs=traffic_logs,
            force_onchain_verification=(self.web3 is not None)
        )
        
        # Override the tree_systems in results to ensure all 5 approaches are included
        multi_day_results['tree_systems'] = tree_systems
        
        # Step 4: Run gas cost analysis on large intra-province scenarios
        print(f"\n--- STEP 4: ANALYZING GAS COSTS FOR LARGE INTRA-PROVINCE SCENARIOS ---")
        gas_analyzer = GasCostAnalyzer(self.web3)
        
        # Filter to only large intra-province events for gas analysis
        large_intra_events = [event['event'] for event in traffic_stats['large_intra_events'][:100]]  # Limit for gas analysis
        
        gas_results = gas_analyzer.run_cross_province_gas_analysis(
            documents=documents,
            properties_by_province=doc_generator.properties_by_province,
            complex_events=large_intra_events
        )
        
        # Step 5: Compile comprehensive results
        total_time = time.time() - start_time
        
        # Filter results to only include selected approaches
        filtered_multi_day_results = self._filter_results_by_approaches(multi_day_results)
        filtered_gas_results = self._filter_results_by_approaches(gas_results)
        
        comprehensive_results = {
            'test_metadata': {
                'test_type': 'intra_province_massive',
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
            'intra_province_insights': self._analyze_intra_province_performance(filtered_multi_day_results, filtered_gas_results)
        }
        
        # Save results
        self._save_results(comprehensive_results)
        
        print(f"\n{'='*80}")
        print(f"INTRA-PROVINCE MASSIVE TEST COMPLETED")
        print(f"{'='*80}")
        print(f"Total Duration: {total_time:.1f} seconds")
        print(f"Intra-Province Events: {traffic_stats['intra_province_events']}")
        print(f"Results saved to: intra_province_massive_test_results.json")
        
        return comprehensive_results
    
    def _analyze_intra_province_performance(self, multi_day_results, gas_results):
        """Analyze performance across all approaches in intra-province scenarios."""
        # Extract analysis from multi_day_results if available (will be cross_province_analysis since we use that method)
        analysis = multi_day_results.get('cross_province_analysis', {})
        
        # Get approach performance data
        approach_performance = analysis.get('approach_performance', {})
        gas_rankings = analysis.get('gas_cost_rankings', {})
        best_approach = analysis.get('overall_best_approach', 'unknown')
        
        # Legacy values for backward compatibility (adapt cross-province keys to intra-province context)
        hierarchical_wins = analysis.get('hierarchical_wins_4plus', 0)
        traditional_wins = analysis.get('traditional_wins_4plus', 0)
        hierarchical_win_rate = analysis.get('hierarchical_win_rate_4plus', 0)
        hierarchical_savings = analysis.get('avg_gas_savings_4plus_when_hierarchical_wins', 0)
        
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
            'hierarchical_wins_intra_province': hierarchical_wins,
            'traditional_wins_intra_province': traditional_wins,
            'hierarchical_win_percentage': hierarchical_win_rate,
            'avg_gas_savings_when_hierarchical_wins': hierarchical_savings,
            'best_hierarchical_scenarios': analysis.get('best_hierarchical_cases_4plus', []),
            'summary': ". ".join(summary_parts) if summary_parts else "No sufficient data for analysis",
            'approach_performance': approach_performance,
            'gas_cost_rankings': gas_rankings,
            'overall_best_approach': best_approach,
            'best_approach_per_event': analysis.get('best_approach_per_event', [])
        }
        
        return insights
    
    def _save_results(self, results):
        """Save comprehensive results to organized file structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intra_province_massive_test_{timestamp}.json"
        
        saved_file = save_organized_file(results, filename, "intra_province_tests")
        print(f"✅ Results saved to: {saved_file}")
        return saved_file

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Intra-Province Massive Scale Test")
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
    
    # Run intra-province massive test
    test_runner = IntraProvinceMassiveTestRunner(web3_instance, approaches=args.approaches)
    results = test_runner.run_intra_province_massive_test(
        document_count=args.documents,
        traffic_events=args.events
    )
    
    print(f"\n🎯 INTRA-PROVINCE TEST SUMMARY:")
    print(f"   Generated: {args.documents:,} documents")
    print(f"   Traffic Events: {args.events}")
    print(f"   Intra-Province Focus: 80% of verification events")
    print(f"   Selected Approaches: {', '.join(args.approaches)}")
    print(f"   Results available for analysis")

if __name__ == "__main__":
    main()

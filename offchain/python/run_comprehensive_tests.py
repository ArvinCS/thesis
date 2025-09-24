#!/usr/bin/env python3
"""
Comprehensive Test Execution Script

This script provides a unified interface for running comprehensive tests:
1. Multi-day verification tests with adaptive learning
2. Gas cost analysis with all available data
3. Performance visualization and comparative analysis

Available approaches:
- hierarchical: Hierarchical Merkle tree with adaptive optimization
- traditional_multiproof: Traditional multiproof Merkle tree
- traditional_single_proof: Traditional single proof Merkle tree
- traditional_huffman: Traditional Merkle tree with Huffman optimization
- clustered_flat: Clustered flat tree with ClusteredFlatVerifier contract
- clustered_flat_with_merkle: Clustered flat tree with MerkleVerifier contract (gas-optimized)

Usage:
    # Run comprehensive multi-day tests with all approaches
    python run_comprehensive_tests.py --test-type all --scale medium
    
    # Run multi-day tests with specific approaches
    python run_comprehensive_tests.py --test-type multi-day --scale large --days 7 --approaches hierarchical clustered_flat_with_merkle
    
    # Compare clustered flat approaches
    python run_comprehensive_tests.py --test-type multi-day --scale small --approaches clustered_flat clustered_flat_with_merkle
    
    # Large scale test with comprehensive analysis
    python run_comprehensive_tests.py --test-type all --scale massive  # 100k docs with adaptive optimization
"""

import argparse
import json
import time
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Import test modules
from multi_day_verification_suite import MultiDayVerificationSuite
from performance_visualizer import PerformanceVisualizer
from web3 import Web3

class ComprehensiveTestRunner:
    """
    Unified test runner for all comprehensive tests.
    
    This runner coordinates:
    1. Test execution across different modules
    2. Result aggregation and reporting
    3. Performance visualization
    4. Report generation
    """
    
    def __init__(self, web3_instance=None, selected_approaches=None):
        self.web3 = web3_instance
        self.results = {}
        self.start_time = None
        self.selected_approaches = selected_approaches
        
        # Setup reports directory structure
        self.reports_dir = self._setup_reports_directory()
        
        # Available approaches
        self.available_approaches = [
            'hierarchical',
            'traditional_multiproof', 
            'traditional_single_proof',
            'traditional_huffman',
            'clustered_flat',
            'clustered_flat_with_merkle'
        ]
        
        # Test configurations
        self.test_configs = {
            'small': {
                'documents': 1000,
                'traffic_events': 500,
                'days': 3,
                'base_events_per_day': 150
            },
            'medium': {
                'documents': 2000,
                'traffic_events': 1000,
                'days': 5,
                'base_events_per_day': 200
            },
            'large': {
                'documents': 5000,
                'traffic_events': 2500,
                'days': 7,
                'base_events_per_day': 350
            },
            'xlarge': {
                'documents': 10000,
                'traffic_events': 5000,
                'days': 10,
                'base_events_per_day': 500
            },
            'massive': {
                'documents': 100000,
                'traffic_events': 25000,  # 25% sampling for comprehensive testing
                'days': 15,
                'base_events_per_day': 600,
            },
            'massive_sparse': {
                'documents': 100000,
                'traffic_events': 50000,  # 50% sampling with sparse verification
                'days': 15,
                'base_events_per_day': 600,
                'sparse_verification': True,  # Enable sparse verification
                'verification_sampling_rate': 0.1  # Use 10% of documents for each verification
            },
            'ultra': {
                'documents': 250000,
                'traffic_events': 75000,  # 30% sampling for ultra scale
                'days': 20,
                'base_events_per_day': 800,
            }
        }
    
    def _setup_reports_directory(self):
        """Setup organized directory structure for reports."""
        base_reports_dir = "reports"
        
        # Create timestamp-based session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(base_reports_dir, f"session_{timestamp}")
        
        # Create subdirectories for different types of reports
        subdirs = [
            "multi_day",
 
            "gas_analysis",
            "visualizations",
            "comprehensive_reports"
        ]
        
        # Create all directories
        for subdir in [session_dir] + [os.path.join(session_dir, sd) for sd in subdirs]:
            os.makedirs(subdir, exist_ok=True)
        
        print(f"📁 Reports will be saved to: {session_dir}")
        return session_dir
    
    def run_all_tests(self, scale: str = 'medium', custom_config: Optional[Dict[str, Any]] = None):
        """
        Run all comprehensive tests.
        
        Args:
            scale: Test scale ('small', 'medium', 'large', 'xlarge')
            custom_config: Custom configuration override
            
        Returns:
            Comprehensive test results
        """
        print(f"{'='*80}")
        print(f"COMPREHENSIVE TEST SUITE EXECUTION")
        print(f"{'='*80}")
        print(f"Scale: {scale}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        if self.selected_approaches:
            print(f"Selected approaches: {', '.join(self.selected_approaches)}")
        else:
            print(f"Using all available approaches: {', '.join(self.available_approaches)}")
        
        self.start_time = time.time()
        
        # Get test configuration
        config = self.test_configs.get(scale, self.test_configs['medium'])
        if custom_config:
            config.update(custom_config)
        
        # Add selected approaches to config
        if self.selected_approaches:
            config['selected_approaches'] = self.selected_approaches
        
        print(f"\nTest Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Run all test suites
        print(f"\n{'='*80}")
        print(f"RUNNING ALL TEST SUITES")
        print(f"{'='*80}")
        
        # 1. Multi-day verification tests (includes all analysis and data aggregation)
        print(f"\n--- RUNNING MULTI-DAY VERIFICATION TESTS ---")
        multi_day_results = self._run_multi_day_tests(config)
        self.results['multi_day'] = multi_day_results
        
        # 2. Generate comprehensive report
        print(f"\n--- GENERATING COMPREHENSIVE REPORT ---")
        final_report = self._generate_comprehensive_report(config)
        
        # 5. Create visualizations
        print(f"\n--- CREATING PERFORMANCE VISUALIZATIONS ---")
        self._create_visualizations(self.selected_approaches)
        
        total_time = time.time() - self.start_time
        print(f"\n{'='*80}")
        print(f"ALL TESTS COMPLETED IN {total_time:.2f}s")
        print(f"{'='*80}")
        
        return final_report
    
    def run_multi_day_tests(self, scale: str = 'medium', days: Optional[int] = None):
        """Run only multi-day verification tests."""
        config = self.test_configs.get(scale, self.test_configs['medium'])
        if days:
            config['days'] = days
        if self.selected_approaches:
            config['selected_approaches'] = self.selected_approaches
        
        print(f"Running multi-day verification tests (scale: {scale})...")
        if self.selected_approaches:
            print(f"Selected approaches: {', '.join(self.selected_approaches)}")
        return self._run_multi_day_tests(config)
    

    

    
    def _run_multi_day_tests(self, config: Dict[str, Any]):
        """Run multi-day verification tests."""
        try:
            if not self.web3:
                raise Exception("Web3 connection required for on-chain multi-day verification")
            
            multi_day_suite = MultiDayVerificationSuite(self.web3, reports_dir=self.reports_dir)
            
            # Check for sparse verification settings
            sparse_verification = config.get('sparse_verification', False)
            verification_sampling_rate = config.get('verification_sampling_rate', 1.0)
            
            results = multi_day_suite.run_comprehensive_multi_day_test(
                document_count=config['documents'],
                num_days=config['days'],
                base_events_per_day=config['base_events_per_day'],
                force_onchain_verification=True,  # Ensure on-chain verification for gas analysis
                sparse_verification=sparse_verification,
                verification_sampling_rate=verification_sampling_rate,
                selected_approaches=self.selected_approaches,
                reports_dir=self.reports_dir
            )
            
            # Gas analysis is now integrated into multi-day results
            print(f"✅ Multi-day tests with integrated gas analysis completed successfully")
            return results
            
        except Exception as e:
            print(f"❌ Multi-day tests failed: {e}")
            return {'error': str(e)}
    

    
    
    def _generate_comprehensive_report(self, config: Dict[str, Any]):
        """Generate comprehensive test report."""
        total_time = time.time() - self.start_time
        
        report = {
            'test_metadata': {
                'execution_time': datetime.now().isoformat(),
                'total_duration_seconds': total_time,
                'test_configuration': config,
                'web3_connected': self.web3 is not None
            },
            'test_results': self.results,
            'summary': self._generate_test_summary(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save comprehensive report in organized directory
        report_filename = f'comprehensive_test_report_{config["documents"]}docs.json'
        report_path = os.path.join(self.reports_dir, "comprehensive_reports", report_filename)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Comprehensive report saved to: {report_path}")
        
        return report
    
    def _generate_test_summary(self):
        """Generate summary of all test results."""
        summary = {
            'total_tests_run': len(self.results),
            'successful_tests': 0,
            'failed_tests': 0,
            'key_findings': [],
            'performance_metrics': {}
        }
        
        for test_name, test_results in self.results.items():
            if 'error' in test_results:
                summary['failed_tests'] += 1
            else:
                summary['successful_tests'] += 1
                
                # Extract key findings
                if test_name == 'multi_day' and 'optimization_analysis' in test_results:
                    opt_analysis = test_results['optimization_analysis']
                    if 'optimization_ratios' in opt_analysis:
                        ratios = opt_analysis['optimization_ratios']
                        if 'proof_size_reduction' in ratios:
                            summary['key_findings'].append(
                                f"Multi-day proof size reduction: {ratios['proof_size_reduction']:.1f}%"
                            )
                        if 'gas_reduction' in ratios:
                            summary['key_findings'].append(
                                f"Multi-day gas reduction: {ratios['gas_reduction']:.1f}%"
                            )
                

        
        return summary
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Analyze results and generate recommendations
        for test_name, test_results in self.results.items():
            if 'error' in test_results:
                continue
            
            if test_name == 'multi_day':
                if 'optimization_analysis' in test_results:
                    opt_analysis = test_results['optimization_analysis']
                    if 'optimization_ratios' in opt_analysis:
                        ratios = opt_analysis['optimization_ratios']
                        if ratios.get('proof_size_reduction', 0) > 20:
                            recommendations.append(
                                "High proof size reduction achieved - hierarchical approach recommended for production"
                            )
                        elif ratios.get('gas_reduction', 0) > 15:
                            recommendations.append(
                                "Significant gas savings achieved - consider hierarchical approach for cost-sensitive applications"
                            )
            

        
        # Remove duplicates
        recommendations = list(set(recommendations))
        
        return recommendations
    
    def _create_visualizations(self, selected_approaches: Optional[list] = None):
        """Create performance visualizations."""
        try:
            visualizations_dir = os.path.join(self.reports_dir, "visualizations")
            visualizer = PerformanceVisualizer(output_dir=visualizations_dir)
            
            if selected_approaches is None or len(selected_approaches) == 0:
                selected_approaches = self.available_approaches

            # Create visualizations for each test type
            if 'multi_day' in self.results and 'error' not in self.results['multi_day']:
                print("📊 Creating multi-day visualizations...")
                try:
                    visualizer.visualize_multi_day_results(self.results['multi_day'], selected_approaches)
                    print("✅ Multi-day visualizations completed successfully")
                except Exception as e:
                    print(f"❌ Multi-day visualization failed: {e}")
                    import traceback
                    traceback.print_exc()
                print("📊 Creating gas cost analysis visualizations...")
                try:
                    visualizer.visualize_gas_cost_analysis(self.results['multi_day'], selected_approaches)
                    print("✅ Gas visualizations completed successfully")
                except Exception as e:
                    print(f"❌ Gas visualization failed: {e}")
                    import traceback
                    traceback.print_exc()
            

            
            print(f"📊 Performance visualizations created in: {visualizations_dir}")
            
        except Exception as e:
            print(f"⚠️  Visualization creation failed: {e}")

def setup_web3_connection():
    """Setup Web3 connection for blockchain testing."""
    try:
        web3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
        if web3.is_connected():
            web3.eth.default_account = web3.eth.accounts[0]
            print("✅ Connected to Hardhat for blockchain testing")
            return web3
        else:
            print("⚠️  No Hardhat connection - running without blockchain testing")
            return None
    except Exception as e:
        print(f"⚠️  Web3 connection failed: {e}")
        return None

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Comprehensive Test Suite Runner')
    parser.add_argument('--test-type', choices=['all', 'multi-day'], 
                       default='all', help='Type of tests to run (comprehensive multi-day analysis with gas cost optimization)')
    parser.add_argument('--scale', choices=['small', 'medium', 'large', 'xlarge', 'massive', 'ultra', 'extreme'], 
                       default='medium', help='Test scale (massive=25k docs, ultra=50k docs, extreme=100k docs with aggressive sampling)')
    parser.add_argument('--days', type=int, help='Number of days for multi-day tests')
    parser.add_argument('--documents', type=int, help='Number of documents to generate')
    parser.add_argument('--traffic-events', type=int, help='Number of traffic events')
    parser.add_argument('--skip-blockchain', action='store_true', 
                       help='Skip blockchain connection')
    parser.add_argument('--gas-price', type=int, default=20, 
                       help='Gas price in gwei for cost analysis')
    parser.add_argument('--approaches', nargs='*', 
                       choices=['hierarchical', 'traditional_multiproof', 'traditional_single_proof', 
                               'traditional_huffman', 'clustered_flat', 'clustered_flat_with_merkle'],
                       help='Select specific approaches to test (default: all approaches)')
    parser.add_argument('--learning-mode', choices=['daily', 'immediate', 'batch', 'hybrid', 'disabled'],
                       default='daily', help='Learning mode for adaptive approaches (default: daily)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for batch learning mode (default: 10)')
    
    args = parser.parse_args()
    
    # Setup learning mode for adaptive approaches
    from learning_config import set_learning_mode, LearningMode, get_learning_config
    
    # Map string to enum
    learning_mode_map = {
        'daily': LearningMode.DAILY,
        'immediate': LearningMode.IMMEDIATE,
        'batch': LearningMode.BATCH,
        'hybrid': LearningMode.HYBRID,
        'disabled': LearningMode.DISABLED
    }
    
    selected_learning_mode = learning_mode_map[args.learning_mode]
    set_learning_mode(selected_learning_mode, batch_size=args.batch_size, verbose_logging=True)
    
    config = get_learning_config()
    print(f"🧠 Learning Mode Configuration:")
    print(f"   Mode: {config.mode.value}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Verbose logging: {config.verbose_logging}")
    print(f"   Max rebuilds per day: {config.max_rebuilds_per_day}")
    print()
    
    # Setup Web3 connection
    web3 = None
    if not args.skip_blockchain:
        web3 = setup_web3_connection()
        
        # Ensure Web3 connection for on-chain verification
        if web3 is None:
            print("❌ ERROR: Web3 connection required for on-chain verification.")
            print("   Please start Hardhat local node: npx hardhat node")
            print("   Or use --skip-blockchain to run without on-chain verification")
            sys.exit(1)
    
    # Create test runner with selected approaches
    selected_approaches = args.approaches if args.approaches else None
    
    # Validate selected approaches
    if selected_approaches:
        available_approaches = ['hierarchical', 'traditional_multiproof', 'traditional_single_proof', 
                              'traditional_huffman', 'clustered_flat', 'clustered_flat_with_merkle']
        invalid_approaches = [a for a in selected_approaches if a not in available_approaches]
        if invalid_approaches:
            print(f"❌ ERROR: Invalid approaches specified: {', '.join(invalid_approaches)}")
            print(f"   Available approaches: {', '.join(available_approaches)}")
            sys.exit(1)
    
    test_runner = ComprehensiveTestRunner(web3, selected_approaches)
    
    # Prepare custom configuration
    custom_config = {}
    if args.days:
        custom_config['days'] = args.days
    if args.documents:
        custom_config['documents'] = args.documents
    if args.traffic_events:
        custom_config['traffic_events'] = args.traffic_events
    
    # Run tests based on type
    try:
        if args.test_type == 'all':
            results = test_runner.run_all_tests(args.scale, custom_config)
        elif args.test_type == 'multi-day':
            results = test_runner.run_multi_day_tests(args.scale, args.days)
        
        # Print summary
        if isinstance(results, dict) and 'summary' in results:
            summary = results['summary']
            print(f"\n🎯 TEST SUMMARY:")
            print(f"  Total tests run: {summary.get('total_tests_run', 0)}")
            print(f"  Successful tests: {summary.get('successful_tests', 0)}")
            print(f"  Failed tests: {summary.get('failed_tests', 0)}")
            
            if 'key_findings' in summary:
                print(f"\n  Key Findings:")
                for finding in summary['key_findings']:
                    print(f"    • {finding}")
            
            if 'recommendations' in results:
                print(f"\n  Recommendations:")
                for rec in results['recommendations']:
                    print(f"    • {rec}")
        
        print(f"\n✅ Comprehensive tests completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

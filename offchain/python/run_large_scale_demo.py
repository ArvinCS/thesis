#!/usr/bin/env python3
"""
Quick Demo Script for Large Scale Hierarchical Merkle Tree Testing

This script provides easy commands to test your thesis system at scale.
Run with different parameters to see how your hierarchical system performs.
"""

import sys
import argparse
from large_scale_hierarchical_main import LargeScaleHierarchicalSystem
from web3 import Web3

def quick_demo(document_count=1000, skip_blockchain=False):
    """Run a quick demonstration of the large scale system."""
    print(f"🚀 QUICK DEMO: Hierarchical Merkle Tree with {document_count} documents")
    print(f"{'='*60}")
    
    # Setup Web3 connection (optional)
    web3 = None
    if not skip_blockchain:
        try:
            web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
            if web3.is_connected():
                web3.eth.default_account = web3.eth.accounts[0]
                print("✅ Connected to Hardhat for on-chain testing")
            else:
                print("⚠️  No Hardhat connection - running off-chain only")
                web3 = None
        except:
            print("⚠️  No Hardhat connection - running off-chain only")
            web3 = None
    else:
        print("⚠️  Skipping blockchain connection")
    
    # Create and run system
    system = LargeScaleHierarchicalSystem(web3)
    
    # Generate data (faster for demo)
    traffic_events = document_count // 5  # Reasonable ratio
    
    print(f"\n📊 Generating {document_count} documents with {traffic_events} traffic events...")
    documents, traffic_logs, properties_by_province = system.generate_large_scale_data(
        document_count=document_count,
        traffic_events=traffic_events,
        save_files=False  # Don't save for quick demo
    )
    
    print(f"\n🌳 Building hierarchical tree system...")
    jurisdiction_root, system_info = system.build_large_scale_hierarchical_system(
        documents, traffic_logs
    )
    
    if web3:
        print(f"\n⛓️  Updating smart contract...")
        success = system.update_on_chain_roots()
        if success:
            print("✅ Smart contract updated successfully")
        else:
            print("❌ Smart contract update failed")
    
    print(f"\n🧪 Running verification tests...")
    verification_results = system.run_large_scale_verification_tests(
        properties_by_province, 
        num_test_scenarios=5  # Fewer for demo
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📋 DEMO SUMMARY")
    print(f"{'='*60}")
    
    perf = system.performance_stats
    if 'data_generation' in perf:
        gen_time = perf['data_generation']['duration_seconds']
        print(f"📊 Data Generation: {gen_time:.2f}s ({document_count/gen_time:.0f} docs/sec)")
    
    if 'system_build' in perf:
        build_time = perf['system_build']['duration_seconds']
        print(f"🌳 System Build: {build_time:.2f}s")
        print(f"📁 Provinces: {perf['system_build']['total_provinces']}")
    
    successful_tests = [r for r in verification_results if 'error' not in r and r.get('local_verification_passed')]
    if successful_tests:
        avg_time = sum(r['verification_time_seconds'] for r in successful_tests) / len(successful_tests)
        avg_docs = sum(r['total_documents'] for r in successful_tests) / len(successful_tests)
        print(f"🧪 Verification Tests: {len(successful_tests)} successful")
        print(f"⚡ Avg verification time: {avg_time:.4f}s ({avg_docs:.1f} docs)")
    
    print(f"\n✅ Demo completed successfully!")
    return system

def benchmark_comparison(document_count=2000):
    """Run a comprehensive benchmark comparison."""
    print(f"🏁 BENCHMARK: Hierarchical vs Traditional Merkle Trees")
    print(f"Document Count: {document_count}")
    print(f"{'='*60}")
    
    # Setup system
    try:
        web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
        if web3.is_connected():
            web3.eth.default_account = web3.eth.accounts[0]
        else:
            web3 = None
    except:
        web3 = None
    
    system = LargeScaleHierarchicalSystem(web3)
    
    # Run comprehensive benchmark
    report = system.run_complete_large_scale_benchmark(
        document_count=document_count,
        traffic_events=document_count // 4
    )
    
    print(f"\n📊 Benchmark report saved!")
    return report

def stress_test(max_documents=10000):
    """Run stress test with increasing document counts."""
    print(f"💪 STRESS TEST: Scaling up to {max_documents} documents")
    print(f"{'='*60}")
    
    # Test at multiple scales
    scales = [500, 1000, 2000, 5000]
    if max_documents > 5000:
        scales.extend([7500, 10000])
    
    scales = [s for s in scales if s <= max_documents]
    
    results = []
    
    for scale in scales:
        print(f"\n🔬 Testing scale: {scale} documents")
        print(f"-" * 40)
        
        system = LargeScaleHierarchicalSystem(None)  # No blockchain for stress test
        
        documents, traffic_logs, properties = system.generate_large_scale_data(
            document_count=scale,
            traffic_events=scale // 5,
            save_files=False
        )
        
        jurisdiction_root, system_info = system.build_large_scale_hierarchical_system(
            documents, traffic_logs
        )
        
        # Quick verification test
        verification_results = system.run_large_scale_verification_tests(
            properties, num_test_scenarios=3
        )
        
        # Record results
        perf = system.performance_stats
        results.append({
            'document_count': scale,
            'build_time': perf.get('system_build', {}).get('duration_seconds', 0),
            'successful_verifications': len([r for r in verification_results if r.get('local_verification_passed')])
        })
        
        print(f"✅ {scale} documents: {perf.get('system_build', {}).get('duration_seconds', 0):.2f}s build time")
    
    # Print scaling analysis
    print(f"\n📈 SCALING ANALYSIS")
    print(f"{'='*60}")
    print(f"{'Documents':<12} {'Build Time':<12} {'Docs/Sec':<12} {'Tests':<8}")
    print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    
    for result in results:
        docs = result['document_count']
        time_s = result['build_time']
        docs_per_sec = docs / time_s if time_s > 0 else 0
        tests = result['successful_verifications']
        
        print(f"{docs:<12} {time_s:<12.2f} {docs_per_sec:<12.0f} {tests:<8}")
    
    return results

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description='Large Scale Hierarchical Merkle Tree Demo')
    parser.add_argument('command', choices=['demo', 'benchmark', 'stress'], 
                       help='Command to run')
    parser.add_argument('--documents', type=int, default=1000,
                       help='Number of documents to generate')
    parser.add_argument('--skip-blockchain', action='store_true',
                       help='Skip blockchain connection')
    parser.add_argument('--max-documents', type=int, default=10000,
                       help='Maximum documents for stress test')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        quick_demo(args.documents, args.skip_blockchain)
    elif args.command == 'benchmark':
        benchmark_comparison(args.documents)
    elif args.command == 'stress':
        stress_test(args.max_documents)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run default demo
        print("No arguments provided. Running default demo...")
        quick_demo(1000, skip_blockchain=False)
    else:
        main()

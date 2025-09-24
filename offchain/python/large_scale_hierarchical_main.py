"""
Large Scale Hierarchical Merkle Tree System

Optimized version of hierarchical_main.py for handling thousands of documents
with memory optimization and batch processing capabilities.
"""

import json
import os
import time
from web3 import Web3
from optimized_tree_builder import Document
from jurisdiction_tree_manager import JurisdictionTreeManager
from large_scale_generator import LargeScaleDocumentGenerator, RealisticTrafficGenerator
from benchmark_suite import ComprehensiveBenchmarkSuite
from report_organizer import get_report_organizer, save_organized_file

# Configuration for large scale testing
HARDHAT_URL = "http://127.0.0.1:8545"
ARTIFACT_PATH = '../../artifacts/contracts/HierarchicalMerkleVerifier.sol/HierarchicalMerkleVerifier.json'

class LargeScaleHierarchicalSystem:
    """Optimized hierarchical system for large scale document processing."""
    
    def __init__(self, web3_instance=None):
        self.web3 = web3_instance
        self.hierarchical_contract = None
        self.current_jurisdiction_manager = None
        self.performance_stats = {}
        
        if self.web3:
            self._setup_contract()
    
    def _setup_contract(self):
        """Setup hierarchical contract for on-chain operations."""
        try:
            with open(ARTIFACT_PATH, 'r') as f:
                artifact = json.load(f)
            
            contract_address = artifact['networks']['31337']['address']
            self.hierarchical_contract = self.web3.eth.contract(
                address=contract_address, 
                abi=artifact['abi']
            )
            
            print(f"Connected to HierarchicalMerkleVerifier at {contract_address}")
            
        except Exception as e:
            print(f"Warning: Could not load hierarchical contract: {e}")
    
    def generate_large_scale_data(self, document_count=5000, traffic_events=2000, save_files=True):
        """Generate large scale test data with realistic patterns."""
        print(f"=== GENERATING LARGE SCALE DATA ===")
        print(f"Target Documents: {document_count}")
        print(f"Target Traffic Events: {traffic_events}")
        
        start_time = time.time()
        
        # Generate documents
        doc_generator = LargeScaleDocumentGenerator(target_document_count=document_count)
        documents = doc_generator.generate_documents()
        
        # Generate traffic logs
        traffic_generator = RealisticTrafficGenerator(documents, doc_generator.properties_by_province)
        traffic_logs = traffic_generator.generate_traffic_logs(traffic_events)
        
        generation_time = time.time() - start_time
        
        # Statistics
        stats = doc_generator.get_generation_stats()
        traffic_stats = traffic_generator.analyze_traffic_patterns(traffic_logs)
        
        print(f"\nGeneration completed in {generation_time:.2f}s")
        print(f"Documents: {stats['total_documents']} across {stats['provinces']} provinces")
        print(f"Properties: {stats['total_properties']}")
        print(f"Traffic Events: {traffic_stats['total_events']}")
        print(f"Cross-Province Events: {traffic_stats['cross_province_events']} ({traffic_stats['cross_province_events']/traffic_stats['total_events']*100:.1f}%)")
        
        if save_files:
            # Save with timestamp for uniqueness
            timestamp = int(time.time())
            docs_file = f'large_scale_documents_{document_count}_{timestamp}.json'
            traffic_file = f'large_scale_traffic_{traffic_events}_{timestamp}.json'
            
            doc_generator.save_documents(docs_file)
            traffic_generator.save_traffic_logs(traffic_logs, traffic_file)
            
            print(f"Saved data files: {docs_file}, {traffic_file}")
        
        self.performance_stats['data_generation'] = {
            'duration_seconds': generation_time,
            'documents_generated': len(documents),
            'traffic_events_generated': len(traffic_logs)
        }
        
        return documents, traffic_logs, doc_generator.properties_by_province
    
    def build_large_scale_hierarchical_system(self, documents, traffic_logs):
        """Build hierarchical system optimized for large datasets."""
        print(f"\n=== BUILDING LARGE SCALE HIERARCHICAL SYSTEM ===")
        print(f"Processing {len(documents)} documents with {len(traffic_logs)} traffic events...")
        
        start_time = time.time()
        
        # Build jurisdiction tree manager
        self.current_jurisdiction_manager = JurisdictionTreeManager(documents, traffic_logs)
        jurisdiction_root = self.current_jurisdiction_manager.build_all_trees()
        
        build_time = time.time() - start_time
        
        # Get system information
        system_info = self.current_jurisdiction_manager.get_system_info()
        
        print(f"Hierarchical system built in {build_time:.2f}s")
        print(f"Jurisdiction Root: {jurisdiction_root}")
        print(f"Total Provinces: {system_info['total_provinces']}")
        print(f"Total Documents: {system_info['total_documents']}")
        
        # Print province-level statistics
        print(f"\nProvince Distribution:")
        for province, info in system_info['provinces'].items():
            print(f"  {province}: {info['document_count']} docs, {info['property_count']} properties")
        
        self.performance_stats['system_build'] = {
            'duration_seconds': build_time,
            'jurisdiction_root': jurisdiction_root,
            'total_provinces': system_info['total_provinces'],
            'total_documents': system_info['total_documents']
        }
        
        return jurisdiction_root, system_info
    
    def update_on_chain_roots(self):
        """Update smart contract with new hierarchical roots."""
        if not self.hierarchical_contract or not self.current_jurisdiction_manager:
            print("Cannot update on-chain roots: missing contract or jurisdiction manager")
            return False
        
        print(f"\n=== UPDATING ON-CHAIN ROOTS ===")
        
        try:
            start_time = time.time()
            
            # Prepare data for contract update
            provinces = sorted(self.current_jurisdiction_manager.provinces)
            province_roots = [self.current_jurisdiction_manager.province_builders[p].merkle_root for p in provinces]
            jurisdiction_root_hex = "0x" + self.current_jurisdiction_manager.jurisdiction_root
            province_roots_hex = ["0x" + root for root in province_roots]
            
            # Update all roots in a single transaction
            tx_hash = self.hierarchical_contract.functions.updateHierarchicalRoots(
                jurisdiction_root_hex,
                provinces,
                province_roots_hex
            ).transact()
            
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            update_time = time.time() - start_time
            
            print(f"Successfully updated hierarchical roots in {update_time:.2f}s")
            print(f"Transaction hash: {receipt.transactionHash.hex()}")
            print(f"Gas used: {receipt.gasUsed:,}")
            
            self.performance_stats['on_chain_update'] = {
                'duration_seconds': update_time,
                'gas_used': receipt.gasUsed,
                'tx_hash': receipt.transactionHash.hex()
            }
            
            return True
            
        except Exception as e:
            print(f"ERROR updating hierarchical roots: {e}")
            return False
    
    def run_large_scale_verification_tests(self, properties_by_province, num_test_scenarios=10):
        """Run multiple verification tests with different scales and patterns."""
        if not self.current_jurisdiction_manager:
            print("No jurisdiction manager available for testing")
            return
        
        print(f"\n=== LARGE SCALE VERIFICATION TESTS ===")
        print(f"Running {num_test_scenarios} test scenarios...")
        
        test_results = []
        province_names = list(properties_by_province.keys())
        
        for i in range(num_test_scenarios):
            print(f"\n--- Test Scenario {i+1}/{num_test_scenarios} ---")
            
            # Create varied test scenarios
            if i < 3:
                # Small cross-province tests
                num_provinces = min(2, len(province_names))
                properties_per_province = [1, 2]
            elif i < 6:
                # Medium cross-province tests
                num_provinces = min(4, len(province_names))
                properties_per_province = [2, 3, 4]
            else:
                # Large cross-province tests
                num_provinces = min(len(province_names), 6)
                properties_per_province = [1, 2, 3]
            
            # Build verification request
            selected_provinces = province_names[:num_provinces]
            verification_request = {}
            total_properties = 0
            total_documents = 0
            
            for province in selected_provinces:
                properties = properties_by_province[province]
                if properties:
                    import random
                    num_props = min(random.choice(properties_per_province), len(properties))
                    selected_props = random.sample(properties, num_props)
                    verification_request[province] = [prop['full_id'] for prop in selected_props]
                    total_properties += len(selected_props)
                    
                    # Count documents
                    for prop in selected_props:
                        total_documents += len(prop['documents'])
            
            if not verification_request:
                continue
            
            print(f"Testing {total_properties} properties ({total_documents} documents) across {len(verification_request)} provinces")
            
            # Run verification test
            # start_time = time.time()
            
            try:
                # Generate hierarchical proof
                proof_package = self.current_jurisdiction_manager.verify_cross_province_batch(verification_request)
                
                # Verify locally
                # is_valid_locally, reason = self.current_jurisdiction_manager.verify_proof_package_locally(proof_package)
                
                # verification_time = time.time() - start_time
                
                test_result = {
                    'scenario': i + 1,
                    'provinces_involved': len(verification_request),
                    'total_properties': total_properties,
                    'total_documents': total_documents,
                    # 'verification_time_seconds': verification_time,
                    # 'local_verification_passed': is_valid_locally,
                    # 'reason': reason,
                    'proof_size_bytes': self._calculate_proof_size(proof_package)
                }
                
                # Try on-chain verification if contract available
                if self.hierarchical_contract and is_valid_locally:
                    try:
                        gas_estimate = self._estimate_on_chain_gas(proof_package)
                        test_result['estimated_gas'] = gas_estimate
                    except Exception as e:
                        test_result['gas_estimation_error'] = str(e)
                
                test_results.append(test_result)
                
                print(f"  ✅ Verification: {is_valid_locally} in {verification_time:.4f}s")
                print(f"  📊 Proof size: {test_result['proof_size_bytes']} bytes")
                if 'estimated_gas' in test_result:
                    print(f"  ⛽ Estimated gas: {test_result['estimated_gas']:,}")
                
            except Exception as e:
                print(f"  ❌ Test failed: {e}")
                test_results.append({
                    'scenario': i + 1,
                    'error': str(e)
                })
        
        # Save test results using organized structure
        results_filename = f'large_scale_verification_results_{int(time.time())}.json'
        results_file = save_organized_file(test_results, results_filename, "verification_results")
        
        print(f"\nTest results saved to: {results_file}")
        
        # Print summary statistics
        successful_tests = [r for r in test_results if 'error' not in r and r.get('local_verification_passed')]
        if successful_tests:
            avg_time = sum(r['verification_time_seconds'] for r in successful_tests) / len(successful_tests)
            avg_docs = sum(r['total_documents'] for r in successful_tests) / len(successful_tests)
            avg_provinces = sum(r['provinces_involved'] for r in successful_tests) / len(successful_tests)
            
            print(f"\nSUMMARY STATISTICS:")
            print(f"  Successful tests: {len(successful_tests)}/{len(test_results)}")
            print(f"  Average verification time: {avg_time:.4f}s")
            print(f"  Average documents per test: {avg_docs:.1f}")
            print(f"  Average provinces per test: {avg_provinces:.1f}")
        
        self.performance_stats['verification_tests'] = test_results
        
        return test_results
    
    def _calculate_proof_size(self, proof_package):
        """Calculate total proof size in bytes."""
        total_size = 0
        
        # Province proofs
        for province_proof in proof_package['province_proofs'].values():
            total_size += len(province_proof['proof']) * 32  # 32 bytes per hash
            total_size += len(province_proof['flags'])  # 1 byte per flag
            total_size += len(province_proof['document_hashes']) * 32
        
        # Jurisdiction proof
        total_size += len(proof_package['jurisdiction_proof']['proof']) * 32
        total_size += len(proof_package['jurisdiction_proof']['flags'])
        
        return total_size
    
    def _estimate_on_chain_gas(self, proof_package):
        """Estimate gas cost for on-chain verification."""
        if not self.hierarchical_contract:
            return None
        
        # Prepare data for on-chain verification
        province_proofs = []
        province_flags = []
        province_leaves_arrays = []
        provinces_involved = proof_package['jurisdiction_proof']['provinces_involved']
        
        for province in provinces_involved:
            province_proof_data = proof_package['province_proofs'][province]
            # Convert to bytes32 format
            proof_bytes = [bytes.fromhex(p) for p in province_proof_data['proof']]
            leaves_bytes = [bytes.fromhex(l) for l in province_proof_data['document_hashes']]
            
            province_proofs.append(proof_bytes)
            province_flags.append(province_proof_data['flags'])
            province_leaves_arrays.append(leaves_bytes)
        
        jurisdiction_proof_bytes = [bytes.fromhex(p) for p in proof_package['jurisdiction_proof']['proof']]
        jurisdiction_flags = proof_package['jurisdiction_proof']['flags']
        
        # Estimate gas
        return self.hierarchical_contract.functions.verifyHierarchicalBatch(
            province_proofs,
            province_flags,
            province_leaves_arrays,
            provinces_involved,
            jurisdiction_proof_bytes,
            jurisdiction_flags
        ).estimate_gas()
    
    def run_complete_large_scale_benchmark(self, document_count=5000, traffic_events=2000):
        """Run complete large scale benchmark including comparison with traditional approach."""
        print(f"{'='*80}")
        print(f"COMPLETE LARGE SCALE HIERARCHICAL MERKLE TREE BENCHMARK")
        print(f"{'='*80}")
        print(f"Target Scale: {document_count} documents, {traffic_events} traffic events")
        
        overall_start_time = time.time()
        
        # Step 1: Generate large scale data
        documents, traffic_logs, properties_by_province = self.generate_large_scale_data(
            document_count, traffic_events
        )
        
        # Step 2: Build hierarchical system
        jurisdiction_root, system_info = self.build_large_scale_hierarchical_system(
            documents, traffic_logs
        )
        
        # Step 3: Update on-chain roots (if available)
        if self.web3:
            self.update_on_chain_roots()
        
        # Step 4: Run verification tests
        verification_results = self.run_large_scale_verification_tests(properties_by_province)
        
        # Step 5: Run comprehensive benchmark comparison
        print(f"\n=== RUNNING COMPREHENSIVE BENCHMARK COMPARISON ===")
        benchmark_suite = ComprehensiveBenchmarkSuite(self.web3)
        comparison_results = benchmark_suite.run_comprehensive_benchmark(
            target_documents=len(documents),
            target_traffic_events=len(traffic_logs)
        )
        
        total_time = time.time() - overall_start_time
        
        # Compile final performance report
        performance_report = {
            'test_configuration': {
                'document_count': len(documents),
                'traffic_events': len(traffic_logs),
                'province_count': len(properties_by_province),
                'total_duration_seconds': total_time
            },
            'hierarchical_system_performance': self.performance_stats,
            'verification_test_results': verification_results,
            'comprehensive_comparison': comparison_results,
            'timestamp': time.time()
        }
        
        # Save complete performance report using organized structure
        report_filename = f'complete_large_scale_report_{document_count}docs_{int(time.time())}.json'
        report_file = save_organized_file(performance_report, report_filename, "performance_reports")
        
        print(f"\n{'='*80}")
        print(f"BENCHMARK COMPLETED IN {total_time:.2f}s")
        print(f"{'='*80}")
        print(f"Complete performance report saved to: {report_file}")
        
        return performance_report

def main():
    """Main execution for large scale testing."""
    print("=== LARGE SCALE HIERARCHICAL MERKLE TREE SYSTEM ===")
    
    # Setup Web3 connection
    try:
        web3 = Web3(Web3.HTTPProvider(HARDHAT_URL))
        if web3.is_connected():
            web3.eth.default_account = web3.eth.accounts[0]
            print(f"Connected to Hardhat: {web3.is_connected()}")
            print(f"Default account: {web3.eth.default_account}")
        else:
            print("Warning: Could not connect to Hardhat")
            web3 = None
    except Exception as e:
        print(f"Warning: Web3 connection failed: {e}")
        web3 = None
    
    # Create large scale system
    large_scale_system = LargeScaleHierarchicalSystem(web3)
    
    # Test different scales
    test_scales = [
        {'documents': 1000, 'traffic': 500},
        {'documents': 2000, 'traffic': 1000},
        {'documents': 5000, 'traffic': 2000},
    ]
    
    for scale in test_scales:
        print(f"\n{'='*100}")
        print(f"TESTING SCALE: {scale['documents']} DOCUMENTS")
        print(f"{'='*100}")
        
        report = large_scale_system.run_complete_large_scale_benchmark(
            document_count=scale['documents'],
            traffic_events=scale['traffic']
        )
        
        # Print key metrics
        if 'hierarchical_system_performance' in report:
            perf = report['hierarchical_system_performance']
            if 'system_build' in perf:
                build_time = perf['system_build']['duration_seconds']
                total_docs = perf['system_build']['total_documents']
                print(f"\nKEY METRICS for {scale['documents']} documents:")
                print(f"  System build time: {build_time:.3f}s")
                print(f"  Documents per second: {total_docs/build_time:.0f}")
                print(f"  Total provinces: {perf['system_build']['total_provinces']}")

if __name__ == "__main__":
    main()

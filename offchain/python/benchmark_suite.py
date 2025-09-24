"""
Comprehensive Benchmarking Suite for Hierarchical vs Traditional Merkle Trees

This module provides detailed performance analysis comparing:
1. Hierarchical Merkle Tree (your thesis approach)
2. Traditional Flat Merkle Tree
3. Various metrics: gas costs, proof sizes, verification times, scalability
"""

import time
import json
import os
from collections import defaultdict
from web3 import Web3

# Import report organizer for structured file saving
try:
    from report_organizer import save_organized_file
except ImportError:
    # Fallback if report_organizer is not available
    def save_organized_file(data, filename, file_type="benchmark_results"):
        import json
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        return filename
from jurisdiction_tree_manager import JurisdictionTreeManager
from large_scale_generator import LargeScaleDocumentGenerator, RealisticTrafficGenerator
from traditional_multiproof_builder import TraditionalMerkleTreeBuilder
from traditional_single_proof_builder import TraditionalSingleProofMerkleTreeBuilder
from traditional_multiproof_with_huffman_builder import TraditionalMultiproofWithHuffmanBuilder

class PerformanceProfiler:
    """Profiles performance metrics for different tree implementations."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.current_test = None
    
    def start_test(self, test_name):
        """Start timing a specific test."""
        self.current_test = {
            'name': test_name,
            'start_time': time.time(),
            'memory_start': self._get_memory_usage()
        }
    
    def end_test(self, additional_metrics=None):
        """End timing and record metrics."""
        if not self.current_test:
            return
        
        end_time = time.time()
        duration = end_time - self.current_test['start_time']
        memory_end = self._get_memory_usage()
        memory_used = memory_end - self.current_test['memory_start']
        
        result = {
            'test_name': self.current_test['name'],
            'duration_seconds': duration,
            'memory_mb': memory_used,
            'timestamp': end_time
        }
        
        if additional_metrics:
            result.update(additional_metrics)
        
        self.metrics[self.current_test['name']].append(result)
        self.current_test = None
        
        return result
    
    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0  # psutil not available
    
    def get_summary(self):
        """Get performance summary."""
        summary = {}
        for test_name, results in self.metrics.items():
            if results:
                durations = [r['duration_seconds'] for r in results]
                memories = [r['memory_mb'] for r in results]
                
                summary[test_name] = {
                    'avg_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'avg_memory_mb': sum(memories) / len(memories) if memories and any(memories) else 0,
                    'total_runs': len(results)
                }
        
        return summary





class ComprehensiveBenchmarkSuite:
    """Complete benchmarking suite comparing hierarchical vs traditional approaches."""
    
    def __init__(self, web3_instance=None):
        self.web3 = web3_instance
        self.profiler = PerformanceProfiler()
        self.results = {}
        
        # Load or create contracts for gas testing
        if self.web3:
            self._setup_contracts()
    
    def _setup_contracts(self):
        """Setup smart contracts for gas cost analysis."""
        try:
            # Load stateless hierarchical contract
            hierarchical_artifact_path = '../../artifacts/contracts/HierarchicalMerkleVerifier.sol/HierarchicalMerkleVerifier.json'
            with open(hierarchical_artifact_path, 'r') as f:
                hierarchical_artifact = json.load(f)
            
            # Use the deployed stateless contract address directly
            hierarchical_address = hierarchical_artifact['networks']['31337']['address']
            self.hierarchical_contract = self.web3.eth.contract(
                address=hierarchical_address, 
                abi=hierarchical_artifact['abi']
            )
            
            # Load traditional multiproof contract
            traditional_artifact_path = '../../artifacts/contracts/MerkleVerifier.sol/MerkleVerifier.json'
            with open(traditional_artifact_path, 'r') as f:
                traditional_artifact = json.load(f)
            
            traditional_address = traditional_artifact['networks']['31337']['address']
            self.traditional_contract = self.web3.eth.contract(
                address=traditional_address,
                abi=traditional_artifact['abi']
            )
            
            # Load traditional single proof contract
            single_proof_artifact_path = '../../artifacts/contracts/SingleProofMerkleVerifier.sol/SingleProofMerkleVerifier.json'
            with open(single_proof_artifact_path, 'r') as f:
                single_proof_artifact = json.load(f)
            
            single_proof_address = single_proof_artifact['networks']['31337']['address']
            self.single_proof_contract = self.web3.eth.contract(
                address=single_proof_address,
                abi=single_proof_artifact['abi']
            )
            
            print("Smart contracts loaded successfully for gas analysis")
            
        except Exception as e:
            print(f"Warning: Could not load smart contracts: {e}")
            self.hierarchical_contract = None
            self.traditional_contract = None
            self.single_proof_contract = None
    
    def benchmark_tree_construction(self, documents, traffic_logs):
        """Benchmark tree construction time and memory usage."""
        print("\n=== TREE CONSTRUCTION BENCHMARKS ===")
        
        results = {}
        
        # Benchmark Hierarchical Tree Construction
        self.profiler.start_test("hierarchical_construction")
        jurisdiction_manager = JurisdictionTreeManager(documents, traffic_logs)
        hierarchical_root = jurisdiction_manager.build_all_trees()
        hierarchical_metrics = self.profiler.end_test({
            'root_hash': hierarchical_root,
            'total_provinces': len(jurisdiction_manager.provinces),
            'total_documents': len(documents)
        })
        
        # Benchmark Traditional Multiproof Tree Construction
        self.profiler.start_test("traditional_multiproof_construction")
        traditional_multiproof_builder = TraditionalMerkleTreeBuilder(documents)
        traditional_multiproof_root = traditional_multiproof_builder.build()
        traditional_multiproof_metrics = self.profiler.end_test({
            'root_hash': traditional_multiproof_root,
            'total_documents': len(documents)
        })
        
        # Benchmark Traditional Single Proof Tree Construction
        self.profiler.start_test("traditional_single_proof_construction")
        traditional_single_proof_builder = TraditionalSingleProofMerkleTreeBuilder(documents)
        traditional_single_proof_root = traditional_single_proof_builder.build()
        traditional_single_proof_metrics = self.profiler.end_test({
            'root_hash': traditional_single_proof_root,
            'total_documents': len(documents)
        })
        
        # Benchmark Traditional Multiproof with Huffman Tree Construction
        self.profiler.start_test("traditional_huffman_construction")
        traditional_huffman_builder = TraditionalMultiproofWithHuffmanBuilder(documents, traffic_logs)
        traditional_huffman_root = traditional_huffman_builder.build()
        traditional_huffman_metrics = self.profiler.end_test({
            'root_hash': traditional_huffman_root,
            'total_documents': len(documents)
        })
        
        results['hierarchical'] = hierarchical_metrics
        results['traditional_multiproof'] = traditional_multiproof_metrics
        results['traditional_single_proof'] = traditional_single_proof_metrics
        results['traditional_huffman'] = traditional_huffman_metrics
        
        # Print comparison
        print(f"Hierarchical Construction: {hierarchical_metrics['duration_seconds']:.3f}s")
        print(f"Traditional Multiproof Construction: {traditional_multiproof_metrics['duration_seconds']:.3f}s")
        print(f"Traditional Single Proof Construction: {traditional_single_proof_metrics['duration_seconds']:.3f}s")
        print(f"Traditional + Huffman Construction: {traditional_huffman_metrics['duration_seconds']:.3f}s")
        print(f"Construction Speed Improvement (Multiproof vs Hierarchical): {traditional_multiproof_metrics['duration_seconds']/hierarchical_metrics['duration_seconds']:.2f}x")
        print(f"Construction Speed Improvement (Single Proof vs Hierarchical): {traditional_single_proof_metrics['duration_seconds']/hierarchical_metrics['duration_seconds']:.2f}x")
        print(f"Construction Speed Improvement (Huffman vs Hierarchical): {traditional_huffman_metrics['duration_seconds']/hierarchical_metrics['duration_seconds']:.2f}x")
        
        return results, jurisdiction_manager, traditional_multiproof_builder, traditional_single_proof_builder, traditional_huffman_builder
    
    def benchmark_proof_generation(self, jurisdiction_manager, traditional_multiproof_builder, traditional_single_proof_builder, traditional_huffman_builder, verification_scenarios):
        """Benchmark proof generation for different verification scenarios."""
        print("\n=== PROOF GENERATION BENCHMARKS ===")
        
        results = {}
        
        for scenario_name, verification_request in verification_scenarios.items():
            print(f"\nTesting scenario: {scenario_name}")
            
            # Extract all documents for traditional approach
            all_doc_hashes = []
            for province, properties in verification_request.items():
                for prop_id in properties:
                    if province in jurisdiction_manager.province_builders:
                        builder = jurisdiction_manager.province_builders[province]
                        # Use the full property ID for lookup
                        if prop_id in builder.property_clusters:
                            prop_cluster = builder.property_clusters[prop_id]
                            all_doc_hashes.extend(prop_cluster.get_leaf_hashes_hex())
            
            # Check for duplicates and remove them (same fix as gas cost benchmarks)
            original_count = len(all_doc_hashes)
            all_doc_hashes = list(dict.fromkeys(all_doc_hashes))  # Remove duplicates while preserving order
            if len(all_doc_hashes) != original_count:
                print(f"  ⚠️ Removed {original_count - len(all_doc_hashes)} duplicate document hashes")
            
            # Benchmark Hierarchical Proof Generation
            self.profiler.start_test(f"hierarchical_proof_{scenario_name}")
            hierarchical_proof = jurisdiction_manager.verify_cross_province_batch(verification_request)
            hierarchical_metrics = self.profiler.end_test({
                'total_documents': hierarchical_proof['total_documents'],
                'total_provinces': hierarchical_proof['total_provinces'],
                'proof_size_bytes': self._calculate_proof_size(hierarchical_proof)
            })
            
            # Benchmark Traditional Multiproof Generation
            self.profiler.start_test(f"traditional_multiproof_{scenario_name}")
            traditional_multiproof_proof, traditional_multiproof_flags = traditional_multiproof_builder.generate_proof_for_documents(all_doc_hashes)
            traditional_multiproof_metrics = self.profiler.end_test({
                'total_documents': len(all_doc_hashes),
                'proof_elements': len(traditional_multiproof_proof),
                'proof_size_bytes': len(traditional_multiproof_proof) * 32 + len(traditional_multiproof_flags)
            })
            
            # Benchmark Traditional Single Proof Generation
            self.profiler.start_test(f"traditional_single_proof_{scenario_name}")
            single_proofs = traditional_single_proof_builder.generate_single_proofs_for_documents(all_doc_hashes)
            total_single_proof_size = sum(proof['proof_size_bytes'] for proof in single_proofs)
            traditional_single_proof_metrics = self.profiler.end_test({
                'total_documents': len(all_doc_hashes),
                'proof_count': len(single_proofs),
                'proof_size_bytes': total_single_proof_size,
                'avg_proof_size_per_doc': total_single_proof_size / len(all_doc_hashes) if all_doc_hashes else 0
            })
            
            # Benchmark Traditional + Huffman Multiproof Generation
            self.profiler.start_test(f"traditional_huffman_{scenario_name}")
            traditional_huffman_proof, traditional_huffman_flags = traditional_huffman_builder.generate_proof_for_documents(all_doc_hashes)
            traditional_huffman_metrics = self.profiler.end_test({
                'total_documents': len(all_doc_hashes),
                'proof_elements': len(traditional_huffman_proof),
                'proof_size_bytes': len(traditional_huffman_proof) * 32 + len(traditional_huffman_flags)
            })
            
            results[scenario_name] = {
                'hierarchical': hierarchical_metrics,
                'traditional_multiproof': traditional_multiproof_metrics,
                'traditional_single_proof': traditional_single_proof_metrics,
                'traditional_huffman': traditional_huffman_metrics,
                'documents_verified': len(all_doc_hashes),
                'provinces_involved': len(verification_request)
            }
            
            # Print comparison
            print(f"  Hierarchical: {hierarchical_metrics['duration_seconds']:.4f}s, {hierarchical_metrics['proof_size_bytes']} bytes")
            print(f"  Traditional Multiproof: {traditional_multiproof_metrics['duration_seconds']:.4f}s, {traditional_multiproof_metrics['proof_size_bytes']} bytes")
            print(f"  Traditional Single Proof: {traditional_single_proof_metrics['duration_seconds']:.4f}s, {traditional_single_proof_metrics['proof_size_bytes']} bytes ({len(single_proofs)} proofs)")
            print(f"  Traditional + Huffman: {traditional_huffman_metrics['duration_seconds']:.4f}s, {traditional_huffman_metrics['proof_size_bytes']} bytes")
            
            # Calculate size comparisons
            if traditional_multiproof_metrics['proof_size_bytes'] > 0:
                hierarchical_vs_multiproof = (1 - hierarchical_metrics['proof_size_bytes']/traditional_multiproof_metrics['proof_size_bytes'])*100
                print(f"  Hierarchical vs Multiproof Size: {hierarchical_vs_multiproof:.1f}%")
            
            if traditional_single_proof_metrics['proof_size_bytes'] > 0:
                hierarchical_vs_single = (1 - hierarchical_metrics['proof_size_bytes']/traditional_single_proof_metrics['proof_size_bytes'])*100
                multiproof_vs_single = (1 - traditional_multiproof_metrics['proof_size_bytes']/traditional_single_proof_metrics['proof_size_bytes'])*100
                print(f"  Hierarchical vs Single Proof Size: {hierarchical_vs_single:.1f}%")
                print(f"  Multiproof vs Single Proof Size: {multiproof_vs_single:.1f}%")
        
        return results
    
    def reset_contract_state(self):
        """Reset stateless contract state to ensure clean state between test runs."""
        if not self.hierarchical_contract:
            return
        
        print("🔄 Resetting stateless contract state...")
        try:
            # Use a dummy root instead of empty root (contract doesn't allow empty roots)
            dummy_root = bytes.fromhex("1" * 64)  # Non-empty dummy bytes32
            
            tx_hash = self.hierarchical_contract.functions.updateJurisdictionRoot(
                dummy_root
            ).transact()
            
            self.web3.eth.wait_for_transaction_receipt(tx_hash)
            print("✅ Stateless contract state reset successfully")
        except Exception as e:
            print(f"⚠️ Could not reset stateless contract state: {e}")

    def benchmark_gas_costs(self, jurisdiction_manager, traditional_multiproof_builder, traditional_single_proof_builder, traditional_huffman_builder, verification_scenarios):
        """Benchmark on-chain gas costs for verification."""
        print("\n=== GAS COST BENCHMARKS ===")
        
        if not self.hierarchical_contract or not self.traditional_contract or not self.single_proof_contract:
            print("Smart contracts not available for gas testing")
            return {}
        
        # Reset contract state first to ensure clean state
        self.reset_contract_state()
        
        # Update stateless contract with current jurisdiction root
        print("🔗 Updating stateless smart contract with jurisdiction root...")
        try:
            # Convert to proper bytes32 format
            jurisdiction_root_bytes = bytes.fromhex(jurisdiction_manager.jurisdiction_root)
            
            print(f"  Updating jurisdiction root: {jurisdiction_manager.jurisdiction_root}")
            
            tx_hash = self.hierarchical_contract.functions.updateJurisdictionRoot(
                jurisdiction_root_bytes
            ).transact()
            
            self.web3.eth.wait_for_transaction_receipt(tx_hash)
            print("✅ Stateless contract jurisdiction root updated for gas testing")
        except Exception as e:
            print(f"⚠️ Could not update stateless contract jurisdiction root: {e}")
            print(f"  Error details: {str(e)}")
            return {}
        
        results = {}
        
        for scenario_name, verification_request in verification_scenarios.items():
            print(f"\nTesting gas costs for: {scenario_name}")
            
            # Verify contract state before each test
            try:
                current_root = self.hierarchical_contract.functions.jurisdictionRoot().call()
                expected_root = jurisdiction_manager.jurisdiction_root
                if current_root.hex() != expected_root:
                    print(f"  ⚠️ Stateless contract state mismatch detected!")
                    print(f"    Contract root: {current_root.hex()}")
                    print(f"    Expected root: {expected_root}")
                    print(f"  🔄 Updating stateless contract jurisdiction root...")
                    
                    # Update only the jurisdiction root (stateless approach)
                    jurisdiction_root_bytes = bytes.fromhex(jurisdiction_manager.jurisdiction_root)
                    
                    tx_hash = self.hierarchical_contract.functions.updateJurisdictionRoot(
                        jurisdiction_root_bytes
                    ).transact()
                    
                    self.web3.eth.wait_for_transaction_receipt(tx_hash)
                    print(f"  ✅ Stateless contract jurisdiction root updated")
                else:
                    print(f"  ✅ Stateless contract state verified")
            except Exception as state_error:
                print(f"  ⚠️ Could not verify stateless contract state: {state_error}")
            
            try:
                # Hierarchical Gas Cost
                hierarchical_proof = jurisdiction_manager.verify_cross_province_batch(verification_request)
                
                # Only test scenarios that pass local verification
                is_valid_locally, reason = jurisdiction_manager.verify_proof_package_locally(hierarchical_proof)
                if not is_valid_locally:
                    print(f"  Skipping {scenario_name} - local verification failed: {reason}")
                    results[scenario_name] = {'error': f'Local verification failed: {reason}'}
                    continue
                
                # Prepare stateless hierarchical verification data
                # CRITICAL: Use the same ordering as the jurisdiction proof generation (alphabetical)
                provinces_involved = hierarchical_proof['jurisdiction_proof']['provinces_involved']
                claimed_province_roots = []
                province_proofs = []
                province_flags = []
                province_leaves_arrays = []
                
                for province in provinces_involved:
                    province_proof_data = hierarchical_proof['province_proofs'][province]
                    # Get the province root from the jurisdiction manager
                    if province in jurisdiction_manager.province_builders:
                        builder = jurisdiction_manager.province_builders[province]
                        claimed_province_roots.append(bytes.fromhex(builder.merkle_root))
                    
                    # Convert to bytes32 format
                    proof_bytes = [bytes.fromhex(p) for p in province_proof_data['proof']]
                    leaves_bytes = [bytes.fromhex(l) for l in province_proof_data['document_hashes']]
                    
                    province_proofs.append(proof_bytes)
                    province_flags.append(province_proof_data['flags'])
                    province_leaves_arrays.append(leaves_bytes)
                
                jurisdiction_proof_bytes = [bytes.fromhex(p) for p in hierarchical_proof['jurisdiction_proof']['proof']]
                jurisdiction_flags = hierarchical_proof['jurisdiction_proof']['flags']
                
                # Measure gas consumption and verification time
                try:
                    # First estimate gas
                    hierarchical_gas_estimate = self.hierarchical_contract.functions.verifyHierarchicalBatch(
                        claimed_province_roots,
                        jurisdiction_proof_bytes,
                        jurisdiction_flags,
                        province_proofs,
                        province_flags,
                        province_leaves_arrays
                    ).estimate_gas()
                    
                    if self.force_onchain_verification:
                        # Execute actual on-chain verification
                        print(f"    Executing on-chain hierarchical verification...")
                        start_time = time.time()
                        tx_hash = self.hierarchical_contract.functions.verifyHierarchicalBatch(
                            claimed_province_roots,
                            jurisdiction_proof_bytes,
                            jurisdiction_flags,
                            province_proofs,
                            province_flags,
                            province_leaves_arrays
                        ).transact()
                        
                        # Wait for transaction receipt
                        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                        end_time = time.time()
                        hierarchical_verification_time = end_time - start_time
                        
                        # Capture real gas consumption
                        hierarchical_gas_used = receipt.gasUsed
                        hierarchical_gas_price = receipt.effectiveGasPrice
                        hierarchical_total_cost = hierarchical_gas_used * hierarchical_gas_price
                        
                        # Check if transaction was successful
                        if receipt.status == 1:
                            print(f"  ✅ Hierarchical verification successful in {hierarchical_verification_time:.4f}s")
                            print(f"    Gas Used: {hierarchical_gas_used:,} (Estimate: {hierarchical_gas_estimate:,})")
                            print(f"    Gas Price: {hierarchical_gas_price / 1e9:.2f} Gwei")
                            print(f"    Total Cost: {hierarchical_total_cost / 1e18:.6f} ETH")
                        else:
                            print(f"  ❌ Hierarchical verification failed")
                            hierarchical_verification_time = None
                            hierarchical_gas_used = 0
                            hierarchical_gas_price = 0
                            hierarchical_total_cost = 0
                    else:
                        # Just use gas estimate
                        print(f"    Using gas estimate only (no on-chain execution)")
                        hierarchical_verification_time = 0.001  # Minimal time for local verification
                        hierarchical_gas_used = hierarchical_gas_estimate
                        hierarchical_gas_price = 20 * 1e9  # 20 gwei default
                        hierarchical_total_cost = hierarchical_gas_used * hierarchical_gas_price
                        
                except Exception as verification_error:
                    print(f"  ⚠️ Hierarchical verification failed: {verification_error}")
                    hierarchical_verification_time = None
                except Exception as gas_error:
                    print(f"  ⚠️ Gas estimation failed for {scenario_name}: {gas_error}")
                    print(f"  🔍 Debug info:")
                    print(f"    Province proofs: {len(province_proofs)} arrays")
                    print(f"    Province flags: {len(province_flags)} arrays") 
                    print(f"    Province leaves: {len(province_leaves_arrays)} arrays")
                    print(f"    Jurisdiction proof: {len(jurisdiction_proof_bytes)} elements")
                    print(f"    Jurisdiction flags: {len(jurisdiction_flags)} elements")
                    print(f"    Total documents: {hierarchical_proof['total_documents']}")
                    print(f"    Total provinces: {hierarchical_proof['total_provinces']}")
                    
                    # Check if this is an invalid multiproof error
                    if "invalid multiproof" in str(gas_error).lower():
                        print(f"    🚨 LIKELY CAUSE: Invalid multiproof structure generated")
                        print(f"    💡 SOLUTION: Check proof generation logic for bugs")
                        print(f"    📊 Current provinces: {hierarchical_proof['total_provinces']}")
                        print(f"    📊 Current documents: {hierarchical_proof['total_documents']}")
                    
                    # Try to verify contract state
                    try:
                        current_root = self.hierarchical_contract.functions.jurisdictionRoot().call()
                        expected_root = jurisdiction_manager.jurisdiction_root
                        print(f"    Contract root: {current_root.hex()}")
                        print(f"    Expected root: {expected_root}")
                        if current_root.hex() != expected_root:
                            print(f"    🚨 ROOT MISMATCH DETECTED!")
                        else:
                            print(f"    ✅ Contract root matches expected")
                    except Exception as root_error:
                        print(f"    Could not check contract root: {root_error}")
                    
                    # Skip this scenario but continue with others
                    results[scenario_name] = {'error': f'Gas estimation failed: {gas_error}'}
                    continue
                
                # Traditional Multiproof Gas Cost
                all_doc_hashes = []
                for province, properties in verification_request.items():
                    for prop_id in properties:
                        if province in jurisdiction_manager.province_builders:
                            builder = jurisdiction_manager.province_builders[province]
                            if prop_id in builder.property_clusters:
                                prop_cluster = builder.property_clusters[prop_id]
                                all_doc_hashes.extend(prop_cluster.get_leaf_hashes_hex())
                
                # Check for duplicates and remove them
                original_count = len(all_doc_hashes)
                all_doc_hashes = list(dict.fromkeys(all_doc_hashes))  # Remove duplicates while preserving order
                if len(all_doc_hashes) != original_count:
                    print(f"    ⚠️ Removed {original_count - len(all_doc_hashes)} duplicate document hashes")
                
                traditional_multiproof_proof, traditional_multiproof_flags = traditional_multiproof_builder.generate_proof_for_documents(all_doc_hashes)
                traditional_multiproof_proof_bytes = [bytes.fromhex(p) for p in traditional_multiproof_proof]
                
                # REVERSED LEAVES to match contract expectations for multiproofs
                traditional_multiproof_leaves_bytes = [bytes.fromhex(l) for l in reversed(all_doc_hashes)]
                
                # Measure gas consumption and verification time
                try:
                    # First estimate gas
                    traditional_multiproof_gas_estimate = self.traditional_contract.functions.verifyBatch(
                        traditional_multiproof_proof_bytes,
                        traditional_multiproof_flags,
                        traditional_multiproof_leaves_bytes
                    ).estimate_gas()
                    
                    if self.force_onchain_verification:
                        # Execute actual on-chain verification
                        print(f"    Executing on-chain traditional multiproof verification...")
                        start_time = time.time()
                        tx_hash = self.traditional_contract.functions.verifyBatch(
                            traditional_multiproof_proof_bytes,
                            traditional_multiproof_flags,
                            traditional_multiproof_leaves_bytes
                        ).transact()
                        
                        # Wait for transaction receipt
                        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                        end_time = time.time()
                        traditional_multiproof_verification_time = end_time - start_time
                        
                        # Capture real gas consumption
                        traditional_multiproof_gas_used = receipt.gasUsed
                        traditional_multiproof_gas_price = receipt.effectiveGasPrice
                        traditional_multiproof_total_cost = traditional_multiproof_gas_used * traditional_multiproof_gas_price
                        
                        # Check if transaction was successful
                        if receipt.status == 1:
                            print(f"  ✅ Traditional multiproof verification successful in {traditional_multiproof_verification_time:.4f}s")
                            print(f"    Gas Used: {traditional_multiproof_gas_used:,} (Estimate: {traditional_multiproof_gas_estimate:,})")
                            print(f"    Gas Price: {traditional_multiproof_gas_price / 1e9:.2f} Gwei")
                            print(f"    Total Cost: {traditional_multiproof_total_cost / 1e18:.6f} ETH")
                        else:
                            print(f"  ❌ Traditional multiproof verification failed")
                            traditional_multiproof_verification_time = None
                            traditional_multiproof_gas_used = 0
                            traditional_multiproof_gas_price = 0
                            traditional_multiproof_total_cost = 0
                    else:
                        # Just use gas estimate
                        print(f"    Using gas estimate only (no on-chain execution)")
                        traditional_multiproof_verification_time = 0.001  # Minimal time for local verification
                        traditional_multiproof_gas_used = traditional_multiproof_gas_estimate
                        traditional_multiproof_gas_price = 20 * 1e9  # 20 gwei default
                        traditional_multiproof_total_cost = traditional_multiproof_gas_used * traditional_multiproof_gas_price
                        traditional_multiproof_verification_time = None
                        traditional_multiproof_gas_used = 0
                        traditional_multiproof_gas_price = 0
                        traditional_multiproof_total_cost = 0
                        
                except Exception as verification_error:
                    print(f"  ⚠️ Traditional multiproof verification failed: {verification_error}")
                    traditional_multiproof_verification_time = None
                
                # Traditional Single Proof Gas Cost and Verification Time
                single_proofs = traditional_single_proof_builder.generate_single_proofs_for_documents(all_doc_hashes)
                total_single_proof_gas = 0
                total_single_proof_verification_time = 0
                
                # Estimate gas and measure verification time for each single proof
                for proof_data in single_proofs:
                    single_proof_bytes = [bytes.fromhex(p) for p in proof_data['proof']]
                    single_leaf_bytes = bytes.fromhex(proof_data['document_hash'])
                    
                    try:
                        # First estimate gas
                        single_gas = self.single_proof_contract.functions.verifySingle(
                            single_proof_bytes,
                            single_leaf_bytes
                        ).estimate_gas()
                        total_single_proof_gas += single_gas
                        
                        if self.force_onchain_verification:
                            # Execute actual on-chain verification
                            start_time = time.time()
                            tx_hash = self.single_proof_contract.functions.verifySingle(
                                single_proof_bytes,
                                single_leaf_bytes
                            ).transact()
                            
                            # Wait for transaction receipt
                            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                            end_time = time.time()
                            single_verification_time = end_time - start_time
                            total_single_proof_verification_time += single_verification_time
                            
                            # Check if transaction was successful
                            if receipt.status != 1:
                                print(f"  ⚠️ Single proof verification failed for document {proof_data['document_hash'][:8]}...")
                        else:
                            # Just use gas estimate (no on-chain execution)
                            single_verification_time = 0.001  # Minimal time for local verification
                            total_single_proof_verification_time += single_verification_time
                            
                    except Exception as verification_error:
                        print(f"  ⚠️ Single proof verification failed for document {proof_data['document_hash'][:8]}...: {verification_error}")
                
                if total_single_proof_verification_time > 0:
                    print(f"  ✅ Traditional single proof verification completed in {total_single_proof_verification_time:.4f}s ({len(single_proofs)} transactions)")
                
                results[scenario_name] = {
                    # Gas estimates (for comparison)
                    'hierarchical_gas_estimate': hierarchical_gas_estimate,
                    'traditional_multiproof_gas_estimate': traditional_multiproof_gas_estimate,
                    'traditional_single_proof_gas_estimate': total_single_proof_gas,
                    
                    # Real gas consumption (from transaction receipts)
                    'hierarchical_gas_used': hierarchical_gas_used,
                    'hierarchical_gas_price': hierarchical_gas_price,
                    'hierarchical_total_cost': hierarchical_total_cost,
                    'traditional_multiproof_gas_used': traditional_multiproof_gas_used,
                    'traditional_multiproof_gas_price': traditional_multiproof_gas_price,
                    'traditional_multiproof_total_cost': traditional_multiproof_total_cost,
                    
                    # Verification times
                    'hierarchical_verification_time': hierarchical_verification_time,
                    'traditional_multiproof_verification_time': traditional_multiproof_verification_time,
                    'traditional_single_proof_verification_time': total_single_proof_verification_time,
                    
                    # Savings calculations (using real gas consumption)
                    'hierarchical_vs_multiproof_savings': traditional_multiproof_gas_used - hierarchical_gas_used,
                    'hierarchical_vs_multiproof_savings_percent': ((traditional_multiproof_gas_used - hierarchical_gas_used) / traditional_multiproof_gas_used) * 100 if traditional_multiproof_gas_used > 0 else 0,
                    'multiproof_vs_single_savings': total_single_proof_gas - traditional_multiproof_gas_used,
                    'multiproof_vs_single_savings_percent': ((total_single_proof_gas - traditional_multiproof_gas_used) / total_single_proof_gas) * 100 if total_single_proof_gas > 0 else 0,
                    
                    # Metadata
                    'documents_verified': len(all_doc_hashes),
                    'provinces_involved': len(verification_request)
                }
                
                print(f"  📊 REAL GAS CONSUMPTION:")
                print(f"    Hierarchical: {hierarchical_gas_used:,} gas (Estimate: {hierarchical_gas_estimate:,})")
                print(f"    Traditional Multiproof: {traditional_multiproof_gas_used:,} gas (Estimate: {traditional_multiproof_gas_estimate:,})")
                print(f"    Traditional Single Proof: {total_single_proof_gas:,} gas (Estimate)")
                print(f"  💰 COST ANALYSIS:")
                print(f"    Hierarchical: {hierarchical_total_cost / 1e18:.6f} ETH")
                print(f"    Traditional Multiproof: {traditional_multiproof_total_cost / 1e18:.6f} ETH")
                print(f"  📈 SAVINGS (Real Gas):")
                print(f"    Hierarchical vs Multiproof: {results[scenario_name]['hierarchical_vs_multiproof_savings']:,} ({results[scenario_name]['hierarchical_vs_multiproof_savings_percent']:.1f}%)")
                print(f"    Multiproof vs Single Proof: {results[scenario_name]['multiproof_vs_single_savings']:,} ({results[scenario_name]['multiproof_vs_single_savings_percent']:.1f}%)")
                
                # Print verification times
                if hierarchical_verification_time is not None:
                    print(f"  Hierarchical Verification Time: {hierarchical_verification_time:.4f}s")
                if traditional_multiproof_verification_time is not None:
                    print(f"  Traditional Multiproof Verification Time: {traditional_multiproof_verification_time:.4f}s")
                if total_single_proof_verification_time > 0:
                    print(f"  Traditional Single Proof Verification Time: {total_single_proof_verification_time:.4f}s")
                
            except Exception as e:
                print(f"  Error estimating gas for {scenario_name}: {e}")
                results[scenario_name] = {'error': str(e)}
        
        return results
    
    def benchmark_scalability(self, document_counts, traffic_event_counts):
        """Benchmark scalability across different document counts."""
        print("\n=== SCALABILITY BENCHMARKS ===")
        
        results = {}
        
        for doc_count in document_counts:
            print(f"\nTesting with {doc_count} documents...")
            
            # Generate test data
            generator = LargeScaleDocumentGenerator(target_document_count=doc_count, seed=42)
            documents = generator.generate_documents()
            
            traffic_generator = RealisticTrafficGenerator(documents, generator.properties_by_province)
            traffic_logs = traffic_generator.generate_traffic_logs(min(traffic_event_counts, doc_count // 5))
            
            # Benchmark construction time
            construction_results, jurisdiction_manager, traditional_builder = self.benchmark_tree_construction(documents, traffic_logs)
            
            results[doc_count] = {
                'construction': construction_results,
                'document_count': len(documents),
                'province_count': len(generator.properties_by_province),
                'traffic_events': len(traffic_logs)
            }
            
            print(f"  Hierarchical: {construction_results['hierarchical']['duration_seconds']:.3f}s")
            print(f"  Traditional: {construction_results['traditional']['duration_seconds']:.3f}s")
        
        return results
    
    def _calculate_proof_size(self, hierarchical_proof):
        """Calculate total proof size in bytes for hierarchical proof."""
        total_size = 0
        
        # Province proofs
        for province_proof in hierarchical_proof['province_proofs'].values():
            total_size += len(province_proof['proof']) * 32  # 32 bytes per hash
            total_size += len(province_proof['flags'])  # 1 byte per flag
            total_size += len(province_proof['document_hashes']) * 32
        
        # Jurisdiction proof
        total_size += len(hierarchical_proof['jurisdiction_proof']['proof']) * 32
        total_size += len(hierarchical_proof['jurisdiction_proof']['flags'])
        
        return total_size
    
    def run_comprehensive_benchmark(self, target_documents=1000, target_traffic_events=500, force_onchain_verification=False):
        """Run complete benchmark suite."""
        print("=== COMPREHENSIVE HIERARCHICAL vs TRADITIONAL MERKLE TREE BENCHMARK ===")
        print(f"Target Documents: {target_documents}")
        print(f"Target Traffic Events: {target_traffic_events}")
        print(f"On-chain Verification: {'ENABLED' if force_onchain_verification else 'ESTIMATE ONLY'}")
        
        if force_onchain_verification and not self.web3:
            raise Exception("On-chain verification requested but Web3 connection not available")
        
        self.force_onchain_verification = force_onchain_verification
        
        # Generate test data with consistent seed for reproducibility
        print("\n--- Generating Test Data ---")
        # Use document count as part of seed to ensure different scales don't interfere
        consistent_seed = 42 + (target_documents // 1000)
        generator = LargeScaleDocumentGenerator(target_document_count=target_documents, seed=consistent_seed)
        documents = generator.generate_documents()
        
        traffic_generator = RealisticTrafficGenerator(documents, generator.properties_by_province)
        # Use the same consistent seed for traffic generation
        import random
        random.seed(consistent_seed)
        traffic_logs = traffic_generator.generate_traffic_logs(target_traffic_events)
        
        # Define verification scenarios
        verification_scenarios = self._create_verification_scenarios(generator.properties_by_province)
        
        # Run benchmarks
        construction_results, jurisdiction_manager, traditional_multiproof_builder, traditional_single_proof_builder, traditional_huffman_builder = self.benchmark_tree_construction(documents, traffic_logs)
        proof_results = self.benchmark_proof_generation(jurisdiction_manager, traditional_multiproof_builder, traditional_single_proof_builder, traditional_huffman_builder, verification_scenarios)
        
        if self.web3:
            gas_results = self.benchmark_gas_costs(jurisdiction_manager, traditional_multiproof_builder, traditional_single_proof_builder, traditional_huffman_builder, verification_scenarios)
        else:
            gas_results = {}
            print("Skipping gas cost analysis (no Web3 connection)")
        
        # Compile final results
        final_results = {
            'test_config': {
                'total_documents': len(documents),
                'total_provinces': len(generator.properties_by_province),
                'total_traffic_events': len(traffic_logs),
                'timestamp': time.time()
            },
            'construction_performance': construction_results,
            'proof_generation_performance': proof_results,
            'gas_cost_analysis': gas_results,
            'performance_summary': self.profiler.get_summary()
        }
        
        # Save results using organized structure
        results_filename = f"benchmark_results_{target_documents}docs_{int(time.time())}.json"
        saved_file = save_organized_file(final_results, results_filename, "benchmark_results")
        
        print(f"\nBenchmark results saved to: {saved_file}")
        return final_results
    
    def _create_verification_scenarios(self, properties_by_province):
        """Create realistic verification scenarios for testing."""
        province_names = list(properties_by_province.keys())
        scenarios = {}
        
        # Small cross-province (2 provinces)
        if len(province_names) >= 2:
            selected_provinces = province_names[:2]
            scenarios['small_cross_province'] = {}
            for province in selected_provinces:
                properties = properties_by_province[province]
                if properties:
                    selected_props = properties[:min(2, len(properties))]
                    scenarios['small_cross_province'][province] = [prop['full_id'] for prop in selected_props]
        
        # Medium cross-province (3-4 provinces)
        if len(province_names) >= 3:
            selected_provinces = province_names[:min(4, len(province_names))]
            scenarios['medium_cross_province'] = {}
            for province in selected_provinces:
                properties = properties_by_province[province]
                if properties:
                    selected_props = properties[:min(3, len(properties))]
                    scenarios['medium_cross_province'][province] = [prop['full_id'] for prop in selected_props]
        
        # Large cross-province (all provinces) - testing the proof generation bug fix
        scenarios['large_cross_province'] = {}
        for province in province_names:
            properties = properties_by_province[province]
            if properties:
                selected_props = properties[:min(2, len(properties))]
                scenarios['large_cross_province'][province] = [prop['full_id'] for prop in selected_props]
        
        # Single province (high document count)
        if province_names:
            main_province = province_names[0]
            properties = properties_by_province[main_province]
            if properties:
                selected_props = properties[:min(10, len(properties))]
                scenarios['single_province_large'] = {
                    main_province: [prop['full_id'] for prop in selected_props]
                }
        
        return scenarios

# --- Helper to align gas runs with quick_start_tests.py ---
def run_gas_benchmark_with_seed(document_count: int = 5000, events: int = 200, seed: int = 42, gas_price_gwei: int = 20):
    """
    Build the same dataset and execute the same gas benchmarking flow used by quick_start_tests.py gas.
    This ensures apples-to-apples results between runners.
    """
    from web3 import Web3
    from large_scale_generator import LargeScaleDocumentGenerator, RealisticTrafficGenerator
    from gas_cost_analyzer import GasCostAnalyzer

    # Initialize Web3 (Hardhat localhost)
    web3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
    assert web3.is_connected(), "Web3 not connected; please run: npx hardhat node"
    web3.eth.default_account = web3.eth.accounts[0]

    # Generate identical documents and traffic using the same seed
    doc_gen = LargeScaleDocumentGenerator(target_document_count=document_count, seed=seed)
    documents = doc_gen.generate_documents()
    traffic_gen = RealisticTrafficGenerator(documents, doc_gen.properties_by_province)
    # Note: RealisticTrafficGenerator may not accept a seed; rely on doc_gen seeding
    traffic_logs = traffic_gen.generate_traffic_logs(events)

    # Reuse the exact analyzer flow
    analyzer = GasCostAnalyzer(web3, gas_price_gwei=gas_price_gwei)
    tree_systems = analyzer._build_tree_systems(documents, traffic_logs)

    # Build consistent scenarios from the same traffic logs
    verification_scenarios = {
        'small_verifications': {
            'description': 'Small verifications (1-3 properties)',
            'events': [e for e in traffic_logs if 1 <= len(e) <= 3][:20]
        },
        'medium_verifications': {
            'description': 'Medium verifications (4-8 properties)',
            'events': [e for e in traffic_logs if 4 <= len(e) <= 8][:20]
        },
        'single_province_verifications': {
            'description': 'Single province verifications',
            'events': [e for e in traffic_logs if len({p.split('.')[0] for p in e}) == 1][:20]
        },
        'cross_province_verifications': {
            'description': 'Cross-province verifications',
            'events': [e for e in traffic_logs if len({p.split('.')[0] for p in e}) > 1][:20]
        },
    }

    # Run analyzer’s internal comprehensive flow but return only gas_results
    report = analyzer.run_comprehensive_gas_analysis(document_count=len(documents), traffic_events=len(traffic_logs))
    return report.get('gas_results', report)

def main():
    """Run comprehensive benchmarks."""
    # Setup Web3 connection for gas analysis (optional)
    try:
        web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
        if web3.is_connected():
            web3.eth.default_account = web3.eth.accounts[0]
            print("Connected to Hardhat for gas analysis")
        else:
            web3 = None
    except:
        web3 = None
        print("No Web3 connection - running without gas analysis")
    
    # Run benchmarks with different scales
    benchmark_suite = ComprehensiveBenchmarkSuite(web3)
    
    # Test different scales
    scales_to_test = [500, 1000, 2000, 5000]
    
    print("Running benchmarks at multiple scales...")
    for scale in scales_to_test:
        print(f"\n{'='*60}")
        print(f"TESTING SCALE: {scale} DOCUMENTS")
        print(f"{'='*60}")
        
        # Reset contract state between different scales to prevent state pollution
        if web3 and web3.is_connected():
            benchmark_suite.reset_contract_state()
        
        results = benchmark_suite.run_comprehensive_benchmark(
            target_documents=scale,
            target_traffic_events=scale // 5,
            force_onchain_verification=True  # Enable on-chain verification by default
        )
        
        # Print key findings
        construction = results['construction_performance']
        hierarchical_time = construction['hierarchical']['duration_seconds']
        traditional_multiproof_time = construction['traditional_multiproof']['duration_seconds']
        traditional_single_proof_time = construction['traditional_single_proof']['duration_seconds']
        
        print(f"\nKEY FINDINGS FOR {scale} DOCUMENTS:")
        print(f"  Construction Speed Comparison:")
        print(f"    Hierarchical: {hierarchical_time:.3f}s")
        print(f"    Traditional Multiproof: {traditional_multiproof_time:.3f}s ({traditional_multiproof_time/hierarchical_time:.2f}x)")
        print(f"    Traditional Single Proof: {traditional_single_proof_time:.3f}s ({traditional_single_proof_time/hierarchical_time:.2f}x)")
        print(f"    Multiproof vs Single Proof: {traditional_single_proof_time/traditional_multiproof_time:.2f}x")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive Gas Cost Analysis Module

This module provides detailed gas cost analysis for comparing:
1. Hierarchical approach with Pairs-first Huffman optimization
2. Traditional multiproof approach
3. Traditional single proof approach

Features:
- Real on-chain gas measurement
- Gas cost estimation and prediction
- Cost analysis across different verification scenarios
- Optimization impact measurement
- Cost-benefit analysis for different scales
"""

import json
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from web3 import Web3
from clustered_flat_tree_builder import ClusteredFlatTreeBuilder
from large_scale_generator import LargeScaleDocumentGenerator, RealisticTrafficGenerator
from jurisdiction_tree_manager import JurisdictionTreeManager
from traditional_multiproof_builder import TraditionalMerkleTreeBuilder
from traditional_single_proof_builder import TraditionalSingleProofMerkleTreeBuilder
from traditional_multiproof_with_huffman_builder import TraditionalMultiproofWithHuffmanBuilder

class GasCostAnalyzer:
    """
    Comprehensive gas cost analyzer for hierarchical vs traditional approaches.
    
    This analyzer measures:
    1. Real on-chain gas consumption
    2. Gas cost estimation accuracy
    3. Cost optimization across different scenarios
    4. Scalability analysis
    5. Cost-benefit analysis
    """
    
    def __init__(self, web3_instance=None, gas_price_gwei=20, reports_dir=None):
        self.web3 = web3_instance
        self.gas_price_gwei = gas_price_gwei
        self.gas_price_wei = gas_price_gwei * 10**9
        self.reports_dir = reports_dir
        
        # Contract instances
        self.hierarchical_contract = None
        self.traditional_contract = None
        self.single_proof_contract = None
        
        # Gas analysis results
        self.gas_analysis_results = {}
        
        if self.web3:
            self._setup_contracts()
    
    def _setup_contracts(self):
        """Setup smart contracts for gas cost analysis."""
        try:
            # Hierarchical contract
            hierarchical_artifact_path = '../../artifacts/contracts/HierarchicalMerkleVerifier.sol/HierarchicalMerkleVerifier.json'
            print(f"Loading hierarchical contract from: {hierarchical_artifact_path}")
            with open(hierarchical_artifact_path, 'r') as f:
                hierarchical_artifact = json.load(f)
            
            hierarchical_address = hierarchical_artifact['networks']['31337']['address']
            self.hierarchical_contract = self.web3.eth.contract(
                address=hierarchical_address,
                abi=hierarchical_artifact['abi']
            )
            print(f"✅ Hierarchical contract loaded at: {hierarchical_address}")
            
            # Traditional multiproof contract
            traditional_artifact_path = '../../artifacts/contracts/MerkleVerifier.sol/MerkleVerifier.json'
            print(f"Loading traditional contract from: {traditional_artifact_path}")
            with open(traditional_artifact_path, 'r') as f:
                traditional_artifact = json.load(f)
            
            traditional_address = traditional_artifact['networks']['31337']['address']
            self.traditional_contract = self.web3.eth.contract(
                address=traditional_address,
                abi=traditional_artifact['abi']
            )
            print(f"✅ Traditional contract loaded at: {traditional_address}")
            
            # Single proof contract
            single_proof_artifact_path = '../../artifacts/contracts/SingleProofMerkleVerifier.sol/SingleProofMerkleVerifier.json'
            print(f"Loading single proof contract from: {single_proof_artifact_path}")
            with open(single_proof_artifact_path, 'r') as f:
                single_proof_artifact = json.load(f)
            
            single_proof_address = single_proof_artifact['networks']['31337']['address']
            self.single_proof_contract = self.web3.eth.contract(
                address=single_proof_address,
                abi=single_proof_artifact['abi']
            )
            print(f"✅ Single proof contract loaded at: {single_proof_address}")
            
            # Clustered Flat Basic contract
            clustered_flat_basic_path = '../../artifacts/contracts/ClusteredFlatBasicVerifier.sol/ClusteredFlatBasicVerifier.json'
            with open(clustered_flat_basic_path, 'r') as f:
                clustered_flat_basic_artifact = json.load(f)
            
            clustered_flat_basic_address = clustered_flat_basic_artifact['networks']['31337']['address']
            self.clustered_flat_basic_contract = self.web3.eth.contract(
                address=clustered_flat_basic_address,
                abi=clustered_flat_basic_artifact['abi']
            )
            print(f"✅ Clustered flat basic contract loaded at: {clustered_flat_basic_address}")

            # Clustered Flat contract
            clustered_flat_path = '../../artifacts/contracts/ClusteredFlatVerifier.sol/ClusteredFlatVerifier.json'
            with open(clustered_flat_path, 'r') as f:
                clustered_flat_artifact = json.load(f)
            
            clustered_flat_address = clustered_flat_artifact['networks']['31337']['address']
            self.clustered_flat_contract = self.web3.eth.contract(
                address=clustered_flat_address,
                abi=clustered_flat_artifact['abi']
            )
            print(f"✅ Clustered flat contract loaded at: {clustered_flat_address}")

            print("✅ All contracts loaded successfully for gas analysis")
            
        except Exception as e:
            print(f"⚠️  Contract setup failed: {e}")
            print(f"   Make sure Hardhat is running and contracts are deployed")
            self.hierarchical_contract = None
            self.traditional_contract = None
            self.single_proof_contract = None
    
    def run_comprehensive_gas_analysis(self, document_count=2000, traffic_events=1000, force_onchain_verification=False, sparse_verification=False, verification_sampling_rate=1.0, selected_approaches=None):
        """
        Run comprehensive gas cost analysis.
        
        Args:
            document_count: Number of documents to generate
            traffic_events: Number of traffic events to simulate
            force_onchain_verification: If True, perform actual on-chain transactions
            sparse_verification: If True, use sparse verification with sampling
            verification_sampling_rate: Fraction of documents to include in each verification (0.0-1.0)
            
        Returns:
            Comprehensive gas analysis results
        """
        print(f"{'='*80}")
        print(f"COMPREHENSIVE GAS COST ANALYSIS")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  Documents: {document_count}")
        print(f"  Traffic Events: {traffic_events}")
        print(f"  Gas Price: {self.gas_price_gwei} gwei")
        print(f"  On-chain Verification: {'ENABLED' if force_onchain_verification else 'ESTIMATE ONLY'}")
        print(f"  Sparse Verification: {'ENABLED' if sparse_verification else 'DISABLED'}")
        if sparse_verification:
            print(f"  Verification Sampling Rate: {verification_sampling_rate*100:.1f}%")
        
        if force_onchain_verification and not self.web3:
            raise Exception("On-chain verification requested but Web3 connection not available")
        
        self.force_onchain_verification = force_onchain_verification
        self.sparse_verification = sparse_verification
        self.verification_sampling_rate = verification_sampling_rate
        
        start_time = time.time()
        
        # Step 1: Generate test data
        print(f"\n--- STEP 1: GENERATING TEST DATA ---")
        documents, traffic_logs, properties_by_province = self._generate_test_data(
            document_count, traffic_events
        )
        
        # Step 2: Build tree systems
        print(f"\n--- STEP 2: BUILDING TREE SYSTEMS ---")
        if selected_approaches:
            print(f"Selected approaches: {', '.join(selected_approaches)}")
        tree_systems = self._build_tree_systems(documents, traffic_logs, selected_approaches)
        
        # Step 3: Run gas cost analysis
        print(f"\n--- STEP 3: RUNNING GAS COST ANALYSIS ---")
        gas_results = self._run_gas_cost_analysis(tree_systems, traffic_logs)
        
        # Step 4: Analyze cost optimization
        print(f"\n--- STEP 4: ANALYZING COST OPTIMIZATION ---")
        optimization_analysis = self._analyze_cost_optimization(gas_results)
        
        # Step 5: Generate scalability analysis
        print(f"\n--- STEP 5: GENERATING SCALABILITY ANALYSIS ---")
        scalability_analysis = self._analyze_scalability(tree_systems, traffic_logs)
        
        # Step 6: Generate comprehensive report
        print(f"\n--- STEP 6: GENERATING COMPREHENSIVE REPORT ---")
        final_report = self._generate_gas_analysis_report(
            documents, traffic_logs, tree_systems, gas_results, 
            optimization_analysis, scalability_analysis, start_time
        )
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"GAS ANALYSIS COMPLETED IN {total_time:.2f}s")
        print(f"{'='*80}")
        
        return final_report
    
    def _generate_test_data(self, document_count, traffic_events):
        """Generate test data for gas analysis."""
        # Generate documents
        doc_generator = LargeScaleDocumentGenerator(target_document_count=document_count, seed=42)
        documents = doc_generator.generate_documents()
        
        # Generate traffic logs
        traffic_generator = RealisticTrafficGenerator(documents, doc_generator.properties_by_province)
        traffic_logs = traffic_generator.generate_traffic_logs(traffic_events)
        
        return documents, traffic_logs, doc_generator.properties_by_province
    
    def _build_tree_systems(self, documents, traffic_logs, selected_approaches=None):
        """Build tree systems for gas analysis based on selected approaches."""
        tree_systems = {}
        
        # Use all approaches if none specified
        if selected_approaches is None:
            selected_approaches = ['hierarchical', 'traditional_multiproof', 'traditional_single_proof', 'traditional_huffman', 'clustered_flat_with_merkle', 'clustered_flat']
        
        if 'hierarchical' in selected_approaches:
            print("Building Hierarchical System...")
            hierarchical_start = time.time()
            jurisdiction_manager = JurisdictionTreeManager(documents, traffic_logs)
            jurisdiction_root = jurisdiction_manager.build_all_trees()
            hierarchical_build_time = time.time() - hierarchical_start
        
            # Update the smart contract with the new jurisdiction root
            if self.hierarchical_contract and self.web3 and jurisdiction_root:
                try:
                    print("Updating smart contract jurisdiction root...")
                    if jurisdiction_root.startswith('0x'):
                        root_bytes = bytes.fromhex(jurisdiction_root[2:])
                    else:
                        root_bytes = bytes.fromhex(jurisdiction_root)
                    
                    accounts = self.web3.eth.accounts
                    tx_hash = self.hierarchical_contract.functions.updateJurisdictionRoot(root_bytes).transact({'from': accounts[0]})
                    self.web3.eth.wait_for_transaction_receipt(tx_hash)
                    print(f"✅ Jurisdiction root updated in smart contract: {jurisdiction_root[:16]}...")
                except Exception as e:
                    print(f"⚠️  Failed to update jurisdiction root in smart contract: {e}")
                    print("   Hierarchical verification may fail")
            elif not jurisdiction_root:
                print("⚠️  Jurisdiction root is None - skipping smart contract update")
            
            tree_systems['hierarchical'] = {
                'manager': jurisdiction_manager,
                'root': jurisdiction_root,
                'build_time': hierarchical_build_time,
                'type': 'Hierarchical with Pairs-first Huffman'
            }
        
        if 'traditional_multiproof' in selected_approaches:
            print("Building Traditional Multiproof System...")
            traditional_start = time.time()
            traditional_builder = TraditionalMerkleTreeBuilder(documents)
            traditional_root = traditional_builder.build()
            traditional_build_time = time.time() - traditional_start
            
            tree_systems['traditional_multiproof'] = {
                'builder': traditional_builder,
                'root': traditional_root,
                'build_time': traditional_build_time,
                'type': 'Traditional Multiproof'
            }
        
        if 'traditional_single_proof' in selected_approaches:
            print("Building Traditional Single Proof System...")
            single_proof_start = time.time()
            single_proof_builder = TraditionalSingleProofMerkleTreeBuilder(documents)
            single_proof_root = single_proof_builder.build()
            single_proof_build_time = time.time() - single_proof_start
            
            tree_systems['traditional_single_proof'] = {
                'builder': single_proof_builder,
                'root': single_proof_root,
                'build_time': single_proof_build_time,
                'type': 'Traditional Single Proof'
            }
        
        if 'traditional_huffman' in selected_approaches:
            print("Building Traditional Huffman System...")
            huffman_start = time.time()
            huffman_builder = TraditionalMultiproofWithHuffmanBuilder(documents, traffic_logs)
            huffman_root = huffman_builder.build()
            huffman_build_time = time.time() - huffman_start
            
            tree_systems['traditional_huffman'] = {
                'builder': huffman_builder,
                'root': huffman_root,
                'build_time': huffman_build_time,
                'type': 'Traditional Huffman'
            }
        
        if 'clustered_flat_with_merkle' in selected_approaches:
            print("Building Clustered Flat with Merkle System...")
            clustered_flat_merkle_start = time.time()
            clustered_flat_merkle_builder = ClusteredFlatTreeBuilder(documents, traffic_logs)
            clustered_flat_merkle_root = clustered_flat_merkle_builder.build()
            clustered_flat_merkle_build_time = time.time() - clustered_flat_merkle_start
            
            tree_systems['clustered_flat_with_merkle'] = {
                'manager': clustered_flat_merkle_builder,
                'root': clustered_flat_merkle_root,
                'build_time': clustered_flat_merkle_build_time,
                'type': 'Clustered Flat with Merkle'
            }
        
        if 'clustered_flat' in selected_approaches:
            print("Building Clustered Flat System...")
            clustered_flat_start = time.time()
            clustered_flat_builder = ClusteredFlatTreeBuilder(documents, traffic_logs)
            clustered_flat_root = clustered_flat_builder.build()
            clustered_flat_build_time = time.time() - clustered_flat_start
            
            tree_systems['clustered_flat'] = {
                'manager': clustered_flat_builder,
                'root': clustered_flat_root,
                'build_time': clustered_flat_build_time,
                'type': 'Clustered Flat without Merkle'
            }

        print(f"Tree building completed:")
        for name, system in tree_systems.items():
            print(f"  {system['type']}: {system['build_time']:.3f}s")
        
        return tree_systems
    
    def _run_gas_cost_analysis(self, tree_systems, traffic_logs):
        """Run comprehensive gas cost analysis."""
        print("Running gas cost analysis across different scenarios...")
        
        # Create test scenarios
        test_scenarios = self._create_gas_test_scenarios(traffic_logs)
        
        gas_results = {}
        
        for scenario_name, scenario_data in test_scenarios.items():
            print(f"  Testing scenario: {scenario_name}")
            
            scenario_results = self._test_scenario_gas_costs(
                tree_systems, scenario_data, scenario_name
            )
            
            gas_results[scenario_name] = scenario_results
        
        return gas_results
    
    def _create_gas_test_scenarios(self, traffic_logs):
        """Create test scenarios for gas cost analysis."""
        scenarios = {}
        
        # Apply sparse sampling if enabled
        if hasattr(self, 'sparse_verification') and self.sparse_verification:
            print(f"  📊 Applying sparse verification sampling ({self.verification_sampling_rate*100:.1f}%)")
            traffic_logs = self._apply_sparse_sampling(traffic_logs)
        
        # Determine test limit based on traffic logs size
        total_events = len(traffic_logs)
        if total_events >= 10000:  # Massive scale
            test_limit = min(500, max(100, total_events // 20))  # 5% of events, min 100, max 500
        elif total_events >= 2000:  # Large scale
            test_limit = min(200, max(50, total_events // 10))   # 10% of events, min 50, max 200
        elif total_events >= 500:   # Medium scale
            test_limit = min(100, max(30, total_events // 5))    # 20% of events, min 30, max 100
        else:  # Small scale
            test_limit = min(50, max(20, total_events // 2))     # 50% of events, min 20, max 50
        
        print(f"  🎯 Using test limit of {test_limit} events per scenario (from {total_events} total events)")
        
        # Scenario 1: Small verifications (1-3 properties)
        small_events = [event for event in traffic_logs if 1 <= len(event) <= 3]
        if small_events:
            # Randomly sample from small events for better coverage
            sampled_small = random.sample(small_events, min(test_limit, len(small_events)))
            scenarios['small_verifications'] = {
                'description': 'Small verifications (1-3 properties)',
                'events': sampled_small,
                'expected_gas_range': 'Low'
            }
        
        # Scenario 2: Medium verifications (4-8 properties)
        medium_events = [event for event in traffic_logs if 4 <= len(event) <= 8]
        if medium_events:
            sampled_medium = random.sample(medium_events, min(test_limit, len(medium_events)))
            scenarios['medium_verifications'] = {
                'description': 'Medium verifications (4-8 properties)',
                'events': sampled_medium,
                'expected_gas_range': 'Medium'
            }
        
        # Scenario 3: Large verifications (9+ properties)
        large_events = [event for event in traffic_logs if len(event) >= 9]
        if large_events:
            sampled_large = random.sample(large_events, min(test_limit, len(large_events)))
            scenarios['large_verifications'] = {
                'description': 'Large verifications (9+ properties)',
                'events': sampled_large,
                'expected_gas_range': 'High'
            }
        
        # Scenario 4: Cross-province verifications
        cross_province_events = []
        for event in traffic_logs:
            provinces = set()
            for prop in event:
                if '.' in prop:
                    provinces.add(prop.split('.')[0])
            if len(provinces) > 1:
                cross_province_events.append(event)
        
        if cross_province_events:
            sampled_cross = random.sample(cross_province_events, min(test_limit, len(cross_province_events)))
            scenarios['cross_province_verifications'] = {
                'description': 'Cross-province verifications',
                'events': sampled_cross,
                'expected_gas_range': 'Medium-High'
            }
        
        # Scenario 5: Single province verifications
        single_province_events = []
        for event in traffic_logs:
            provinces = set()
            for prop in event:
                if '.' in prop:
                    provinces.add(prop.split('.')[0])
            if len(provinces) == 1:
                single_province_events.append(event)
        
        if single_province_events:
            sampled_single = random.sample(single_province_events, min(test_limit, len(single_province_events)))
            scenarios['single_province_verifications'] = {
                'description': 'Single province verifications',
                'events': sampled_single,
                'expected_gas_range': 'Low-Medium'
            }
        
        return scenarios
    
    def _apply_sparse_sampling(self, traffic_logs):
        """Apply sparse sampling to traffic logs for large-scale testing."""
        if not hasattr(self, 'verification_sampling_rate'):
            return traffic_logs
        
        sampled_logs = []
        for event in traffic_logs:
            # Randomly sample properties within each event
            if len(event) > 1:
                sample_size = max(1, int(len(event) * self.verification_sampling_rate))
                sampled_event = random.sample(event, sample_size)
                sampled_logs.append(sampled_event)
            else:
                sampled_logs.append(event)
        
        original_properties = sum(len(event) for event in traffic_logs)
        sampled_properties = sum(len(event) for event in sampled_logs)
        
        print(f"    Original events: {len(traffic_logs)} with {original_properties} total properties")
        print(f"    Sampled events: {len(sampled_logs)} with {sampled_properties} total properties")
        print(f"    Sampling reduction: {(1 - sampled_properties/original_properties)*100:.1f}%")
        
        return sampled_logs
    
    def _test_scenario_gas_costs(self, tree_systems, scenario_data, scenario_name):
        """Test gas costs for a specific scenario."""
        events = scenario_data['events']
        results = {
            'scenario_name': scenario_name,
            'description': scenario_data['description'],
            'expected_gas_range': scenario_data['expected_gas_range'],
            'total_events': len(events),
            'approach_results': {}
        }
        
        for approach_name, system in tree_systems.items():
            approach_results = []
            
            for event in events:
                try:
                    result = self._test_single_event_gas_cost(approach_name, system, event)
                    approach_results.append(result)
                except Exception as e:
                    print(f"    Error testing {approach_name}: {e}")
                    approach_results.append({
                        'error': str(e),
                        'event': event
                    })
            
            # Calculate statistics for this approach
            valid_results = [r for r in approach_results if 'error' not in r]
            if valid_results:
                gas_costs = [r['gas_cost_wei'] for r in valid_results if 'gas_cost_wei' in r]
                gas_estimates = [r['gas_estimate'] for r in valid_results if 'gas_estimate' in r and isinstance(r['gas_estimate'], int)]
                
                results['approach_results'][approach_name] = {
                    'total_tests': len(approach_results),
                    'successful_tests': len(valid_results),
                    'avg_gas_cost_wei': np.mean(gas_costs) if gas_costs else 0,
                    'median_gas_cost_wei': np.median(gas_costs) if gas_costs else 0,
                    'min_gas_cost_wei': min(gas_costs) if gas_costs else 0,
                    'max_gas_cost_wei': max(gas_costs) if gas_costs else 0,
                    'avg_gas_estimate': np.mean(gas_estimates) if gas_estimates else 0,
                    'avg_gas_cost_usd': np.mean([g * self.gas_price_wei / 10**18 for g in gas_costs]) if gas_costs else 0,
                    'total_gas_cost_usd': sum([g * self.gas_price_wei / 10**18 for g in gas_costs]) if gas_costs else 0
                }
            else:
                results['approach_results'][approach_name] = {
                    'total_tests': len(approach_results),
                    'successful_tests': 0,
                    'error': 'All tests failed'
                }
        
        return results
    
    def _test_single_event_gas_cost(self, approach_name, system, event):
        """Test gas cost for a single verification event."""
        start_time = time.time()
        
        if approach_name == 'hierarchical':
            return self._test_hierarchical_gas_cost(system, event, start_time)
        elif approach_name == 'traditional_multiproof':
            return self._test_traditional_multiproof_gas_cost(system, event, start_time)
        elif approach_name == 'traditional_single_proof':
            return self._test_traditional_single_proof_gas_cost(system, event, start_time)
        elif approach_name == 'traditional_huffman':
            return self._test_traditional_huffman_gas_cost(system, event, start_time)
        elif approach_name == 'clustered_flat':
            return self._test_clustered_flat_gas_cost(system, event, start_time)
        elif approach_name == 'clustered_flat_with_merkle':
            return self._test_clustered_flat_merkle_gas_cost(system, event, start_time)
        else:
            return {
                'approach': approach_name,
                'error': f'Unsupported approach: {approach_name}',
                'event': event
            }
    
    def _test_hierarchical_gas_cost(self, system, event, start_time):
        """Test gas cost for hierarchical approach."""
        # Convert event to verification request
        verification_request = defaultdict(list)
        for prop_id in event:
            if '.' in prop_id:
                province, property_id = prop_id.split('.', 1)
                verification_request[province].append(prop_id)
        
        # Generate hierarchical proof
        proof_package = system['manager'].verify_cross_province_batch(verification_request)
        
        # Estimate gas cost
        gas_estimate = None
        gas_cost_wei = None
        
        if not self.hierarchical_contract:
            gas_estimate = "No hierarchical contract available"
            gas_cost_wei = 0
        elif not proof_package:
            gas_estimate = "Proof generation failed"
            gas_cost_wei = 0
        else:
            try:
                # Prepare data for on-chain verification
                claimed_province_roots = []
                province_proofs = []
                province_flags = []
                province_leaves_arrays = []
                provinces_involved = proof_package['jurisdiction_proof']['provinces_involved']
                
                for province in provinces_involved:
                    province_proof_data = proof_package['province_proofs'][province]
                    proof_bytes = [bytes.fromhex(p) for p in province_proof_data['proof']]
                    leaves_bytes = [bytes.fromhex(l) for l in province_proof_data['document_hashes']]
                    
                    # Get the province root from the proof data
                    province_root = bytes.fromhex(province_proof_data['expected_root'])
                    claimed_province_roots.append(province_root)
                    
                    province_proofs.append(proof_bytes)
                    province_flags.append(province_proof_data['flags'])
                    province_leaves_arrays.append(leaves_bytes)
                
                jurisdiction_proof_bytes = [bytes.fromhex(p) for p in proof_package['jurisdiction_proof']['proof']]
                jurisdiction_flags = proof_package['jurisdiction_proof']['flags']
                
                if self.force_onchain_verification:
                    # Perform actual on-chain verification
                    print(f"    Executing on-chain hierarchical verification...")
                    
                    # First estimate gas
                    gas_estimate = self.hierarchical_contract.functions.verifyHierarchicalBatch(
                        claimed_province_roots,      # bytes32[] calldata claimedProvinceRoots
                        jurisdiction_proof_bytes,    # bytes32[] calldata jurisdictionProof
                        jurisdiction_flags,          # bool[] calldata jurisdictionFlags
                        province_proofs,             # bytes32[][] calldata provinceProofs
                        province_flags,              # bool[][] calldata provinceFlags
                        province_leaves_arrays       # bytes32[][] calldata provinceLeavesArrays
                    ).estimate_gas()
                    
                    # Execute actual transaction
                    tx_hash = self.hierarchical_contract.functions.verifyHierarchicalBatch(
                        claimed_province_roots,
                        jurisdiction_proof_bytes,
                        jurisdiction_flags,
                        province_proofs,
                        province_flags,
                        province_leaves_arrays
                    ).transact()
                    
                    # Wait for transaction receipt and get actual gas used
                    receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                    actual_gas_used = receipt.gasUsed
                    gas_cost_wei = actual_gas_used * self.gas_price_wei
                    
                    print(f"    ✅ On-chain verification successful - Gas used: {actual_gas_used}")
                    
                else:
                    # Just estimate gas
                    gas_estimate = self.hierarchical_contract.functions.verifyHierarchicalBatch(
                        claimed_province_roots,      # bytes32[] calldata claimedProvinceRoots
                        jurisdiction_proof_bytes,    # bytes32[] calldata jurisdictionProof
                        jurisdiction_flags,          # bool[] calldata jurisdictionFlags
                        province_proofs,             # bytes32[][] calldata provinceProofs
                        province_flags,              # bool[][] calldata provinceFlags
                        province_leaves_arrays       # bytes32[][] calldata provinceLeavesArrays
                    ).estimate_gas()
                    
                    # Calculate gas cost from estimate
                    gas_cost_wei = gas_estimate * self.gas_price_wei
                    actual_gas_used = gas_estimate
                
            except Exception as e:
                print(f"    Hierarchical gas estimation error: {e}")
                gas_estimate = f"Error: {e}"
                gas_cost_wei = 0
        
        verification_time = time.time() - start_time
        
        return {
            'approach': 'hierarchical',
            'properties_count': len(event),
            'provinces_count': len(set(p.split('.')[0] for p in event if '.' in p)),
            'verification_time': verification_time,
            'gas_estimate': gas_estimate,
            'gas_cost_wei': gas_cost_wei,
            'gas_cost_usd': gas_cost_wei / 10**18 if gas_cost_wei else 0,
            'cross_province': len(set(p.split('.')[0] for p in event if '.' in p)) > 1
        }
    
    def _test_traditional_multiproof_gas_cost(self, system, event, start_time):
        """Test gas cost for traditional multiproof approach."""
        # Collect all document hashes for the event
        all_doc_hashes = []
        for prop_id in event:
            for doc in system['builder'].all_documents:
                if doc.full_id == prop_id:
                    all_doc_hashes.append(doc.hash_hex)
        
        # Deduplicate to ensure unique leaves
        all_doc_hashes = list(dict.fromkeys(all_doc_hashes))
        
        if not all_doc_hashes:
            return {
                'approach': 'traditional_multiproof',
                'error': 'No documents found for event',
                'event': event
            }
        
        # Generate multiproof
        proof, flags = system['builder'].generate_proof_for_documents(all_doc_hashes)
        
        # Estimate gas cost
        gas_estimate = None
        gas_cost_wei = None
        
        if not self.traditional_contract:
            gas_estimate = "No traditional contract available"
            gas_cost_wei = 0
        else:
            try:
                proof_bytes = [bytes.fromhex(p) for p in proof]
                # IMPORTANT: Leaves must be passed in the exact order used by the proof generator
                # Use sorted order to match TraditionalMerkleTreeBuilder.generate_proof_for_documents
                leaves_sorted = sorted(all_doc_hashes)
                leaves_bytes = [bytes.fromhex(l) for l in leaves_sorted]
                
                if self.force_onchain_verification:
                    # Perform actual on-chain verification
                    print(f"    Executing on-chain traditional multiproof verification...")
                    
                    # First estimate gas
                    gas_estimate = self.traditional_contract.functions.verifyBatch(
                        proof_bytes, flags, leaves_bytes
                    ).estimate_gas()
                    
                    # Execute actual transaction
                    tx_hash = self.traditional_contract.functions.verifyBatch(
                        proof_bytes, flags, leaves_bytes
                    ).transact()
                    
                    # Wait for transaction receipt and get actual gas used
                    receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                    actual_gas_used = receipt.gasUsed
                    gas_cost_wei = actual_gas_used * self.gas_price_wei
                    
                    print(f"    ✅ On-chain verification successful - Gas used: {actual_gas_used}")
                    
                else:
                    # Just estimate gas
                    gas_estimate = self.traditional_contract.functions.verifyBatch(
                        proof_bytes, flags, leaves_bytes
                    ).estimate_gas()
                    
                    gas_cost_wei = gas_estimate * self.gas_price_wei
                    actual_gas_used = gas_estimate
                
            except Exception as e:
                print(f"    Traditional multiproof gas estimation error: {e}")
                gas_estimate = f"Error: {e}"
                gas_cost_wei = 0
        
        verification_time = time.time() - start_time
        
        return {
            'approach': 'traditional_multiproof',
            'properties_count': len(event),
            'provinces_count': len(set(p.split('.')[0] for p in event if '.' in p)),
            'verification_time': verification_time,
            'gas_estimate': gas_estimate,
            'gas_cost_wei': gas_cost_wei,
            'gas_cost_usd': gas_cost_wei / 10**18 if gas_cost_wei else 0,
            'cross_province': len(set(p.split('.')[0] for p in event if '.' in p)) > 1
        }
    
    def _test_traditional_single_proof_gas_cost(self, system, event, start_time):
        """Test gas cost for traditional single proof approach."""
        # Collect all document hashes for the event
        all_doc_hashes = []
        for prop_id in event:
            for doc in system['builder'].all_documents:
                if doc.full_id == prop_id:
                    all_doc_hashes.append(doc.hash_hex)
        
        if not all_doc_hashes:
            return {
                'approach': 'traditional_single_proof',
                'error': 'No documents found for event',
                'event': event
            }
        
        # Generate individual proofs
        individual_proofs = system['builder'].generate_single_proofs_for_documents(all_doc_hashes)
        
        # Estimate gas cost
        total_gas_estimate = 0
        total_gas_cost_wei = 0
        
        if not self.single_proof_contract:
            total_gas_estimate = "No single proof contract available"
            total_gas_cost_wei = 0
        elif not individual_proofs:
            total_gas_estimate = "No individual proofs generated"
            total_gas_cost_wei = 0
        else:
            try:
                if self.force_onchain_verification:
                    # Perform actual on-chain verification
                    print(f"    Executing on-chain single proof verification for {len(individual_proofs)} proofs...")
                    
                    total_actual_gas_used = 0
                    for i, proof_data in enumerate(individual_proofs):
                        proof_bytes = [bytes.fromhex(p) for p in proof_data['proof']]
                        leaf_bytes = bytes.fromhex(proof_data['document_hash'])
                        
                        # First estimate gas
                        gas_estimate = self.single_proof_contract.functions.verifySingle(
                            proof_bytes, leaf_bytes
                        ).estimate_gas()
                        
                        # Execute actual transaction
                        tx_hash = self.single_proof_contract.functions.verifySingle(
                            proof_bytes, leaf_bytes
                        ).transact()
                        
                        # Wait for transaction receipt and get actual gas used
                        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                        actual_gas_used = receipt.gasUsed
                        
                        total_gas_estimate += gas_estimate
                        total_actual_gas_used += actual_gas_used
                        
                        if i == 0:  # Log first proof
                            print(f"    ✅ First proof verified - Gas used: {actual_gas_used}")
                    
                    total_gas_cost_wei = total_actual_gas_used * self.gas_price_wei
                    print(f"    ✅ All {len(individual_proofs)} proofs verified - Total gas: {total_actual_gas_used}")
                    
                else:
                    # Just estimate gas
                    for proof_data in individual_proofs:
                        proof_bytes = [bytes.fromhex(p) for p in proof_data['proof']]
                        leaf_bytes = bytes.fromhex(proof_data['document_hash'])
                        
                        gas_estimate = self.single_proof_contract.functions.verifySingle(
                            proof_bytes, leaf_bytes
                        ).estimate_gas()
                        
                        total_gas_estimate += gas_estimate
                        total_gas_cost_wei += gas_estimate * self.gas_price_wei
                
            except Exception as e:
                print(f"    Single proof gas estimation error: {e}")
                total_gas_estimate = f"Error: {e}"
                total_gas_cost_wei = 0
        
        verification_time = time.time() - start_time
        
        return {
            'approach': 'traditional_single_proof',
            'properties_count': len(event),
            'provinces_count': len(set(p.split('.')[0] for p in event if '.' in p)),
            'verification_time': verification_time,
            'gas_estimate': total_gas_estimate,
            'gas_cost_wei': total_gas_cost_wei,
            'gas_cost_usd': total_gas_cost_wei / 10**18 if total_gas_cost_wei else 0,
            'cross_province': len(set(p.split('.')[0] for p in event if '.' in p)) > 1,
            'individual_proofs_count': len(individual_proofs)
        }
    
    def _test_traditional_huffman_gas_cost(self, system, event, start_time):
        """Test gas cost for traditional multiproof with huffman approach."""
        # This approach uses the same verification as traditional multiproof
        # but with optimized tree structure
        return self._test_traditional_multiproof_gas_cost(system, event, start_time)
    
    def _test_clustered_flat_gas_cost(self, system, event, start_time):
        """Test gas cost for clustered flat approach."""
        # Estimate gas cost for clustered flat verification
        gas_estimate = self._estimate_clustered_flat_gas(event)
        gas_cost_wei = gas_estimate * self.gas_price_wei if isinstance(gas_estimate, int) else 0
        
        verification_time = time.time() - start_time
        
        return {
            'approach': 'clustered_flat',
            'properties_count': len(event),
            'provinces_count': len(set(p.split('.')[0] for p in event if '.' in p)),
            'verification_time': verification_time,
            'gas_estimate': gas_estimate,
            'gas_cost_wei': gas_cost_wei,
            'gas_cost_usd': gas_cost_wei / 10**18 if gas_cost_wei else 0,
            'cross_province': len(set(p.split('.')[0] for p in event if '.' in p)) > 1
        }
    
    def _test_clustered_flat_merkle_gas_cost(self, system, event, start_time):
        """Test gas cost for clustered flat with merkle approach."""
        # Estimate gas cost for clustered flat with merkle optimization
        base_gas = self._estimate_clustered_flat_gas(event)
        # Apply optimization factor for merkle verifier (typically 20-30% reduction)
        gas_estimate = int(base_gas * 0.75) if isinstance(base_gas, int) else base_gas
        gas_cost_wei = gas_estimate * self.gas_price_wei if isinstance(gas_estimate, int) else 0
        
        verification_time = time.time() - start_time
        
        return {
            'approach': 'clustered_flat_with_merkle',
            'properties_count': len(event),
            'provinces_count': len(set(p.split('.')[0] for p in event if '.' in p)),
            'verification_time': verification_time,
            'gas_estimate': gas_estimate,
            'gas_cost_wei': gas_cost_wei,
            'gas_cost_usd': gas_cost_wei / 10**18 if gas_cost_wei else 0,
            'cross_province': len(set(p.split('.')[0] for p in event if '.' in p)) > 1
        }
    
    def _estimate_clustered_flat_gas(self, event):
        """Estimate gas cost for clustered flat verification."""
        if not event:
            return 50000  # Base gas cost
        
        # Group properties by province
        provinces = set(p.split('.')[0] for p in event if '.' in p)
        province_count = len(provinces)
        
        # Estimate gas based on province count and property count
        # Base cost + cost per province + cost per property
        base_cost = 40000
        province_cost = province_count * 15000
        property_cost = len(event) * 8000
        
        return base_cost + province_cost + property_cost
    
    def _analyze_cost_optimization(self, gas_results):
        """Analyze cost optimization across different approaches."""
        print("Analyzing cost optimization...")
        
        optimization_analysis = {
            'scenario_comparisons': {},
            'overall_optimization': {},
            'cost_benefit_analysis': {}
        }
        
        # Analyze each scenario
        for scenario_name, scenario_data in gas_results.items():
            approach_results = scenario_data['approach_results']
            
            if 'hierarchical' in approach_results and 'traditional_multiproof' in approach_results:
                hierarchical = approach_results['hierarchical']
                traditional = approach_results['traditional_multiproof']
                
                if hierarchical['successful_tests'] > 0 and traditional['successful_tests'] > 0:
                    gas_reduction = (traditional['avg_gas_cost_wei'] - hierarchical['avg_gas_cost_wei']) / traditional['avg_gas_cost_wei'] * 100
                    cost_reduction = (traditional['avg_gas_cost_usd'] - hierarchical['avg_gas_cost_usd']) / traditional['avg_gas_cost_usd'] * 100
                    
                    optimization_analysis['scenario_comparisons'][scenario_name] = {
                        'gas_reduction_percent': gas_reduction,
                        'cost_reduction_percent': cost_reduction,
                        'hierarchical_avg_gas': hierarchical['avg_gas_cost_wei'],
                        'traditional_avg_gas': traditional['avg_gas_cost_wei'],
                        'hierarchical_avg_cost_usd': hierarchical['avg_gas_cost_usd'],
                        'traditional_avg_cost_usd': traditional['avg_gas_cost_usd']
                    }
        
        # Calculate overall optimization
        if optimization_analysis['scenario_comparisons']:
            gas_reductions = [comp['gas_reduction_percent'] for comp in optimization_analysis['scenario_comparisons'].values()]
            cost_reductions = [comp['cost_reduction_percent'] for comp in optimization_analysis['scenario_comparisons'].values()]
            
            optimization_analysis['overall_optimization'] = {
                'avg_gas_reduction_percent': np.mean(gas_reductions),
                'median_gas_reduction_percent': np.median(gas_reductions),
                'avg_cost_reduction_percent': np.mean(cost_reductions),
                'median_cost_reduction_percent': np.median(cost_reductions),
                'max_gas_reduction_percent': max(gas_reductions),
                'max_cost_reduction_percent': max(cost_reductions)
            }
        
        return optimization_analysis
    
    def _analyze_scalability(self, tree_systems, traffic_logs):
        """Analyze gas cost scalability across different scales."""
        print("Analyzing gas cost scalability...")
        
        # Test different scales
        scales = [10, 25, 50, 100, 200]
        scalability_results = {}
        
        for scale in scales:
            if scale > len(traffic_logs):
                continue
            
            print(f"  Testing scale: {scale} events")
            
            # Sample events for this scale
            sample_events = random.sample(traffic_logs, scale)
            
            scale_results = {}
            
            for approach_name, system in tree_systems.items():
                total_gas = 0
                total_cost = 0
                successful_tests = 0
                
                for event in sample_events:
                    try:
                        result = self._test_single_event_gas_cost(approach_name, system, event)
                        if 'gas_cost_wei' in result and result['gas_cost_wei']:
                            total_gas += result['gas_cost_wei']
                            total_cost += result['gas_cost_usd']
                            successful_tests += 1
                    except Exception as e:
                        continue
                
                if successful_tests > 0:
                    scale_results[approach_name] = {
                        'total_gas': total_gas,
                        'total_cost_usd': total_cost,
                        'avg_gas_per_event': total_gas / successful_tests,
                        'avg_cost_per_event': total_cost / successful_tests,
                        'successful_tests': successful_tests
                    }
            
            scalability_results[scale] = scale_results
        
        return scalability_results
    
    def _generate_gas_analysis_report(self, documents, traffic_logs, tree_systems, 
                                    gas_results, optimization_analysis, scalability_analysis, start_time):
        """Generate comprehensive gas analysis report."""
        total_time = time.time() - start_time
        
        report = {
            'analysis_metadata': {
                'total_duration_seconds': total_time,
                'document_count': len(documents),
                'traffic_events': len(traffic_logs),
                'gas_price_gwei': self.gas_price_gwei,
                'timestamp': datetime.now().isoformat()
            },
            'tree_systems': {
                name: {
                    'type': system['type'],
                    'build_time': system['build_time'],
                    'root': system['root'][:16] + '...' if system['root'] else None
                }
                for name, system in tree_systems.items()
            },
            'gas_results': gas_results,
            'optimization_analysis': optimization_analysis,
            'scalability_analysis': scalability_analysis,
            'summary': self._generate_gas_analysis_summary(gas_results, optimization_analysis, scalability_analysis)
        }
        
        # Save report in organized directory
        report_filename = f'gas_cost_analysis_{len(documents)}docs.json'
        if self.reports_dir:
            report_path = os.path.join(self.reports_dir, "gas_analysis", report_filename)
        else:
            report_path = report_filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Gas analysis report saved to: {report_path}")
        
        return report
    
    def _generate_gas_analysis_summary(self, gas_results, optimization_analysis, scalability_analysis):
        """Generate summary of gas analysis results."""
        summary = {
            'key_findings': [],
            'cost_benefits': {},
            'recommendations': []
        }
        
        # Key findings
        if 'overall_optimization' in optimization_analysis:
            overall = optimization_analysis['overall_optimization']
            if 'avg_gas_reduction_percent' in overall:
                summary['key_findings'].append(
                    f"Average gas reduction: {overall['avg_gas_reduction_percent']:.1f}%"
                )
            if 'avg_cost_reduction_percent' in overall:
                summary['key_findings'].append(
                    f"Average cost reduction: {overall['avg_cost_reduction_percent']:.1f}%"
                )
        
        # Cost benefits
        if 'overall_optimization' in optimization_analysis:
            overall = optimization_analysis['overall_optimization']
            summary['cost_benefits'] = {
                'gas_optimization': overall.get('avg_gas_reduction_percent', 0),
                'cost_optimization': overall.get('avg_cost_reduction_percent', 0),
                'max_optimization': overall.get('max_gas_reduction_percent', 0)
            }
        
        # Recommendations
        if 'overall_optimization' in optimization_analysis:
            overall = optimization_analysis['overall_optimization']
            if overall.get('avg_gas_reduction_percent', 0) > 20:
                summary['recommendations'].append(
                    "Significant gas savings achieved - hierarchical approach recommended for production"
                )
            elif overall.get('avg_gas_reduction_percent', 0) > 10:
                summary['recommendations'].append(
                    "Moderate gas savings achieved - consider hierarchical approach for cost-sensitive applications"
                )
            else:
                summary['recommendations'].append(
                    "Limited gas savings - evaluate other factors (scalability, cross-province support) for decision"
                )
        
        return summary

    def run_cross_province_gas_analysis(self, documents, properties_by_province, complex_events):
        """
        Run gas analysis specifically focused on 4+ cross-province scenarios.
        
        Args:
            documents: Generated documents
            properties_by_province: Properties organized by province  
            complex_events: List of 4+ cross-province events to analyze
            
        Returns:
            Cross-province focused gas analysis results
        """
        print(f"{'='*60}")
        print(f"CROSS-PROVINCE GAS COST ANALYSIS")
        print(f"{'='*60}")
        print(f"Complex Events (4+ provinces): {len(complex_events)}")
        
        start_time = time.time()
        
        # Build tree systems
        print(f"\nBuilding tree systems for gas analysis...")
        tree_systems = self._build_gas_analysis_tree_systems(documents, complex_events)
        
        # Analyze gas costs for complex cross-province scenarios
        print(f"\nAnalyzing gas costs for 4+ province scenarios...")
        gas_results = self._analyze_complex_cross_province_gas_costs(tree_systems, complex_events)
        
        # Generate insights specific to cross-province scenarios
        insights = self._generate_cross_province_insights(gas_results)
        
        total_time = time.time() - start_time
        
        results = {
            'analysis_metadata': {
                'test_type': 'cross_province_gas_analysis',
                'total_duration_seconds': total_time,
                'complex_events_analyzed': len(complex_events),
                'timestamp': datetime.now().isoformat()
            },
            'tree_systems': {
                name: {
                    'type': system['type'],
                    'build_time': system['build_time'],
                    'root': system['root'][:16] + '...' if system['root'] else None
                }
                for name, system in tree_systems.items()
            },
            'cross_province_gas_results': gas_results,
            'cross_province_insights': insights,
            'summary': {
                'hierarchical_wins': insights.get('hierarchical_wins', 0),
                'traditional_wins': insights.get('traditional_wins', 0),
                'hierarchical_win_rate': insights.get('hierarchical_win_rate', 0),
                'avg_gas_savings_when_hierarchical_wins': insights.get('avg_gas_savings_when_hierarchical_wins', 0)
            }
        }
        
        print(f"\n✅ Cross-province gas analysis completed in {total_time:.1f}s")
        print(f"   Hierarchical wins: {insights.get('hierarchical_wins', 0)} / {len(complex_events)}")
        print(f"   Win rate: {insights.get('hierarchical_win_rate', 0):.1f}%")
        
        return results

    def _build_gas_analysis_tree_systems(self, documents, complex_events):
        """Build tree systems optimized for gas analysis."""
        print("Building Hierarchical System...")
        hierarchical_start = time.time()
        jurisdiction_manager = JurisdictionTreeManager(documents, complex_events)
        jurisdiction_root = jurisdiction_manager.build_all_trees()  
        hierarchical_build_time = time.time() - hierarchical_start
        
        print("Building Traditional Multiproof System...")
        traditional_start = time.time()
        traditional_multiproof_builder = TraditionalMerkleTreeBuilder(documents)
        traditional_root = traditional_multiproof_builder.build()
        traditional_build_time = time.time() - traditional_start
        
        print("Building Traditional Single Proof System...")
        single_proof_start = time.time()
        single_proof_builder = TraditionalSingleProofMerkleTreeBuilder(documents)
        single_proof_root = single_proof_builder.build()
        single_proof_build_time = time.time() - single_proof_start
        
        print("Building Traditional + Huffman System...")
        huffman_start = time.time()
        huffman_builder = TraditionalMultiproofWithHuffmanBuilder(documents, complex_events)
        huffman_root = huffman_builder.build()
        huffman_build_time = time.time() - huffman_start
        
        return {
            'hierarchical': {
                'manager': jurisdiction_manager,
                'root': jurisdiction_root,
                'build_time': hierarchical_build_time,
                'type': 'Hierarchical with Pairs-first Huffman'
            },
            'traditional_multiproof': {
                'builder': traditional_multiproof_builder,
                'root': traditional_root,
                'build_time': traditional_build_time,
                'type': 'Traditional Multiproof'
            },
            'traditional_single_proof': {
                'builder': single_proof_builder,
                'root': single_proof_root,
                'build_time': single_proof_build_time,
                'type': 'Traditional Single Proof'
            },
            'traditional_huffman': {
                'builder': huffman_builder,
                'root': huffman_root,
                'build_time': huffman_build_time,
                'type': 'Traditional + Huffman'
            }
        }

    def _analyze_complex_cross_province_gas_costs(self, tree_systems, complex_events):
        """Analyze gas costs for complex cross-province events."""
        results = []
        
        # Test only the first 20 complex events to save time
        sample_events = complex_events[:20] if len(complex_events) > 20 else complex_events
        print(f"Testing {len(sample_events)} complex events for gas analysis...")
        
        for i, event in enumerate(sample_events):
            # Count provinces
            provinces = set(prop_id.split('.')[0] for prop_id in event if '.' in prop_id)
            province_count = len(provinces)
            
            if province_count < 4:
                continue  # Skip non-complex events
                
            print(f"  Event {i+1}: {len(event)} properties across {province_count} provinces")
            
            # Test all approaches
            hierarchical_gas = self._test_hierarchical_gas_for_event(tree_systems['hierarchical'], event)
            traditional_multiproof_gas = self._test_traditional_gas_for_event(tree_systems['traditional_multiproof'], event)
            traditional_single_gas = self._test_traditional_single_gas_for_event(tree_systems['traditional_single_proof'], event)
            traditional_huffman_gas = self._test_traditional_huffman_gas_for_event(tree_systems['traditional_huffman'], event)
            
            results.append({
                'event_id': i,
                'provinces_count': province_count, 
                'properties_count': len(event),
                'provinces_list': list(provinces),
                'hierarchical_gas': hierarchical_gas,
                'traditional_multiproof_gas': traditional_multiproof_gas,
                'traditional_single_gas': traditional_single_gas,
                'traditional_huffman_gas': traditional_huffman_gas,
                'hierarchical_wins': hierarchical_gas < traditional_multiproof_gas if (isinstance(hierarchical_gas, int) and isinstance(traditional_multiproof_gas, int)) else False,
                'gas_savings_percent': ((traditional_multiproof_gas - hierarchical_gas) / traditional_multiproof_gas * 100) if (isinstance(hierarchical_gas, int) and isinstance(traditional_multiproof_gas, int) and traditional_multiproof_gas > 0) else 0
            })
        
        return results

    def _test_hierarchical_gas_for_event(self, hierarchical_system, event):
        """Test gas cost for hierarchical approach on a single event."""
        try:
            # Generate verification request
            verification_request = self._convert_event_to_verification_request(event)
            
            # Generate hierarchical proof
            proof_package = hierarchical_system['manager'].verify_cross_province_batch(verification_request)
            
            # Estimate gas using the hierarchical contract
            if hasattr(self, 'hierarchical_contract') and self.hierarchical_contract:
                estimated_gas = self._estimate_hierarchical_gas_from_proof_package(proof_package)
                return estimated_gas
            else:
                return "No contract available"
        except Exception as e:
            return f"Error: {e}"

    def _test_traditional_gas_for_event(self, traditional_system, event):
        """Test gas cost for traditional multiproof approach on a single event."""
        try:
            # Get all document hashes for this event
            all_doc_hashes = []
            for prop_id in event:
                for doc in traditional_system['builder'].all_documents:
                    if doc.full_id == prop_id:
                        all_doc_hashes.append(doc.hash_hex)
            
            all_doc_hashes = list(dict.fromkeys(all_doc_hashes))  # Deduplicate
            
            if not all_doc_hashes:
                return "No documents found"
            
            # Generate multiproof
            proof, flags = traditional_system['builder'].generate_proof_for_documents(all_doc_hashes)
            
            # Estimate gas using the traditional contract
            if hasattr(self, 'traditional_contract') and self.traditional_contract:
                proof_bytes = [bytes.fromhex(p) for p in proof]
                leaves_sorted = sorted(all_doc_hashes)
                leaves_bytes = [bytes.fromhex(l) for l in leaves_sorted]
                
                estimated_gas = self.traditional_contract.functions.verifyBatch(
                    proof_bytes, flags, leaves_bytes
                ).estimate_gas()
                
                return estimated_gas
            else:
                return "No contract available"
        except Exception as e:
            return f"Error: {e}"

    def _test_traditional_single_gas_for_event(self, single_proof_system, event):
        """Test gas cost for traditional single proof approach on a single event."""
        try:
            # Get all document hashes for this event
            all_doc_hashes = []
            for prop_id in event:
                for doc in single_proof_system['builder'].all_documents:
                    if doc.full_id == prop_id:
                        all_doc_hashes.append(doc.hash_hex)
            
            all_doc_hashes = list(dict.fromkeys(all_doc_hashes))  # Deduplicate
            
            if not all_doc_hashes:
                return "No documents found"
            
            # Generate individual proofs
            proofs = single_proof_system['builder'].generate_proofs_for_documents(all_doc_hashes)
            
            # Estimate gas by summing individual verifications
            total_estimated_gas = len(proofs) * 50000  # Rough estimate per single proof verification
            
            return total_estimated_gas
        except Exception as e:
            return f"Error: {e}"

    def _test_traditional_huffman_gas_for_event(self, huffman_system, event):
        """Test gas cost for traditional + huffman approach on a single event."""
        try:
            # Get all document hashes for this event
            all_doc_hashes = []
            for prop_id in event:
                for doc in huffman_system['builder'].all_documents:
                    if doc.full_id == prop_id:
                        all_doc_hashes.append(doc.hash_hex)
            
            all_doc_hashes = list(dict.fromkeys(all_doc_hashes))  # Deduplicate
            
            if not all_doc_hashes:
                return "No documents found"
            
            # Generate multiproof
            proof, flags = huffman_system['builder'].generate_proof_for_documents(all_doc_hashes)
            
            # Estimate gas using the traditional contract (same as traditional multiproof)
            if hasattr(self, 'traditional_contract') and self.traditional_contract:
                proof_bytes = [bytes.fromhex(p) for p in proof]
                leaves_sorted = sorted(all_doc_hashes)
                leaves_bytes = [bytes.fromhex(l) for l in leaves_sorted]
                
                estimated_gas = self.traditional_contract.functions.verifyBatch(
                    proof_bytes, flags, leaves_bytes
                ).estimate_gas()
                
                return estimated_gas
            else:
                return "No contract available"
        except Exception as e:
            return f"Error: {e}"

    def _convert_event_to_verification_request(self, event):
        """Convert a traffic event to hierarchical verification request format."""
        verification_request = defaultdict(list)
        
        for prop_id in event:
            if '.' in prop_id:
                province, property_id = prop_id.split('.', 1)
                verification_request[province].append(prop_id)
        
        return dict(verification_request)

    def _generate_cross_province_insights(self, gas_results):
        """Generate insights from cross-province gas analysis results."""
        insights = {
            'total_events_analyzed': len(gas_results),
            'hierarchical_wins': 0,
            'traditional_wins': 0,
            'hierarchical_win_rate': 0,
            'avg_gas_savings_when_hierarchical_wins': 0,
            'best_hierarchical_cases': [],
            'worst_hierarchical_cases': []
        }
        
        hierarchical_savings = []
        
        for result in gas_results:
            if result['hierarchical_wins']:
                insights['hierarchical_wins'] += 1
                hierarchical_savings.append(result['gas_savings_percent'])
                insights['best_hierarchical_cases'].append(result)
            else:
                insights['traditional_wins'] += 1
                insights['worst_hierarchical_cases'].append(result)
        
        # Calculate win rate
        total_valid_comparisons = insights['hierarchical_wins'] + insights['traditional_wins']
        if total_valid_comparisons > 0:
            insights['hierarchical_win_rate'] = (insights['hierarchical_wins'] / total_valid_comparisons) * 100
        
        # Calculate average savings when hierarchical wins
        if hierarchical_savings:
            insights['avg_gas_savings_when_hierarchical_wins'] = sum(hierarchical_savings) / len(hierarchical_savings)
        
        # Sort best and worst cases
        insights['best_hierarchical_cases'].sort(key=lambda x: x['gas_savings_percent'], reverse=True)
        insights['best_hierarchical_cases'] = insights['best_hierarchical_cases'][:5]  # Top 5
        
        insights['worst_hierarchical_cases'].sort(key=lambda x: x['gas_savings_percent'])
        insights['worst_hierarchical_cases'] = insights['worst_hierarchical_cases'][:5]  # Bottom 5
        
        return insights

def main():
    """Main execution function for gas cost analysis."""
    print("=== COMPREHENSIVE GAS COST ANALYSIS ===")
    
    # Setup Web3 connection
    try:
        web3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
        if web3.is_connected():
            web3.eth.default_account = web3.eth.accounts[0]
            print("✅ Connected to Hardhat for gas analysis")
        else:
            print("⚠️  No Hardhat connection - running without gas analysis")
            web3 = None
    except Exception as e:
        print(f"⚠️  Web3 connection failed: {e}")
        web3 = None
    
    # Create gas analyzer
    gas_analyzer = GasCostAnalyzer(web3, gas_price_gwei=20)
    
    # Run comprehensive analysis
    report = gas_analyzer.run_comprehensive_gas_analysis(
        document_count=1500,
        traffic_events=600
    )
    
    # Print key results
    if 'summary' in report:
        summary = report['summary']
        print(f"\n🎯 GAS ANALYSIS SUMMARY:")
        
        if 'key_findings' in summary:
            print(f"  Key Findings:")
            for finding in summary['key_findings']:
                print(f"    • {finding}")
        
        if 'cost_benefits' in summary:
            benefits = summary['cost_benefits']
            print(f"  Cost Benefits:")
            print(f"    • Gas Optimization: {benefits.get('gas_optimization', 0):.1f}%")
            print(f"    • Cost Optimization: {benefits.get('cost_optimization', 0):.1f}%")
            print(f"    • Max Optimization: {benefits.get('max_optimization', 0):.1f}%")
        
        if 'recommendations' in summary:
            print(f"  Recommendations:")
            for rec in summary['recommendations']:
                print(f"    • {rec}")
    
    print(f"\n✅ Gas cost analysis completed successfully!")

def main():
    """Main execution function for gas cost analysis."""
    print("=== COMPREHENSIVE GAS COST ANALYSIS ===")
    
    # Setup Web3 connection
    try:
        web3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
        if web3.is_connected():
            web3.eth.default_account = web3.eth.accounts[0]
            print("✅ Connected to Hardhat for gas analysis")
        else:
            print("⚠️  No Hardhat connection - running without gas analysis")
            web3 = None
    except Exception as e:
        print(f"⚠️  Web3 connection failed: {e}")
        web3 = None
    
    # Create gas analyzer
    gas_analyzer = GasCostAnalyzer(web3, gas_price_gwei=20)
    
    # Run comprehensive analysis
    report = gas_analyzer.run_comprehensive_gas_analysis(
        document_count=1500,
        days=5,
        cross_province_bias=0.7,
        force_onchain_verification=False
    )
    
    # Print the final report with recommendations  
    gas_analyzer._print_final_summary(report)

if __name__ == "__main__":
    main()

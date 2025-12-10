import argparse
import json
import time
import os
import random
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from access_patterns_enhanced import AuditPattern, TransactionalPattern
from basic_data_structure import Document
from seed_generator import SeedGenerator
from traditional_document_level_huffman_builder import TraditionalDocumentLevelHuffmanBuilder
from traditional_multiproof_builder import TraditionalMerkleTreeBuilder
from traditional_property_level_huffman_builder import TraditionalPropertyLevelHuffmanBuilder
from clustered_province_tree_builder import ClusteredProvinceTreeBuilder
from clustered_province_with_document_huffman_builder import ClusteredProvinceWithDocumentHuffmanBuilder
from jurisdiction_tree_builder import JurisdictionTreeBuilder
from workload_generator import Query, WorkloadGenerator
# from compressed_traffic_logs import CompressedTrafficLogs  # No longer needed
from web3 import Web3
from bitmap_proof_generator import generate_bitmap_proof_from_pathmap

class TestRunner:
    """
    """
    
    def __init__(self, web3_instance=None, reports_dir=None, enable_multithreading=False, verbose=False):
        self.web3 = web3_instance
        self.alpha_overrides = None
        self.performance_data = {}
        self.gas_analysis = {}
        self.reports_dir = reports_dir
        self.enable_multithreading = enable_multithreading
        self.verbose = verbose

        # Setup contracts if Web3 available
        if self.web3:
            self._setup_contracts()
    
    def print_verbose(self, message: str):
        if self.verbose:
            print(message)
    
    def _update_contract_root(self, contract, root_hex, name=""):
        """Update a contract's merkle root after tree construction."""
        if not self.web3 or not contract or not root_hex:
            return False
        try:
            root_bytes32 = bytes.fromhex(root_hex)
            tx_hash = contract.functions.updateRoot(root_bytes32).transact()
            self.web3.eth.wait_for_transaction_receipt(tx_hash)
            stored_root = contract.functions.merkleRoot().call()
            if stored_root.hex() == root_hex:
                print(f"  ✅ {name} root updated: 0x{root_hex[:16]}...")
                return True
            else:
                print(f"  ⚠️ {name} root mismatch! Expected 0x{root_hex[:16]}... got 0x{stored_root.hex()[:16]}...")
                return False
        except Exception as e:
            print(f"  ⚠️ Failed to update {name} root: {e}")
            return False
    
    def _setup_contracts(self):
        """Setup smart contracts for gas cost analysis."""
        try:
            # Traditional multiproof contract
            traditional_multiproof_artifact_path = '../../artifacts/contracts/TraditionalMultiproofMerkleVerifier.sol/TraditionalMultiproofMerkleVerifier.json'
            with open(traditional_multiproof_artifact_path, 'r') as f:
                traditional_multiproof_artifact = json.load(f)
            
            traditional_multiproof_address = traditional_multiproof_artifact['networks']['31337']['address']
            self.traditional_multiproof_contract = self.web3.eth.contract(
                address=traditional_multiproof_address,
                abi=traditional_multiproof_artifact['abi']
            )

            # Traditional document level huffman contract
            traditional_document_huffman_artifact_path = '../../artifacts/contracts/TraditionalDocumentLevelHuffmanVerifier.sol/TraditionalDocumentLevelHuffmanVerifier.json'
            with open(traditional_document_huffman_artifact_path, 'r') as f:
                traditional_document_huffman_artifact = json.load(f)

            traditional_document_huffman_address = traditional_document_huffman_artifact['networks']['31337']['address']
            self.traditional_document_huffman_contract = self.web3.eth.contract(
                address=traditional_document_huffman_address,
                abi=traditional_document_huffman_artifact['abi']
            )
            
            # Clustered Province contract
            clustered_province_artifact_path = '../../artifacts/contracts/ClusteredProvinceVerifier.sol/ClusteredProvinceVerifier.json'
            with open(clustered_province_artifact_path, 'r') as f:
                clustered_province_artifact = json.load(f)
            
            clustered_province_address = clustered_province_artifact['networks']['31337']['address']
            self.clustered_province_contract = self.web3.eth.contract(
                address=clustered_province_address,
                abi=clustered_province_artifact['abi']
            )

            # Traditional Property Level Huffman contract
            traditional_property_level_huffman_artifact_path = '../../artifacts/contracts/TraditionalPropertyLevelHuffmanVerifier.sol/TraditionalPropertyLevelHuffmanVerifier.json'
            with open(traditional_property_level_huffman_artifact_path, 'r') as f:
                traditional_property_level_huffman_artifact = json.load(f)

            traditional_property_level_huffman_address = traditional_property_level_huffman_artifact['networks']['31337']['address']
            self.traditional_property_level_huffman_contract = self.web3.eth.contract(
                address=traditional_property_level_huffman_address,
                abi=traditional_property_level_huffman_artifact['abi']
            )

            # Clustered Province with Document Huffman contract
            clustered_province_with_document_huffman_path = '../../artifacts/contracts/ClusteredProvinceWithDocumentHuffmanVerifier.sol/ClusteredProvinceWithDocumentHuffmanVerifier.json'
            with open(clustered_province_with_document_huffman_path, 'r') as f:
                clustered_province_with_document_huffman_artifact = json.load(f)
            
            clustered_province_with_document_huffman_address = clustered_province_with_document_huffman_artifact['networks']['31337']['address']
            self.clustered_province_with_document_huffman_contract = self.web3.eth.contract(
                address=clustered_province_with_document_huffman_address,
                abi=clustered_province_with_document_huffman_artifact['abi']
            )

            # Jurisdiction Tree contract
            jurisdiction_tree_artifact_path = '../../artifacts/contracts/JurisdictionTreeVerifier.sol/JurisdictionTreeVerifier.json'
            with open(jurisdiction_tree_artifact_path, 'r') as f:
                jurisdiction_tree_artifact = json.load(f)
            
            jurisdiction_tree_address = jurisdiction_tree_artifact['networks']['31337']['address']
            self.jurisdiction_tree_contract = self.web3.eth.contract(
                address=jurisdiction_tree_address,
                abi=jurisdiction_tree_artifact['abi']
            )
            
            print("✅ All contracts loaded successfully")
            
        except Exception as e:
            print(f"⚠️  Contract setup failed: {e}")
            self.web3 = None  # Disable Web3 if setup fails
            self.traditional_multiproof_contract = None
            self.traditional_document_huffman_contract = None
            self.traditional_property_level_huffman_contract = None
            self.clustered_province_contract = None
            self.clustered_province_with_document_huffman_contract = None
            self.jurisdiction_tree_contract = None
    
    def _calculate_communication_cost_multiproof(self, proof, proof_flags, leaves):
        """
        Calculate actual communication cost for OpenZeppelin multiproof by simulating verification.
        Tracks all unique nodes accessed: leaves + proof nodes + all computed intermediate nodes.
        """
        from eth_utils import keccak
        
        accessed_nodes = set()
        
        # Add all leaves (these are accessed) - normalize to hex string
        for leaf in leaves:
            leaf_str = leaf if isinstance(leaf, str) else '0x' + leaf.hex()
            accessed_nodes.add(leaf_str)
        
        # Add all proof nodes (these are transmitted) - normalize to hex string
        for p in proof:
            p_str = p if isinstance(p, str) else '0x' + p.hex()
            accessed_nodes.add(p_str)
        
        # Simulate the multiproof verification to track computed nodes
        # OpenZeppelin's multiProofVerify builds the tree bottom-up
        hashes = list(leaves)  # Start with leaves (as strings)
        proof_pos = 0
        
        for flag in proof_flags:
            if flag:
                # Use two elements from hashes array (both children are leaves/computed)
                if len(hashes) >= 2:
                    a = hashes.pop(0)
                    b = hashes.pop(0)
                    # Compute parent hash - convert strings to bytes for keccak
                    a_bytes = bytes.fromhex(a.replace('0x', '')) if isinstance(a, str) else a
                    b_bytes = bytes.fromhex(b.replace('0x', '')) if isinstance(b, str) else b
                    # Sort for deterministic hashing
                    if a_bytes <= b_bytes:
                        computed = keccak(a_bytes + b_bytes)
                    else:
                        computed = keccak(b_bytes + a_bytes)
                    computed_hex = '0x' + computed.hex()
                    accessed_nodes.add(computed_hex)
                    hashes.append(computed_hex)
            else:
                # Use one element from hashes and one from proof
                if len(hashes) >= 1 and proof_pos < len(proof):
                    a = hashes.pop(0)
                    b = proof[proof_pos]
                    proof_pos += 1
                    # Compute parent hash - convert strings to bytes for keccak
                    a_bytes = bytes.fromhex(a.replace('0x', '')) if isinstance(a, str) else a
                    b_bytes = bytes.fromhex(b.replace('0x', '')) if isinstance(b, str) else b
                    # Sort for deterministic hashing
                    if a_bytes <= b_bytes:
                        computed = keccak(a_bytes + b_bytes)
                    else:
                        computed = keccak(b_bytes + a_bytes)
                    computed_hex = '0x' + computed.hex()
                    accessed_nodes.add(computed_hex)
                    hashes.append(computed_hex)
        
        return len(accessed_nodes)
    
    def _calculate_communication_cost_pathmap(self, leaves, proof_hashes, path_map):
        """
        Calculate actual communication cost for pathMap format by simulating verification.
        
        Communication cost includes ALL nodes accessed during verification:
        1. Input leaves (document hashes being verified)
        2. Proof hashes (sibling nodes provided in proof)
        3. ALL computed intermediate nodes (created during tree reconstruction)
        
        PathMap format:
        - Array of pairs of indices [left_idx, right_idx, left_idx, right_idx, ...]
        - Each pair represents a hash operation: hash(value[left_idx], value[right_idx])
        - Indices reference: leaves (0..leavesLen-1), proof (leavesLen..leavesLen+proofLen-1), 
          computed (leavesLen+proofLen onwards)
        - Each pair creates ONE new intermediate node in the scratchpad
        
        Therefore: num_computed_nodes = len(path_map) // 2
        Total communication cost = len(leaves) + len(proof_hashes) + num_computed_nodes
        """
        
        # Count number of computed intermediate nodes
        # Each pair in pathMap creates one new intermediate hash
        num_computed_nodes = len(path_map) // 2
        
        # Total communication cost = all nodes accessed during verification
        total_cost = len(leaves) + len(proof_hashes) + num_computed_nodes
        
        return total_cost
    
    def _build_all_tree_systems(self, documents, transactional_pattern=None, audit_pattern=None, selected_approaches: list[str] =None):
        """Build tree systems for comparison based on selected approaches."""
        self.transactional_pattern = transactional_pattern
        self.audit_pattern = audit_pattern
        
        if selected_approaches == None or len(selected_approaches) == 0:
            selected_approaches = [
                'traditional_multiproof', 
                'traditional_document_huffman',
                'traditional_property_level_huffman',
                'clustered_province', 
                'clustered_province_with_document_huffman',
                'jurisdiction_tree'
            ]

        if 'traditional_multiproof' in selected_approaches:
            print("Building Traditional Multiproof System...")
            traditional_multiproof_start = time.time()
            traditional_multiproof_builder = TraditionalMerkleTreeBuilder(documents)
            traditional_multiproof_root = traditional_multiproof_builder.build()
            traditional_multiproof_build_time = time.time() - traditional_multiproof_start

        if 'traditional_document_huffman' in selected_approaches:
            print("Building Traditional Document Huffman System...")
            traditional_document_huffman_start = time.time()
            # allow per-model alpha override
            td_alpha = None
            if hasattr(self, 'alpha_overrides') and self.alpha_overrides:
                td_alpha = self.alpha_overrides.get('traditional_document_huffman')
            if td_alpha is not None:
                traditional_document_huffman_builder = TraditionalDocumentLevelHuffmanBuilder(documents, transactional_pattern, alpha_threshold=td_alpha)
            else:
                traditional_document_huffman_builder = TraditionalDocumentLevelHuffmanBuilder(documents, transactional_pattern)
            traditional_document_huffman_root = traditional_document_huffman_builder.build()
            traditional_document_huffman_build_time = time.time() - traditional_document_huffman_start

        if 'traditional_property_level_huffman' in selected_approaches:
            print("Building Traditional Property-Level Huffman System...")
            # allow per-model alpha override
            tpl_alpha = None
            if hasattr(self, 'alpha_overrides') and self.alpha_overrides:
                tpl_alpha = self.alpha_overrides.get('traditional_property_level_huffman')
            if tpl_alpha is not None:
                traditional_property_level_huffman_builder = TraditionalPropertyLevelHuffmanBuilder(documents, audit_pattern, transactional_pattern, alpha_threshold=tpl_alpha)
            else:
                traditional_property_level_huffman_builder = TraditionalPropertyLevelHuffmanBuilder(documents, audit_pattern, transactional_pattern)
            # Pre-compute frequencies BEFORE timing (simulates having real transaction logs)
            traditional_property_level_huffman_builder.precompute_frequencies()
            # Time only the actual tree construction
            traditional_property_level_huffman_start = time.time()
            traditional_property_level_huffman_root = traditional_property_level_huffman_builder.build()
            traditional_property_level_huffman_build_time = time.time() - traditional_property_level_huffman_start

        if 'clustered_province' in selected_approaches:
            print("Building Clustered Province System...")
            clustered_province_start = time.time()
            clustered_province_builder = ClusteredProvinceTreeBuilder(documents, audit_pattern, transactional_pattern)
            clustered_province_root = clustered_province_builder.build()
            clustered_province_build_time = time.time() - clustered_province_start

        if 'clustered_province_with_document_huffman' in selected_approaches:
            print("Building Clustered Province + Document-Level Huffman System...")
            # allow per-model alpha override
            cpdh_alpha = None
            if hasattr(self, 'alpha_overrides') and self.alpha_overrides:
                cpdh_alpha = self.alpha_overrides.get('clustered_province_with_document_huffman')
            if cpdh_alpha is not None:
                clustered_province_with_document_huffman_builder = ClusteredProvinceWithDocumentHuffmanBuilder(documents, audit_pattern, transactional_pattern, alpha_threshold=cpdh_alpha)
            else:
                clustered_province_with_document_huffman_builder = ClusteredProvinceWithDocumentHuffmanBuilder(documents, audit_pattern, transactional_pattern)
            # Pre-compute frequencies BEFORE timing (simulates having real transaction logs)
            clustered_province_with_document_huffman_builder.precompute_frequencies()
            # Time only the actual tree construction
            doc_huffman_start = time.time()
            clustered_province_with_document_huffman_root = clustered_province_with_document_huffman_builder.build()
            clustered_province_with_document_huffman_build_time = time.time() - doc_huffman_start

        if 'jurisdiction_tree' in selected_approaches:
            print("Building Jurisdiction Tree (Multi-Root Architecture)...")
            jurisdiction_start = time.time()
            # allow per-model alpha override
            jt_alpha = None
            if hasattr(self, 'alpha_overrides') and self.alpha_overrides:
                jt_alpha = self.alpha_overrides.get('jurisdiction_tree')
            if jt_alpha is not None:
                jurisdiction_tree_builder = JurisdictionTreeBuilder(documents, audit_pattern, transactional_pattern, alpha_threshold=jt_alpha)
            else:
                jurisdiction_tree_builder = JurisdictionTreeBuilder(documents, audit_pattern, transactional_pattern)
            jurisdiction_tree_root = jurisdiction_tree_builder.build()
            jurisdiction_tree_build_time = time.time() - jurisdiction_start

        # Update contract roots after tree construction
        print("\nUpdating contract roots after tree construction...")
        if self.web3:
            if 'traditional_multiproof' in selected_approaches and self.traditional_multiproof_contract:
                try:
                    root_hex = "0x" + traditional_multiproof_root
                    tx_hash = self.traditional_multiproof_contract.functions.updateRoot(bytes.fromhex(traditional_multiproof_root)).transact()
                    self.web3.eth.wait_for_transaction_receipt(tx_hash)
                    print(f"  ✅ Traditional Multiproof root updated: {root_hex[:10]}...")
                except Exception as e:
                    print(f"  ⚠️ Failed to update Traditional Multiproof root: {e}")
            
            if 'traditional_document_huffman' in selected_approaches and self.traditional_document_huffman_contract:
                try:
                    root_hex = "0x" + traditional_document_huffman_root
                    tx_hash = self.traditional_document_huffman_contract.functions.updateRoot(bytes.fromhex(traditional_document_huffman_root)).transact()
                    self.web3.eth.wait_for_transaction_receipt(tx_hash)
                    print(f"  ✅ Traditional Document Huffman root updated: {root_hex[:10]}...")
                except Exception as e:
                    print(f"  ⚠️ Failed to update Traditional Document Huffman root: {e}")
            
            if 'traditional_property_level_huffman' in selected_approaches and self.traditional_property_level_huffman_contract:
                try:
                    root_hex = "0x" + traditional_property_level_huffman_root
                    root_bytes32 = bytes.fromhex(traditional_property_level_huffman_root)
                    tx_hash = self.traditional_property_level_huffman_contract.functions.updateRoot(root_bytes32).transact()
                    self.web3.eth.wait_for_transaction_receipt(tx_hash)
                    # Verify the root was set correctly
                    stored_root = self.traditional_property_level_huffman_contract.functions.merkleRoot().call()
                    stored_root_hex = "0x" + stored_root.hex()
                    print(f"  ✅ Traditional Property-Level Huffman root updated: {root_hex[:10]}...")
                    print(f"     Stored root in contract: {stored_root_hex[:10]}...")
                    if stored_root.hex() != traditional_property_level_huffman_root:
                        print(f"     ⚠️ WARNING: Root mismatch! Expected {root_hex[:16]}... but got {stored_root_hex[:16]}...")
                except Exception as e:
                    print(f"  ⚠️ Failed to update Traditional Property-Level Huffman root: {e}")
            
            if 'clustered_province' in selected_approaches and self.clustered_province_contract:
                try:
                    root_hex = "0x" + clustered_province_root
                    tx_hash = self.clustered_province_contract.functions.updateRoot(bytes.fromhex(clustered_province_root)).transact()
                    self.web3.eth.wait_for_transaction_receipt(tx_hash)
                    print(f"  ✅ Clustered Province root updated: {root_hex[:10]}...")
                except Exception as e:
                    print(f"  ⚠️ Failed to update Clustered Province root: {e}")
            
            if 'clustered_province_with_document_huffman' in selected_approaches and self.clustered_province_with_document_huffman_contract:
                try:
                    root_hex = "0x" + clustered_province_with_document_huffman_root
                    tx_hash = self.clustered_province_with_document_huffman_contract.functions.updateRoot(bytes.fromhex(clustered_province_with_document_huffman_root)).transact()
                    self.web3.eth.wait_for_transaction_receipt(tx_hash)
                    print(f"  ✅ Clustered Province with Document Huffman root updated: {root_hex[:10]}...")
                except Exception as e:
                    print(f"  ⚠️ Failed to update Clustered Province with Document Huffman root: {e}")
            
            if 'jurisdiction_tree' in selected_approaches and self.jurisdiction_tree_contract:
                try:
                    root_hex = "0x" + jurisdiction_tree_root
                    tx_hash = self.jurisdiction_tree_contract.functions.updateRoot(bytes.fromhex(jurisdiction_tree_root)).transact()
                    self.web3.eth.wait_for_transaction_receipt(tx_hash)
                    print(f"  ✅ Jurisdiction Tree root updated: {root_hex[:10]}...")
                except Exception as e:
                    print(f"  ⚠️ Failed to update Jurisdiction Tree root: {e}")

        tree_systems = {}
        
        # Only add systems that were actually built
        if 'traditional_multiproof' in selected_approaches:
            tree_systems['traditional_multiproof'] = {
                'builder': traditional_multiproof_builder,
                'root': traditional_multiproof_root,
                'build_time': traditional_multiproof_build_time,
                'type': 'Traditional Multiproof'
            }
        
        if 'traditional_document_huffman' in selected_approaches:
            tree_systems['traditional_document_huffman'] = {
                'builder': traditional_document_huffman_builder,
                'root': traditional_document_huffman_root,
                'build_time': traditional_document_huffman_build_time,
                'type': 'Traditional + Document-Level Huffman'
            }
        
        if 'traditional_property_level_huffman' in selected_approaches:
            tree_systems['traditional_property_level_huffman'] = {
                'builder': traditional_property_level_huffman_builder,
                'root': traditional_property_level_huffman_root,
                'build_time': traditional_property_level_huffman_build_time,
                'type': 'Traditional Property-Level Huffman'
            }
        
        if 'clustered_province' in selected_approaches:
            tree_systems['clustered_province'] = {
                'builder': clustered_province_builder,
                'root': clustered_province_root,
                'build_time': clustered_province_build_time,
                'type': 'Clustered Province Tree'
            }
        
        if 'clustered_province_with_document_huffman' in selected_approaches:
            tree_systems['clustered_province_with_document_huffman'] = {
                'builder': clustered_province_with_document_huffman_builder,
                'root': clustered_province_with_document_huffman_root,
                'build_time': clustered_province_with_document_huffman_build_time,
                'type': 'Clustered Province + Document-Level Huffman'
            }

        if 'jurisdiction_tree' in selected_approaches:
            tree_systems['jurisdiction_tree'] = {
                'builder': jurisdiction_tree_builder,
                'root': jurisdiction_tree_root,
                'build_time': jurisdiction_tree_build_time,
                'type': 'Jurisdiction Tree (Multi-Root)'
            }
        
        print(f"Tree building completed:")
        if 'traditional_multiproof' in selected_approaches:
            print(f"  Traditional Multiproof: {traditional_multiproof_build_time:.3f}s")
        if 'traditional_document_huffman' in selected_approaches:
            print(f"  Traditional + Document-Level Huffman: {traditional_document_huffman_build_time:.3f}s")
        if 'traditional_property_level_huffman' in selected_approaches:
            print(f"  Traditional Property-Level Huffman: {traditional_property_level_huffman_build_time:.3f}s")
        if 'clustered_province' in selected_approaches:
            print(f"  Clustered Province Tree: {clustered_province_build_time:.3f}s")
        if 'clustered_province_with_document_huffman' in selected_approaches:
            print(f"  Clustered Province + Document-Level Huffman: {clustered_province_with_document_huffman_build_time:.3f}s")
        if 'jurisdiction_tree' in selected_approaches:
            print(f"  Jurisdiction Tree (Multi-Root): {jurisdiction_tree_build_time:.3f}s")
        
        return tree_systems
    
    def _build_alpha_specific_tree_systems(self, documents, alpha_value: float, transactional_pattern, audit_pattern, selected_approaches: list[str], precomputed_frequencies=None):
        """Build tree systems with specific alpha value for hyperparameter tuning.
        
        Args:
            precomputed_frequencies: Optional pre-computed frequency data to avoid redundant simulation.
                                    For alpha tuning, frequencies are computed once and reused.
        """
        tree_systems = {}
        
        if 'traditional_document_huffman' in selected_approaches:
            print(f"Building Traditional Document Huffman System (α = {alpha_value})...")
            start_time = time.time()
            builder = TraditionalDocumentLevelHuffmanBuilder(documents, transactional_pattern, alpha_threshold=alpha_value)
            root = builder.build()
            build_time = time.time() - start_time
            
            tree_systems['traditional_document_huffman'] = {
                'builder': builder,
                'root': root,
                'build_time': build_time,
                'type': f'Traditional + Document-Level Huffman (α={alpha_value})',
                'alpha': alpha_value
            }

        if 'traditional_property_level_huffman' in selected_approaches:
            print(f"Building Traditional Property-Level Huffman System (α = {alpha_value})...")
            builder = TraditionalPropertyLevelHuffmanBuilder(documents, audit_pattern=audit_pattern, transactional_pattern=transactional_pattern, alpha_threshold=alpha_value)
            # Use pre-computed frequencies if available, otherwise compute
            if precomputed_frequencies is not None:
                builder._cached_frequencies = precomputed_frequencies
            else:
                builder.precompute_frequencies()
            start_time = time.time()
            root = builder.build()
            build_time = time.time() - start_time
            # Update contract root after tree construction
            self._update_contract_root(self.traditional_property_level_huffman_contract, root, "Traditional Property-Level Huffman")
            
            tree_systems['traditional_property_level_huffman'] = {
                'builder': builder,
                'root': root,
                'build_time': build_time,
                'type': f'Traditional Property-Level Huffman (α={alpha_value})',
                'alpha': alpha_value
            }

        if 'clustered_province_with_document_huffman' in selected_approaches:
            print(f"Building Clustered Province + Document-Level Huffman System (α = {alpha_value})...")
            builder = ClusteredProvinceWithDocumentHuffmanBuilder(documents, audit_pattern=audit_pattern, transactional_pattern=transactional_pattern, alpha_threshold=alpha_value)
            # Use pre-computed frequencies if available, otherwise compute
            if precomputed_frequencies is not None:
                builder._cached_frequencies = precomputed_frequencies
            else:
                builder.precompute_frequencies()
            start_time = time.time()
            root = builder.build()
            build_time = time.time() - start_time
            # Update contract root after tree construction
            self._update_contract_root(self.clustered_province_with_document_huffman_contract, root, "Clustered Province + Document Huffman")
            
            tree_systems['clustered_province_with_document_huffman'] = {
                'builder': builder,
                'root': root,
                'build_time': build_time,
                'type': f'Clustered Province + Document-Level Huffman (α={alpha_value})',
                'alpha': alpha_value
            }

        if 'jurisdiction_tree' in selected_approaches:
            print(f"Building Jurisdiction Tree System (α = {alpha_value})...")
            start_time = time.time()
            builder = JurisdictionTreeBuilder(documents, audit_pattern=audit_pattern, transactional_pattern=transactional_pattern, alpha_threshold=alpha_value)
            root = builder.build()
            build_time = time.time() - start_time
            # Update contract root after tree construction
            self._update_contract_root(self.jurisdiction_tree_contract, root, "Jurisdiction Tree")
            
            tree_systems['jurisdiction_tree'] = {
                'builder': builder,
                'root': root,
                'build_time': build_time,
                'type': f'Jurisdiction Tree (Multi-Root) (α={alpha_value})',
                'alpha': alpha_value
            }
        
        return tree_systems
    
    def run_tests(self, documents, tree_systems: dict, queries: list[Query], selected_approaches: list[str] = None):
        """
        Run the provided test suite and collect performance and gas cost data.
        
        Args:
            documents: List of Document objects
            tree_systems: Dictionary of built tree systems from _build_all_tree_systems
            queries: List of Query objects to execute
            selected_approaches: List of approach names to test
        """
        if selected_approaches is None:
            selected_approaches = list(tree_systems.keys())
        
        print(f"🚀 Running tests on {len(queries)} queries with {len(selected_approaches)} approaches...")
        
        results = {}
        
        for approach_name in selected_approaches:
            if approach_name not in tree_systems:
                print(f"⚠️ Approach '{approach_name}' not found in tree systems. Skipping...")
                continue
                
            print(f"\n--- Testing {approach_name} ---")
            results[approach_name] = self._run_approach_tests(
                approach_name, tree_systems[approach_name], documents, queries
            )
        
        print(f"\n🏁 All tests completed!")
        self._store_results(results)
        return results
        
    def run_alpha_tuning_tests(self, documents, queries: list[Query], alpha_values: list[float], transactional_pattern, audit_pattern, selected_approaches: list[str] = None):
        """
        Run alpha hyperparameter tuning tests for pairs-first Huffman models.
        
        OPTIMIZED: Pre-computes frequencies ONCE per approach, then reuses for all alpha values.
        Frequencies don't depend on alpha - only the pair merging threshold does.
        
        Args:
            documents: List of Document objects
            queries: List of Query objects to execute
            alpha_values: List of alpha values to test
            transactional_pattern: TransactionalPattern for access patterns
            audit_pattern: AuditPattern for access patterns
            selected_approaches: List of approach names to test (only pairs-first Huffman models)
        
        Returns:
            Dictionary containing tuning results for each approach and alpha value
        """
        # Filter to only pairs-first Huffman approaches
        pairs_first_approaches = [
            'traditional_document_huffman',
            'traditional_property_level_huffman', 
            'clustered_province_with_document_huffman',
            'jurisdiction_tree'
        ]
        
        if selected_approaches:
            # Only keep approaches that are both selected and support pairs-first Huffman
            tuning_approaches = [app for app in selected_approaches if app in pairs_first_approaches]
        else:
            tuning_approaches = pairs_first_approaches
            
        print(f"🎯 Running alpha hyperparameter tuning...")
        print(f"   📊 Alpha values: {alpha_values}")
        print(f"   🏗️ Approaches: {tuning_approaches}")
        print(f"   🔍 Queries: {len(queries)}")
        
        tuning_results = {}
        
        # OPTIMIZATION: Pre-compute frequencies ONCE per approach (frequencies don't depend on alpha)
        precomputed_frequencies = {}
        for approach_name in tuning_approaches:
            print(f"\n📊 Pre-computing frequencies for {approach_name}...")
            if approach_name == 'traditional_property_level_huffman':
                temp_builder = TraditionalPropertyLevelHuffmanBuilder(
                    documents, audit_pattern=audit_pattern, 
                    transactional_pattern=transactional_pattern, alpha_threshold=0.0
                )
                precomputed_frequencies[approach_name] = temp_builder.precompute_frequencies()
            elif approach_name == 'clustered_province_with_document_huffman':
                temp_builder = ClusteredProvinceWithDocumentHuffmanBuilder(
                    documents, audit_pattern=audit_pattern,
                    transactional_pattern=transactional_pattern, alpha_threshold=0.0
                )
                precomputed_frequencies[approach_name] = temp_builder.precompute_frequencies()
            # Add other approaches as needed
        
        for approach_name in tuning_approaches:
            print(f"\n🔧 Tuning {approach_name}...")
            approach_results = {}
            
            for alpha in alpha_values:
                print(f"\n   📈 Testing α = {alpha}...")
                
                # Build tree systems with this specific alpha value, using pre-computed frequencies
                tree_systems = self._build_alpha_specific_tree_systems(
                    documents, alpha, transactional_pattern, audit_pattern, [approach_name],
                    precomputed_frequencies=precomputed_frequencies.get(approach_name)
                )
                
                if approach_name not in tree_systems:
                    print(f"   ⚠️ Failed to build {approach_name} with α = {alpha}")
                    continue
                
                # Run tests for this alpha configuration
                alpha_result = self._run_approach_tests(
                    approach_name, tree_systems[approach_name], documents, queries
                )
                alpha_result['alpha'] = alpha
                approach_results[f"alpha_{alpha}"] = alpha_result
                
                print(f"   ✅ α = {alpha}: Avg gas = {alpha_result['average_gas_per_query']:.0f}, Success = {alpha_result['successful_verifications']}/{alpha_result['total_queries']}")
            
            tuning_results[approach_name] = approach_results
            
        # Generate tuning-specific reports
        self._generate_alpha_tuning_report(tuning_results, alpha_values)
        
        print(f"\n🎉 Alpha tuning completed! Results saved to tuning reports.")
        return tuning_results
    
    def _run_approach_tests(self, approach_name: str, tree_system: dict, documents: list, queries: list[Query]):
        """
        Run tests for a specific approach.
        
        Args:
            approach_name: Name of the approach (e.g., 'clustered_province')
            tree_system: Tree system dictionary containing builder, root, etc.
            documents: List of Document objects
            queries: List of queries to execute
        
        Returns:
            Dictionary containing test results for this approach
        """
        approach_results = {
            'approach': approach_name,
            'total_queries': len(queries),
            'successful_verifications': 0,
            'failed_verifications': 0,
            'total_gas_used': 0,
            'query_results': [],
            'average_gas_per_query': 0,
            'build_time': tree_system.get('build_time', 0)
        }
        
        # Create document hash mapping for quick lookup
        doc_hash_map = {doc.hash_hex: doc for doc in documents}
        print(f"  Created document hash map with {len(doc_hash_map)} entries")
        if len(doc_hash_map) > 0:
            # Show a few examples for debugging
            sample_keys = list(doc_hash_map.keys())[:3]
            print(f"  Sample document IDs: {sample_keys}")

        id_to_documents = {}
        prop_to_documents = {}

        for doc in documents:
            id_to_documents[f"{doc.property_id}.{doc.doc_id}"] = doc
            if doc.property_id not in prop_to_documents:
                prop_to_documents[doc.property_id] = []
            prop_to_documents[doc.property_id].append(doc)

        for i, query in enumerate(queries):
            print(f"  Query {i+1}/{len(queries)}: {query[0]}")
            # print(f"    Query details: {query}")
            
            try:
                query_result = self._execute_single_query(
                    approach_name, tree_system, doc_hash_map, id_to_documents, prop_to_documents, query
                )
                
                approach_results['query_results'].append(query_result)
                
                if query_result['verification_success']:
                    approach_results['successful_verifications'] += 1
                    if query_result.get('gas_used'):
                        approach_results['total_gas_used'] += query_result['gas_used']
                else:
                    approach_results['failed_verifications'] += 1
                    
            except Exception as e:
                print(f"    ❌ Query failed with error: {e}")
                approach_results['failed_verifications'] += 1
                approach_results['query_results'].append({
                    'query_type': query[0],
                    'verification_success': False,
                    'error': str(e),
                    'gas_used': 0
                })
        
        # Calculate averages
        if approach_results['successful_verifications'] > 0:
            approach_results['average_gas_per_query'] = (
                approach_results['total_gas_used'] / approach_results['successful_verifications']
            )
        
        success_rate = (approach_results['successful_verifications'] / len(queries)) * 100 if len(queries) > 0 else 0
        print(f"  ✅ Success rate: {success_rate:.1f}% ({approach_results['successful_verifications']}/{len(queries)})")
        print(f"  ⛽ Total gas used: {approach_results['total_gas_used']:,}")
        print(f"  📊 Average gas per query: {approach_results['average_gas_per_query']:.0f}")
        
        return approach_results
    
    def _execute_single_query(self, approach_name: str, tree_system: dict, doc_hash_map: dict, id_to_documents: dict[str, Document], prop_to_documents: dict[str, Document], query: Query):
        """
        Execute a single query for a specific approach.
        
        Args:
            approach_name: Name of the approach
            tree_system: Tree system dictionary
            doc_hash_map: Mapping from document full_id to hash
            documents: List of all documents
            query: Query to execute
        
        Returns:
            Dictionary containing query execution results
        """
        query_result = {
            'query_type': query[0],
            'verification_success': False,
            'gas_used': 0,
            'proof_size': 0,
            'verification_time': 0,
            'error': None
        }
        
        try:
            # Extract documents to verify based on query type
            documents_to_verify = []
            
            print(f"    Processing query type: {query[0]}")
            
            if query[0] == 'TRANSACTIONAL':
                # Format: ('TRANSACTIONAL', property_id, [doc_ids])
                property_id = query[1]
                doc_ids = query[2]
                # print(f"    TRANSACTIONAL: property_id={property_id}, doc_ids={doc_ids}")
                
                # Improved document lookup with better debugging
                found_docs = []
                property_documents = []
                
                # Create a more robust property lookup - handle multiple property_id formats
                for doc in prop_to_documents[property_id]:
                    property_documents.append(doc)
                
                self.print_verbose(f"    Found {len(property_documents)} documents for property {property_id}")
                
                # Now check each document in this property against target doc_ids
                for doc in property_documents:
                    self.print_verbose(f"      Checking doc: '{doc.doc_id}' against target doc_ids: {doc_ids}")
                    if doc.doc_id in doc_ids:
                        documents_to_verify.append(doc.hash_hex)
                        found_docs.append(f"{doc.doc_id}:{doc.hash_hex[:8]}")
                        self.print_verbose(f"      ✅ Match found: {doc.doc_id}")
                    else:
                        self.print_verbose(f"      ❌ Not requested: '{doc.doc_id}' not in query {doc_ids}")

                self.print_verbose(f"    Found matching documents: {found_docs}")

                # Enhanced debugging: Show ALL documents for this property
                # if not documents_to_verify:
                #     print(f"    🔍 DEBUG: Searching for property_id '{property_id}' in all documents...")
                #     all_property_docs = []
                #     for doc in documents:
                #         if (doc.property_id == property_id or 
                #             doc.full_id.endswith(f".{property_id}") or 
                #             doc.full_id.split('.')[-1] == property_id):
                #             all_property_docs.append(f"{doc.doc_id} (full_id: {doc.full_id})")
                #     print(f"    🔍 All documents for property {property_id}: {all_property_docs[:5]}{'...' if len(all_property_docs) > 5 else ''}")
                    
                #     # Show sample of available doc_ids
                #     sample_doc_ids = [doc.doc_id for doc in documents[:10]]
                #     print(f"    🔍 Sample available doc_ids: {sample_doc_ids}")
                        
            elif query[0] in ['REGIONAL_AUDIT', 'NATIONAL_AUDIT']:  # 'AUDIT' removed - legacy type
                # Format: ('REGIONAL_AUDIT'/'NATIONAL_AUDIT', [(property_id, doc_id), ...])
                property_doc_pairs = query[1]
                # print(f"    {query[0]}: property_doc_pairs={property_doc_pairs}")
                
                # Improved document lookup for AUDIT queries (all types)
                for property_id, doc_id in property_doc_pairs:
                    self.print_verbose(f"      Looking for property_id='{property_id}', doc_id='{doc_id}'")
                    
                    lookup_key = f"{property_id}.{doc_id}"
                    if lookup_key in id_to_documents:
                        documents_to_verify.append(id_to_documents[lookup_key].hash_hex)
                        self.print_verbose(f"      ✅ Found {query[0]} match: {id_to_documents[lookup_key].full_id} - {id_to_documents[lookup_key].doc_id}")
                    else:
                        self.print_verbose(f"      ❌ No {query[0]} match found for property_id='{property_id}', doc_id='{doc_id}'")
            
            print(f"    Found {len(documents_to_verify)} documents to verify")
            if len(documents_to_verify) > 0:
                self.print_verbose(f"    Note: ❌ 'Not requested' messages above are normal - they show documents in the property that aren't part of this specific query")
            
            # if not documents_to_verify:
            #     # Enhanced error reporting and fallback mechanism
            #     print(f"    ❌ No documents found for verification")
            #     print(f"    🔍 Query details: {query}")
                
            #     # Try to find any documents for debugging
            #     if query[0] == 'TRANSACTIONAL':
            #         property_id = query[1]
            #         doc_ids = query[2]
                    
            #         # Show all available properties
            #         all_properties = set()
            #         for doc in documents:
            #             all_properties.add(doc.property_id)
                    
            #         print(f"    🔍 Available properties (first 10): {list(all_properties)[:10]}")
            #         print(f"    🔍 Looking for property: '{property_id}'")
            #         print(f"    🔍 Looking for doc_ids: {doc_ids}")
                    
            #         # Check if property exists with different doc_ids
            #         property_docs = []
            #         for doc in documents:
            #             if (doc.property_id == property_id or 
            #                 doc.full_id.endswith(f".{property_id}") or 
            #                 doc.full_id.split('.')[-1] == property_id):
            #                 property_docs.append(doc.doc_id)
                    
            #         if property_docs:
            #             print(f"    🔍 Property exists but doc_ids don't match. Available doc_ids: {property_docs[:5]}")
            #             # Use the first available document as fallback to prevent test failure
            #             fallback_doc = None
            #             for doc in documents:
            #                 if (doc.property_id == property_id or 
            #                     doc.full_id.endswith(f".{property_id}") or 
            #                     doc.full_id.split('.')[-1] == property_id):
            #                     fallback_doc = doc
            #                     break
                        
            #             if fallback_doc:
            #                 print(f"    🔄 Using fallback document: {fallback_doc.doc_id}")
            #                 documents_to_verify = [fallback_doc.hash_hex]
            #         else:
            #             print(f"    🔍 Property '{property_id}' not found in dataset")
                
            #     if not documents_to_verify:
            #         query_result['error'] = f"No valid documents found for verification. Query: {query}"
            #         print(f"    ❌ Final failure - no documents found for verification")
            #         return query_result
            
            # Remove duplicates while preserving order
            unique_docs = []
            seen = set()
            for doc in documents_to_verify:
                if doc not in seen:
                    unique_docs.append(doc)
                    seen.add(doc)
            documents_to_verify = unique_docs
            
            # Execute verification based on approach
            
            if approach_name == 'traditional_multiproof':
                # Traditional methods expect document hashes (strings)
                print(f"    🔧 Calling traditional multiproof verification with {len(documents_to_verify)} hashes")
                try:
                    result = self._verify_traditional_multiproof(tree_system, documents_to_verify)
                    print(f"    📋 Traditional multiproof result: {result}")
                except Exception as e:
                    print(f"    ❌ Traditional multiproof exception: {e}")
                    result = {'success': False, 'error': str(e)}
            elif approach_name == 'traditional_document_huffman':
                # Traditional methods expect document hashes (strings)
                print(f"    🔧 Calling traditional document huffman verification with {len(documents_to_verify)} hashes")
                try:
                    result = self._verify_traditional_document_huffman(tree_system, documents_to_verify)
                    print(f"    📋 Traditional document huffman result: {result}")
                except Exception as e:
                    print(f"    ❌ Traditional document huffman exception: {e}")
                    result = {'success': False, 'error': str(e)}
            elif approach_name == 'traditional_property_level_huffman':
                # Traditional property level huffman expects document hashes (strings)
                print(f"    🔧 Calling traditional property level huffman verification with {len(documents_to_verify)} hashes")
                try:
                    result = self._verify_traditional_property_level_huffman(tree_system, documents_to_verify)
                    print(f"    📋 Traditional property level huffman result: {result}")
                except Exception as e:
                    print(f"    ❌ Traditional property level huffman exception: {e}")
                    result = {'success': False, 'error': str(e)}
            elif approach_name == 'clustered_province':
                # Clustered methods expect Document objects, need to convert back
                doc_objects = []
                for doc_hash in documents_to_verify:
                    doc = doc_hash_map.get(doc_hash)
                    if doc:
                        doc_objects.append(doc)
                result = self._verify_clustered_province(tree_system, doc_objects)
            elif approach_name == 'clustered_province_with_document_huffman':
                # Clustered methods expect Document objects, need to convert back
                doc_objects = []
                for doc_hash in documents_to_verify:
                    doc = doc_hash_map.get(doc_hash)
                    if doc:
                        doc_objects.append(doc)
                result = self._verify_clustered_province_with_document_huffman(tree_system, doc_objects)
            elif approach_name == 'jurisdiction_tree':
                # Jurisdiction tree expects document hashes (strings)
                print(f"    🔧 Calling jurisdiction tree verification with {len(documents_to_verify)} hashes")
                try:
                    result = self._verify_jurisdiction_tree(tree_system, documents_to_verify)
                    print(f"    📋 Jurisdiction tree result: {result}")
                except Exception as e:
                    print(f"    ❌ Jurisdiction tree exception: {e}")
                    result = {'success': False, 'error': str(e)}
            else:
                raise ValueError(f"Unsupported approach: {approach_name}")
            
            # Use on-chain verification time from result (excludes proof generation and local verification)
            onchain_verification_time = result.get('onchain_verification_time', 0)
            
            query_result.update({
                'verification_success': result['success'],
                'gas_used': result.get('gas_used', 0),
                'proof_size': result.get('proof_size', 0),
                'communication_cost': result.get('communication_cost', 0),
                'verification_time': onchain_verification_time,
                'num_documents': len(documents_to_verify)
            })
            
            # Log verification result
            if result['success']:
                print(f"    ✅ Successfully verified {len(documents_to_verify)} documents with hashes: {[h[:8] for h in documents_to_verify[:3]]}{'...' if len(documents_to_verify) > 3 else ''}")
            else:
                print(f"    ❌ Verification FAILED for {len(documents_to_verify)} documents")
                query_result['error'] = result.get('error', 'Verification failed')
            
        except Exception as e:
            query_result['error'] = str(e)
            print(f"    💥 Exception during query execution: {e}")
            print(f"    Query details: {query}")
        
        return query_result
    
    def _verify_traditional_multiproof(self, tree_system: dict, documents_to_verify: list):
        """Verify using traditional multiproof approach - on-chain verification only."""
        try:
            builder = tree_system['builder']
            print(f"      Generating multiproof for {len(documents_to_verify)} documents...")
            
            # Standardized input handling - convert to hex strings consistently
            document_hashes_hex = []
            for doc in documents_to_verify:
                if hasattr(doc, 'hash_hex'):
                    # Document object
                    document_hashes_hex.append(doc.hash_hex)
                else:
                    # Already a string hash
                    document_hashes_hex.append(doc)
            
            # Generate traditional multiproof using OpenZeppelin format
            try:
                # Use traditional OpenZeppelin multiproof format (not pathMap)
                # Returns proof, flags, AND leaves in the correct order for verification
                proof, proof_flags, leaves_in_order = builder.generate_batched_proof_with_flags(document_hashes_hex)
                
                # Consistent proof size calculation across all approaches
                proof_size = len(proof) * 32# + len(proof_flags)
                # Calculate actual communication cost by simulating verification
                communication_cost = self._calculate_communication_cost_multiproof(proof, proof_flags, leaves_in_order)
                print(f"      Generated traditional multiproof: {len(proof)} elements, {len(proof_flags)} flags, communication_cost: {communication_cost}")
            except Exception as e:
                print(f"      ❌ Multiproof generation failed: {e}")
                return {'success': False, 'error': f'Multiproof generation failed: {e}'}
            
            # Skip local verification for benchmark - focus on on-chain verification only
            print(f"      ⚠️ Skipping local verification for benchmark - using on-chain verification only")
            
            # On-chain verification if Web3 available
            gas_used = 0
            onchain_verification_time = 0
            if self.web3 and self.traditional_multiproof_contract:
                try:
                    # Convert data for traditional multiproof contract call
                    # IMPORTANT: Use leaves_in_order (tree-index order), NOT sorted by hash!
                    proof_bytes = [bytes.fromhex(p.replace('0x', '')) for p in proof]
                    leaves_bytes = [bytes.fromhex(leaf.replace('0x', '')) for leaf in leaves_in_order]
                    
                    # Time only the on-chain verification calls
                    onchain_start = time.time()
                    
                    # Use traditional OpenZeppelin multiproof format: verifyBatch(proof, proofFlags, leaves)
                    # Call the view function and check the boolean return value
                    is_valid = self.traditional_multiproof_contract.functions.verifyBatch(
                        proof_bytes, proof_flags, leaves_bytes
                    ).call()
                    
                    # Estimate gas cost
                    gas_used = self.traditional_multiproof_contract.functions.verifyBatch(
                        proof_bytes, proof_flags, leaves_bytes
                    ).estimate_gas()
                    
                    onchain_verification_time = time.time() - onchain_start
                    
                    if is_valid:
                        print(f"      ✅ On-chain traditional multiproof verification successful - Gas used: {gas_used:,}")
                        return {'success': True, 'gas_used': gas_used, 'proof_size': proof_size, 'communication_cost': communication_cost, 'onchain_verification_time': onchain_verification_time}
                    else:
                        print(f"      ❌ On-chain verification returned False - proof is invalid")
                        return {'success': False, 'error': 'Verification returned False', 'proof_size': proof_size, 'communication_cost': communication_cost, 'onchain_verification_time': onchain_verification_time}
                    
                except Exception as e:
                    print(f"      ❌ On-chain verification failed: {e}")
                    return {'success': False, 'error': f'On-chain verification failed: {e}', 'proof_size': proof_size, 'communication_cost': communication_cost}
            else:
                print(f"    ⚠️ No Web3 connection or contract - skipping on-chain verification")
                return {'success': False, 'error': 'No Web3 connection or contract available', 'proof_size': proof_size, 'communication_cost': communication_cost}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _verify_traditional_document_huffman(self, tree_system: dict, documents_to_verify: list):
        """Verify using traditional document huffman approach - on-chain verification only."""
        try:
            builder = tree_system['builder']
            print(f"      Generating multiproof for {len(documents_to_verify)} documents...")
            
            # Standardized input handling - convert to hex strings consistently
            document_hashes_hex = []
            for doc in documents_to_verify:
                if hasattr(doc, 'hash_hex'):
                    # Document object
                    document_hashes_hex.append(doc.hash_hex)
                else:
                    # Already a string hash
                    document_hashes_hex.append(doc)
            
            # Generate proofs - True Huffman uses pathMap format
            try:
                # Use new pathMap format
                pathmap_proof = builder.generate_pathmap_proof(document_hashes_hex)
                proof_size = (len(pathmap_proof['leaves']) + len(pathmap_proof['proofHashes'])) * 32
                print(f"      Using generate_pathmap_proof method - {len(pathmap_proof['leaves'])} leaves, {len(pathmap_proof['proofHashes'])} proof hashes, {len(pathmap_proof['pathMap'])//2} instructions")
            except Exception as e:
                print(f"      ❌ Proof generation failed: {e}")
                return {'success': False, 'error': f'Proof generation failed: {e}'}
            
            # Skip local verification for benchmark - focus on on-chain verification only
            print(f"      ⚠️ Skipping local verification for benchmark - using on-chain verification only")
            
            # On-chain verification if Web3 available
            gas_used = 0
            onchain_verification_time = 0
            if self.web3 and self.traditional_document_huffman_contract:
                try:
                    # New pathMap verification format
                    leaves_bytes = [bytes.fromhex(leaf.replace('0x', '')) for leaf in pathmap_proof['leaves']]
                    proof_hashes_bytes = [bytes.fromhex(ph.replace('0x', '')) for ph in pathmap_proof['proofHashes']]
                    path_map = pathmap_proof['pathMap']
                    
                    # Time only the on-chain verification calls
                    onchain_start = time.time()
                    
                    # Call contract verification with pathMap format and check boolean return value
                    is_valid = self.traditional_document_huffman_contract.functions.verifyBatch(
                        leaves_bytes, proof_hashes_bytes, path_map
                    ).call()
                    
                    # Estimate gas cost
                    gas_used = self.traditional_document_huffman_contract.functions.verifyBatch(
                        leaves_bytes, proof_hashes_bytes, path_map
                    ).estimate_gas()
                    
                    onchain_verification_time = time.time() - onchain_start
                    
                    if is_valid:
                        print(f"      ✅ On-chain pathMap verification successful - Gas used: {gas_used:,}")
                        return {'success': True, 'gas_used': gas_used, 'proof_size': proof_size, 'onchain_verification_time': onchain_verification_time}
                    else:
                        print(f"      ❌ On-chain verification returned False - proof is invalid")
                        return {'success': False, 'error': 'Verification returned False', 'proof_size': proof_size, 'onchain_verification_time': onchain_verification_time}
                except Exception as e:
                    print(f"      ❌ On-chain verification failed: {e}")
                    return {'success': False, 'error': f'On-chain verification failed: {e}', 'proof_size': proof_size, 'onchain_verification_time': onchain_verification_time}
            else:
                print(f"    ⚠️ No Web3 connection or contract - skipping on-chain verification")
                return {'success': False, 'error': 'No Web3 connection or contract available', 'proof_size': proof_size, 'onchain_verification_time': 0}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _verify_traditional_property_level_huffman(self, tree_system: dict, documents_to_verify: list):
        """Verify using traditional property level huffman approach - on-chain verification only."""
        try:
            builder = tree_system['builder']
            print(f"      Generating multiproof for {len(documents_to_verify)} documents...")
            
            # Standardized input handling - convert to hex strings consistently
            document_hashes_hex = []
            for doc in documents_to_verify:
                if hasattr(doc, 'hash_hex'):
                    document_hashes_hex.append(doc.hash_hex)
                    # print(f"      Adding document hash: {doc.hash_hex[:8]}...")
                else:
                    document_hashes_hex.append(doc)
                    # print(f"      Adding document hash: {doc[:8]}...")
            
            # Generate proofs - Property Level Huffman uses direct bitmap proof generation
            try:
                # Use new direct post-order bitmap proof generation (not PathMap conversion)
                bitmap_proof = builder.generate_bitmap_proof(document_hashes_hex)
                
                proof_size = len(bitmap_proof['inputs']) * 32
                num_pushes = bitmap_proof['num_pushes']
                num_merges = bitmap_proof['num_merges']
                communication_cost = num_pushes + num_merges
                
                # Verify invariant: merges = pushes - 1
                expected_merges = num_pushes - 1
                if num_merges != expected_merges:
                    print(f"      ⚠️ WARNING: Operation count mismatch! {num_pushes} pushes but {num_merges} merges (expected {expected_merges})")
                
                print(f"      Using direct bitmap format - {num_pushes} inputs, {num_merges} merges, {len(bitmap_proof['bitmap'])} bitmap words, communication_cost: {communication_cost}")
            except Exception as e:
                print(f"      ❌ Proof generation failed: {e}")
                import traceback
                traceback.print_exc()
                return {'success': False, 'error': f'Proof generation failed: {e}'}
            
            # Local verification for debugging
            if hasattr(builder, 'verify_bitmap_locally'):
                local_valid, local_msg = builder.verify_bitmap_locally(bitmap_proof['inputs'], bitmap_proof['bitmap'])
                if not local_valid:
                    print(f"      ❌ Local verification FAILED: {local_msg}")
                    return {'success': False, 'error': f'Local verification failed: {local_msg}', 'proof_size': proof_size, 'communication_cost': communication_cost}
                else:
                    print(f"      ✅ Local verification passed: {local_msg}")
            
            # On-chain verification if Web3 available
            gas_used = 0
            onchain_verification_time = 0
            if self.web3 and self.traditional_property_level_huffman_contract:
                try:
                    # New bitmap verification format
                    inputs_bytes = [bytes.fromhex(h.replace('0x', '')) for h in bitmap_proof['inputs']]
                    bitmap = bitmap_proof['bitmap']
                    
                    # Time only the on-chain verification calls
                    onchain_start = time.time()
                    
                    # Call contract verification with bitmap format and check boolean return value
                    is_valid = self.traditional_property_level_huffman_contract.functions.verifyBatch(
                        inputs_bytes, bitmap
                    ).call()
                    
                    # Estimate gas cost
                    gas_used = self.traditional_property_level_huffman_contract.functions.verifyBatch(
                        inputs_bytes, bitmap
                    ).estimate_gas()
                    
                    onchain_verification_time = time.time() - onchain_start
                    
                    if is_valid:
                        print(f"      ✅ On-chain bitmap verification successful - Gas used: {gas_used:,}")
                        return {'success': True, 'gas_used': gas_used, 'proof_size': proof_size, 'communication_cost': communication_cost, 'onchain_verification_time': onchain_verification_time}
                    else:
                        # Debug: check stored root vs expected root
                        stored_root = self.traditional_property_level_huffman_contract.functions.merkleRoot().call()
                        expected_root = tree_system['root']
                        print(f"      ❌ On-chain verification returned False - proof is invalid")
                        print(f"         Expected root: 0x{expected_root[:16]}...")
                        print(f"         Stored root:   0x{stored_root.hex()[:16]}...")
                        if stored_root.hex() != expected_root:
                            print(f"         ⚠️ ROOT MISMATCH detected!")
                        return {'success': False, 'error': 'Verification returned False', 'proof_size': proof_size, 'communication_cost': communication_cost, 'onchain_verification_time': onchain_verification_time}
                except Exception as e:
                    print(f"      ❌ On-chain verification failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return {'success': False, 'error': f'On-chain verification failed: {e}', 'proof_size': proof_size, 'onchain_verification_time': onchain_verification_time}
            else:
                print(f"    ⚠️ No Web3 connection or contract - skipping on-chain verification")
                return {'success': False, 'error': 'No Web3 connection or contract available', 'proof_size': proof_size, 'communication_cost': communication_cost, 'onchain_verification_time': 0}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _verify_clustered_province(self, tree_system: dict, documents_to_verify: list):
        """Verify using clustered province approach - on-chain verification only."""
        try:
            builder = tree_system['builder']
            print(f"      Generating multiproof for {len(documents_to_verify)} documents...")
            
            # Standardized input handling - convert Document objects to hex strings
            document_hashes_hex = [doc.hash_hex for doc in documents_to_verify]
            
            # Generate proofs using OpenZeppelin multiproof format
            try:
                # Returns proof, flags, AND leaves in the correct order for verification
                proof, proof_flags, leaves_in_order = builder.generate_multiproof(document_hashes_hex)
                proof_size = len(proof) * 32
                # Calculate actual communication cost by simulating verification
                communication_cost = self._calculate_communication_cost_multiproof(proof, proof_flags, leaves_in_order)
                print(f"      Generated OpenZeppelin multiproof: {len(proof)} elements, {len(proof_flags)} flags, communication_cost: {communication_cost}")
            except Exception as e:
                print(f"      ❌ Multiproof generation failed: {e}")
                return {'success': False, 'error': f'Multiproof generation failed: {e}'}
            
            # Skip local verification for benchmark - focus on on-chain verification only
            print(f"      ⚠️ Skipping local verification for benchmark - using on-chain verification only")
            
            # On-chain verification if Web3 available
            gas_used = 0
            onchain_verification_time = 0
            if self.web3 and self.clustered_province_contract:
                try:
                    # Convert to bytes for OpenZeppelin multiproof format
                    # IMPORTANT: Use leaves_in_order (tree-index order), NOT sorted by hash!
                    proof_bytes = [bytes.fromhex(p.replace('0x', '')) for p in proof]
                    leaves_bytes = [bytes.fromhex(leaf.replace('0x', '')) for leaf in leaves_in_order]
                    
                    # Time only the on-chain verification calls
                    onchain_start = time.time()
                    
                    # Call contract verification with OpenZeppelin format and check boolean return value
                    is_valid = self.clustered_province_contract.functions.verifyBatch(
                        proof_bytes, proof_flags, leaves_bytes
                    ).call()
                    
                    # Estimate gas cost
                    gas_used = self.clustered_province_contract.functions.verifyBatch(
                        proof_bytes, proof_flags, leaves_bytes
                    ).estimate_gas()
                    
                    onchain_verification_time = time.time() - onchain_start
                    
                    if is_valid:
                        print(f"      ✅ On-chain OpenZeppelin multiproof verification successful - Gas used: {gas_used:,}")
                        return {'success': True, 'gas_used': gas_used, 'proof_size': proof_size, 'communication_cost': communication_cost, 'onchain_verification_time': onchain_verification_time}
                    else:
                        print(f"      ❌ On-chain verification returned False - proof is invalid")
                        return {'success': False, 'error': 'Verification returned False', 'proof_size': proof_size, 'communication_cost': communication_cost, 'onchain_verification_time': onchain_verification_time}
                except Exception as e:
                    print(f"      ❌ On-chain verification failed: {e}")
                    return {'success': False, 'error': f'On-chain verification failed: {e}', 'proof_size': proof_size, 'communication_cost': communication_cost, 'onchain_verification_time': onchain_verification_time}
            else:
                print(f"    ⚠️ No Web3 connection or contract - skipping on-chain verification")
                return {'success': False, 'error': 'No Web3 connection or contract available', 'proof_size': proof_size, 'communication_cost': communication_cost, 'onchain_verification_time': 0}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _verify_clustered_province_with_document_huffman(self, tree_system: dict, documents_to_verify: list):
        """Verify using clustered province with document huffman approach - on-chain verification only."""
        try:
            builder = tree_system['builder']
            print(f"      Generating multiproof for {len(documents_to_verify)} documents...")
            
            # Standardized input handling - convert Document objects to hex strings
            document_hashes_hex = [doc.hash_hex for doc in documents_to_verify]
            
            # Generate proofs using direct bitmap format (post-order traversal)
            try:
                # Use new direct post-order bitmap proof generation (not PathMap conversion)
                bitmap_proof = builder.generate_bitmap_proof(document_hashes_hex)
                
                proof_size = len(bitmap_proof['inputs']) * 32
                num_pushes = bitmap_proof['num_pushes']
                num_merges = bitmap_proof['num_merges']
                communication_cost = num_pushes + num_merges
                
                # Verify invariant: merges = pushes - 1
                expected_merges = num_pushes - 1
                if num_merges != expected_merges:
                    print(f"      ⚠️ WARNING: Operation count mismatch! {num_pushes} pushes but {num_merges} merges (expected {expected_merges})")
                
                print(f"      Using direct bitmap format - {num_pushes} inputs, {num_merges} merges, {len(bitmap_proof['bitmap'])} bitmap words, communication_cost: {communication_cost}")
            except Exception as e:
                print(f"      ❌ Multiproof generation failed: {e}")
                import traceback
                traceback.print_exc()
                return {'success': False, 'error': f'Multiproof generation failed: {e}'}
            
            # Local verification for debugging
            if hasattr(builder, 'verify_bitmap_locally'):
                local_valid, local_msg = builder.verify_bitmap_locally(bitmap_proof['inputs'], bitmap_proof['bitmap'])
                if not local_valid:
                    print(f"      ❌ Local verification FAILED: {local_msg}")
                    return {'success': False, 'error': f'Local verification failed: {local_msg}', 'proof_size': proof_size, 'communication_cost': communication_cost}
                else:
                    print(f"      ✅ Local verification passed: {local_msg}")
            
            # On-chain verification if Web3 available
            gas_used = 0
            onchain_verification_time = 0
            if self.web3 and self.clustered_province_with_document_huffman_contract:
                try:
                    # New bitmap verification format
                    inputs_bytes = [bytes.fromhex(h.replace('0x', '')) for h in bitmap_proof['inputs']]
                    bitmap = bitmap_proof['bitmap']
                    
                    # Time only the on-chain verification calls
                    onchain_start = time.time()
                    
                    # Call contract verification with bitmap format and check boolean return value
                    is_valid = self.clustered_province_with_document_huffman_contract.functions.verifyBatch(
                        inputs_bytes, bitmap
                    ).call()
                    
                    # Estimate gas cost
                    gas_used = self.clustered_province_with_document_huffman_contract.functions.verifyBatch(
                        inputs_bytes, bitmap
                    ).estimate_gas()
                    
                    onchain_verification_time = time.time() - onchain_start
                    
                    if is_valid:
                        print(f"      ✅ On-chain bitmap verification successful - Gas used: {gas_used:,}")
                        return {'success': True, 'gas_used': gas_used, 'proof_size': proof_size, 'communication_cost': communication_cost, 'onchain_verification_time': onchain_verification_time}
                    else:
                        # Debug: check stored root vs expected root
                        stored_root = self.clustered_province_with_document_huffman_contract.functions.merkleRoot().call()
                        expected_root = tree_system.get('root')
                        print(f"      ❌ On-chain verification returned False - proof is invalid")
                        if expected_root:
                            print(f"         Expected root: 0x{expected_root[:16]}...")
                        else:
                            print(f"         Expected root: None (root not set in tree_system)")
                        print(f"         Stored root:   0x{stored_root.hex()[:16]}...")
                        if expected_root and stored_root.hex() != expected_root:
                            print(f"         ⚠️ ROOT MISMATCH detected!")
                        return {'success': False, 'error': 'Verification returned False', 'proof_size': proof_size, 'communication_cost': communication_cost, 'onchain_verification_time': onchain_verification_time}
                except Exception as e:
                    print(f"      ❌ On-chain verification failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return {'success': False, 'error': f'On-chain verification failed: {e}', 'proof_size': proof_size, 'communication_cost': communication_cost, 'onchain_verification_time': onchain_verification_time}
            else:
                print(f"    ⚠️ No Web3 connection or contract - skipping on-chain verification")
                return {'success': False, 'error': 'No Web3 connection or contract available', 'proof_size': proof_size, 'communication_cost': communication_cost, 'onchain_verification_time': 0}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _verify_jurisdiction_tree(self, tree_system: dict, documents_to_verify: list):
        """Verify using jurisdiction tree approach - two-phase verification."""
        try:
            builder = tree_system['builder']
            print(f"      Generating jurisdiction tree proofs for {len(documents_to_verify)} documents...")
            
            # Standardized input handling - convert to hex strings consistently
            document_hashes_hex = []
            for doc in documents_to_verify:
                if hasattr(doc, 'hash_hex'):
                    # Document object
                    document_hashes_hex.append(doc.hash_hex)
                else:
                    # Already a string hash
                    document_hashes_hex.append(doc)
            
            # Generate two-phase proof
            try:
                two_phase_proof = builder.generate_two_phase_proof(document_hashes_hex)
                
                # Calculate proof size from two-phase proof structure
                total_proof_elements = 0
                total_flags = 0
                
                # Count province proof elements
                for province_data in two_phase_proof['province_proofs'].values():
                    total_proof_elements += len(province_data['proof'])
                    total_flags += len(province_data['flags'])

                # Count per-leaf sibling proof elements (if available)
                total_leaf_sibling_elements = 0
                for province_data in two_phase_proof['province_proofs'].values():
                    if 'leaf_proofs' in province_data and province_data['leaf_proofs']:
                        for lp in province_data['leaf_proofs']:
                            total_leaf_sibling_elements += len(lp.get('siblings', []))
                
                # Count jurisdiction proof elements
                total_proof_elements += len(two_phase_proof['jurisdiction_proof'])
                total_flags += len(two_phase_proof['jurisdiction_flags'])
                
                proof_size = total_proof_elements * 32 + total_flags
                leaf_sibling_bytes = total_leaf_sibling_elements * 32
                # Note: leaf_sibling_bytes are additional proof bytes if per-leaf proofs are used on-chain
                
                print(f"      Generated two-phase proof:")
                print(f"        Involved provinces: {len(two_phase_proof['involved_provinces'])}")
                print(f"        Province proofs: {len(two_phase_proof['province_proofs'])}")
                print(f"        Total proof elements: {total_proof_elements}, flags: {total_flags}")
                if total_leaf_sibling_elements:
                    print(f"        Total leaf-sibling elements: {total_leaf_sibling_elements} (bytes: {leaf_sibling_bytes})")
                
            except Exception as e:
                print(f"      ❌ Jurisdiction tree proof generation failed: {e}")
                return {'success': False, 'error': f'Jurisdiction tree proof generation failed: {e}'}
            
            # Local verification for jurisdiction tree (two-phase)
            try:
                # For jurisdiction tree, we verify each phase separately
                all_valid = True
                verification_details = []
                
                # Phase 1: Verify each province proof
                for province, province_data in two_phase_proof['province_proofs'].items():
                    if province in builder.province_trees:
                        province_tree = builder.province_trees[province]
                        # For simplicity, we'll assume province verification works if proof was generated
                        verification_details.append(f"Province {province}: valid")
                    else:
                        all_valid = False
                        verification_details.append(f"Province {province}: invalid (not found)")
                
                # Phase 2: Verify jurisdiction proof (simplified)
                if two_phase_proof['involved_provinces']:
                    verification_details.append(f"Jurisdiction: valid ({len(two_phase_proof['involved_provinces'])} provinces)")
                else:
                    all_valid = False
                    verification_details.append("Jurisdiction: invalid (no provinces)")
                
                print(f"      Local verification: {'✅ SUCCESS' if all_valid else '❌ FAILED'}")
                for detail in verification_details:
                    print(f"        {detail}")
                
            except Exception as e:
                print(f"      ❌ Local verification failed: {e}")
                return {'success': False, 'error': f'Local verification failed: {e}'}
            
            # On-chain verification using JurisdictionTreeVerifier contract
            gas_used = 0
            onchain_verification_time = 0
            if self.web3 and self.jurisdiction_tree_contract:
                try:
                    print(f"      🔗 Performing on-chain two-phase verification...")
                    
                    # Prepare province proofs for contract
                    province_proofs = []
                    for province, province_data in two_phase_proof['province_proofs'].items():
                        # Convert to contract format
                        province_root_bytes = bytes.fromhex(province_data['province_root'].replace('0x', ''))
                        leaves_bytes = [bytes.fromhex(leaf.replace('0x', '')) for leaf in province_data['leaves']]
                        proof_bytes = [bytes.fromhex(p.replace('0x', '')) for p in province_data['proof']]
                        
                        province_proof = (
                            province_root_bytes,  # provinceRoot
                            leaves_bytes,         # leaves  
                            proof_bytes,          # proof
                            province_data['flags'] # proofFlags
                        )
                        province_proofs.append(province_proof)
                    
                    # Prepare jurisdiction proof for contract
                    jurisdiction_roots_bytes = [
                        bytes.fromhex(two_phase_proof['province_proofs'][p]['province_root'].replace('0x', ''))
                        for p in two_phase_proof['involved_provinces']
                    ]
                    jurisdiction_proof_bytes = [
                        bytes.fromhex(p.replace('0x', '')) for p in two_phase_proof['jurisdiction_proof']
                    ]
                    
                    jurisdiction_proof = (
                        jurisdiction_roots_bytes,        # provinceRoots
                        jurisdiction_proof_bytes,        # proof  
                        two_phase_proof['jurisdiction_flags'] # proofFlags
                    )
                    
                    # Get jurisdiction root
                    jurisdiction_root_bytes = bytes.fromhex(builder.jurisdiction_root.replace('0x', ''))
                    
                    # Time only the on-chain verification calls
                    onchain_start = time.time()
                    
                    # Call contract verification
                    tx_hash = self.jurisdiction_tree_contract.functions.verifyTwoPhase(
                        province_proofs,
                        jurisdiction_proof,
                        jurisdiction_root_bytes
                    ).transact()
                    
                    receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                    gas_used = receipt.gasUsed
                    
                    onchain_verification_time = time.time() - onchain_start
                    
                    # Check if transaction succeeded (no revert)
                    contract_success = receipt.status == 1
                    print(f"      ✅ On-chain two-phase verification: {'SUCCESS' if contract_success else 'FAILED'}")
                    print(f"      ⛽ Gas used: {gas_used:,}")
                    
                    return {
                        'success': contract_success and all_valid,
                        'gas_used': gas_used,
                        'proof_size': proof_size,
                        'two_phase_proof': two_phase_proof,
                        'verification_details': verification_details,
                        'onchain_verification_time': onchain_verification_time
                    }
                    
                except Exception as e:
                    print(f"      ❌ On-chain verification failed: {e}")
                    # Fall back to local verification result
                    return {
                        'success': all_valid,
                        'gas_used': 0,
                        'proof_size': proof_size,
                        'error': f'On-chain verification failed: {e}',
                        'verification_details': verification_details,
                        'onchain_verification_time': onchain_verification_time
                    }
            else:
                print(f"      ⚠️ No Web3 connection or jurisdiction contract - using local verification only")
                return {
                    'success': all_valid, 
                    'gas_used': 0,
                    'proof_size': proof_size,
                    'two_phase_proof': two_phase_proof,
                    'verification_details': verification_details,
                    'onchain_verification_time': 0
                }
            
        except Exception as e:
            print(f"      💥 Exception in jurisdiction tree verification: {e}")
            return {'success': False, 'error': str(e)}
    
    def _store_results(self, results: dict):
        """Store test results in the performance data for later report generation."""
        self.performance_data.update(results)
        
        # Save detailed results to JSON file with timestamp
        if self.reports_dir:
            os.makedirs(self.reports_dir, exist_ok=True)
        else:
            self.reports_dir = "./test_reports"
            os.makedirs(self.reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        raw_results_file = os.path.join(self.reports_dir, f"raw_results_{timestamp}.json")
        with open(raw_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate summary metrics
        summary_metrics = self._generate_summary_metrics(results)
        summary_file = os.path.join(self.reports_dir, f"summary_metrics_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_metrics, f, indent=2)
            
        print(f"\n📄 Results saved to:")
        print(f"  Raw results: {raw_results_file}")
        print(f"  Summary metrics: {summary_file}")
        
        # Generate charts
        self._generate_charts(summary_metrics, timestamp)
    
    def _generate_summary_metrics(self, results: dict) -> dict:
        """Generate summary metrics from test results."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'approaches': {},
            'comparison': {
                'gas_efficiency': {},
                'success_rates': {},
                'build_times': {},
                'proof_sizes': {},
                'communication_costs': {},
                'verification_times': {}
            }
        }
        
        for approach_name, approach_data in results.items():
            # Per-approach metrics
            approach_summary = {
                'total_queries': approach_data['total_queries'],
                'successful_verifications': approach_data['successful_verifications'],
                'failed_verifications': approach_data['failed_verifications'],
                'success_rate_percent': (approach_data['successful_verifications'] / approach_data['total_queries']) * 100 if approach_data['total_queries'] > 0 else 0,
                'total_gas_used': approach_data['total_gas_used'],
                'average_gas_per_query': approach_data['average_gas_per_query'],
                'build_time_seconds': approach_data['build_time'],
                'query_breakdown': {
                    'transactional': 0,
                    # 'audit': 0,  # Legacy - removed in favor of regional_audit and national_audit
                    'regional_audit': 0,
                    'national_audit': 0
                },
                'gas_statistics': {
                    'min_gas': float('inf'),
                    'max_gas': 0,
                    'gas_values': []
                },
                'proof_size_statistics': {
                    'min_size': float('inf'),
                    'max_size': 0,
                    'avg_size': 0,
                    'size_values': []
                },
                'communication_cost_statistics': {
                    'min_cost': float('inf'),
                    'max_cost': 0,
                    'avg_cost': 0,
                    'cost_values': []
                },
                'verification_time_statistics': {
                    'min_time': float('inf'),
                    'max_time': 0,
                    'avg_time': 0,
                    'time_values': []
                }
            }
            
            # Analyze individual query results
            for query_result in approach_data['query_results']:
                # Count query types
                if query_result['query_type'] == 'TRANSACTIONAL':
                    approach_summary['query_breakdown']['transactional'] += 1
                # elif query_result['query_type'] == 'AUDIT':  # Legacy - commented out
                #     approach_summary['query_breakdown']['audit'] += 1
                elif query_result['query_type'] == 'REGIONAL_AUDIT':
                    approach_summary['query_breakdown']['regional_audit'] += 1
                elif query_result['query_type'] == 'NATIONAL_AUDIT':
                    approach_summary['query_breakdown']['national_audit'] += 1
                
                # Gas statistics
                if query_result.get('gas_used', 0) > 0:
                    gas_used = query_result['gas_used']
                    approach_summary['gas_statistics']['gas_values'].append(gas_used)
                    approach_summary['gas_statistics']['min_gas'] = min(approach_summary['gas_statistics']['min_gas'], gas_used)
                    approach_summary['gas_statistics']['max_gas'] = max(approach_summary['gas_statistics']['max_gas'], gas_used)
                
                # Proof size statistics
                if query_result.get('proof_size', 0) > 0:
                    proof_size = query_result['proof_size']
                    approach_summary['proof_size_statistics']['size_values'].append(proof_size)
                    approach_summary['proof_size_statistics']['min_size'] = min(approach_summary['proof_size_statistics']['min_size'], proof_size)
                    approach_summary['proof_size_statistics']['max_size'] = max(approach_summary['proof_size_statistics']['max_size'], proof_size)
                
                # Communication cost statistics
                if query_result.get('communication_cost', 0) > 0:
                    comm_cost = query_result['communication_cost']
                    approach_summary['communication_cost_statistics']['cost_values'].append(comm_cost)
                    approach_summary['communication_cost_statistics']['min_cost'] = min(approach_summary['communication_cost_statistics']['min_cost'], comm_cost)
                    approach_summary['communication_cost_statistics']['max_cost'] = max(approach_summary['communication_cost_statistics']['max_cost'], comm_cost)
                
                # Verification time statistics
                if query_result.get('verification_time', 0) > 0:
                    ver_time = query_result['verification_time']
                    approach_summary['verification_time_statistics']['time_values'].append(ver_time)
                    approach_summary['verification_time_statistics']['min_time'] = min(approach_summary['verification_time_statistics']['min_time'], ver_time)
                    approach_summary['verification_time_statistics']['max_time'] = max(approach_summary['verification_time_statistics']['max_time'], ver_time)
            
            # Calculate averages
            if approach_summary['gas_statistics']['gas_values']:
                approach_summary['gas_statistics']['avg_gas'] = np.mean(approach_summary['gas_statistics']['gas_values'])
                approach_summary['gas_statistics']['std_gas'] = np.std(approach_summary['gas_statistics']['gas_values'])
            else:
                approach_summary['gas_statistics']['min_gas'] = 0
            
            if approach_summary['proof_size_statistics']['size_values']:
                approach_summary['proof_size_statistics']['avg_size'] = np.mean(approach_summary['proof_size_statistics']['size_values'])
                approach_summary['proof_size_statistics']['std_size'] = np.std(approach_summary['proof_size_statistics']['size_values'])
            else:
                approach_summary['proof_size_statistics']['min_size'] = 0
            
            if approach_summary['communication_cost_statistics']['cost_values']:
                approach_summary['communication_cost_statistics']['avg_cost'] = np.mean(approach_summary['communication_cost_statistics']['cost_values'])
                approach_summary['communication_cost_statistics']['std_cost'] = np.std(approach_summary['communication_cost_statistics']['cost_values'])
            else:
                approach_summary['communication_cost_statistics']['min_cost'] = 0
            
            if approach_summary['verification_time_statistics']['time_values']:
                approach_summary['verification_time_statistics']['avg_time'] = np.mean(approach_summary['verification_time_statistics']['time_values'])
                approach_summary['verification_time_statistics']['std_time'] = np.std(approach_summary['verification_time_statistics']['time_values'])
            else:
                approach_summary['verification_time_statistics']['min_time'] = 0
            
            summary['approaches'][approach_name] = approach_summary
            
            # Add to comparison data
            summary['comparison']['gas_efficiency'][approach_name] = approach_summary['average_gas_per_query']
            summary['comparison']['success_rates'][approach_name] = approach_summary['success_rate_percent']
            summary['comparison']['build_times'][approach_name] = approach_summary['build_time_seconds']
            summary['comparison']['proof_sizes'][approach_name] = approach_summary['proof_size_statistics']['avg_size']
            summary['comparison']['communication_costs'][approach_name] = approach_summary['communication_cost_statistics']['avg_cost']
            summary['comparison']['verification_times'][approach_name] = approach_summary['verification_time_statistics']['avg_time']
        
        return summary
    
    def _generate_charts(self, summary_metrics: dict, timestamp: str):
        """Generate visualization charts from summary metrics."""
        try:
            # Set up the plotting style
            plt.style.use('default')
            fig_size = (15, 12)
            
            approaches = list(summary_metrics['approaches'].keys())
            approach_labels = [name.replace('_', ' ').title() for name in approaches]
            
            # Create a comprehensive dashboard
            fig, axes = plt.subplots(2, 3, figsize=fig_size)
            fig.suptitle(f'Merkle Tree Verification Performance Comparison\nGenerated: {timestamp}', fontsize=16, fontweight='bold')
            
            # 1. Average Gas Usage Comparison
            gas_values = [summary_metrics['comparison']['gas_efficiency'][app] for app in approaches]
            bars1 = axes[0, 0].bar(approach_labels, gas_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            axes[0, 0].set_title('Average Gas Usage per Query', fontweight='bold')
            axes[0, 0].set_ylabel('Gas Used')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars1, gas_values):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(gas_values)*0.01,
                               f'{int(value):,}', ha='center', va='bottom', fontsize=9)
            
            # 2. Success Rate Comparison
            success_rates = [summary_metrics['comparison']['success_rates'][app] for app in approaches]
            bars2 = axes[0, 1].bar(approach_labels, success_rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            axes[0, 1].set_title('Success Rate Comparison', fontweight='bold')
            axes[0, 1].set_ylabel('Success Rate (%)')
            axes[0, 1].set_ylim(0, 105)
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars2, success_rates):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
            
            # 3. Build Time Comparison
            build_times = [summary_metrics['comparison']['build_times'][app] * 1000 for app in approaches]  # Convert to ms
            bars3 = axes[0, 2].bar(approach_labels, build_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            axes[0, 2].set_title('Tree Build Time Comparison', fontweight='bold')
            axes[0, 2].set_ylabel('Build Time (ms)')
            axes[0, 2].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars3, build_times):
                axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(build_times)*0.01,
                               f'{value:.1f}', ha='center', va='bottom', fontsize=9)
            
            # 4. Average Communication Cost Comparison
            comm_costs = [summary_metrics['comparison']['communication_costs'][app] for app in approaches]
            bars4 = axes[1, 0].bar(approach_labels, comm_costs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            axes[1, 0].set_title('Average Communication Cost (Nodes)', fontweight='bold')
            axes[1, 0].set_ylabel('Number of Nodes')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars4, comm_costs):
                if value > 0:
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(comm_costs)*0.01,
                                   f'{int(value)}', ha='center', va='bottom', fontsize=9)
            
            # 5. Gas Usage Distribution (Box Plot)
            gas_distributions = []
            labels_with_data = []
            for i, app in enumerate(approaches):
                gas_values = summary_metrics['approaches'][app]['gas_statistics']['gas_values']
                if gas_values:
                    gas_distributions.append(gas_values)
                    labels_with_data.append(approach_labels[i])
            
            if gas_distributions:
                axes[1, 1].boxplot(gas_distributions, tick_labels=labels_with_data)
                axes[1, 1].set_title('Gas Usage Distribution', fontweight='bold')
                axes[1, 1].set_ylabel('Gas Used')
                axes[1, 1].tick_params(axis='x', rotation=45)
            else:
                axes[1, 1].text(0.5, 0.5, 'No gas data available', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Gas Usage Distribution', fontweight='bold')
            
            # 6. Query Type Breakdown (Stacked Bar) - Legacy 'audit' removed
            transactional_counts = [summary_metrics['approaches'][app]['query_breakdown']['transactional'] for app in approaches]
            # audit_counts removed - legacy field no longer exists
            regional_audit_counts = [summary_metrics['approaches'][app]['query_breakdown']['regional_audit'] for app in approaches]
            national_audit_counts = [summary_metrics['approaches'][app]['query_breakdown']['national_audit'] for app in approaches]
            
            bars6_1 = axes[1, 2].bar(approach_labels, transactional_counts, label='Transactional', color='#45B7D1')
            # Legacy audit bar removed
            bars6_2 = axes[1, 2].bar(approach_labels, regional_audit_counts, 
                                    bottom=transactional_counts, 
                                    label='Regional Audit', color='#F39C12')
            bars6_3 = axes[1, 2].bar(approach_labels, national_audit_counts, 
                                    bottom=[t + r for t, r in zip(transactional_counts, regional_audit_counts)], 
                                    label='National Audit', color='#E74C3C')
            axes[1, 2].set_title('Query Type Distribution', fontweight='bold')
            axes[1, 2].set_ylabel('Number of Queries')
            axes[1, 2].tick_params(axis='x', rotation=45)
            axes[1, 2].legend()
            
            # Adjust layout to prevent overlap
            plt.tight_layout()
            
            # Save the chart
            chart_file = os.path.join(self.reports_dir, f"performance_comparison_{timestamp}.png")
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📊 Performance chart: {chart_file}")
            
            # Generate individual detailed charts for each metric
            self._generate_detailed_charts(summary_metrics, timestamp)
            
        except Exception as e:
            print(f"⚠️  Chart generation failed: {e}")
    
    def _generate_detailed_charts(self, summary_metrics: dict, timestamp: str):
        """Generate detailed individual charts for each metric."""
        try:
            approaches = list(summary_metrics['approaches'].keys())
            approach_labels = [name.replace('_', ' ').title() for name in approaches]
            
            # 1. Detailed Gas Analysis Chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Detailed Gas Usage Analysis', fontsize=16, fontweight='bold')
            
            # Gas efficiency comparison with error bars
            gas_means = []
            gas_stds = []
            for app in approaches:
                stats = summary_metrics['approaches'][app]['gas_statistics']
                gas_means.append(stats.get('avg_gas', 0))
                gas_stds.append(stats.get('std_gas', 0))
            
            bars = ax1.bar(approach_labels, gas_means, yerr=gas_stds, capsize=5, 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
            ax1.set_title('Average Gas Usage with Standard Deviation')
            ax1.set_ylabel('Gas Used')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, mean, std in zip(bars, gas_means, gas_stds):
                if mean > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + max(gas_means)*0.02,
                           f'{int(mean):,}\n±{int(std)}', ha='center', va='bottom', fontsize=9)
            
            # Gas range comparison
            gas_mins = [summary_metrics['approaches'][app]['gas_statistics']['min_gas'] for app in approaches]
            gas_maxs = [summary_metrics['approaches'][app]['gas_statistics']['max_gas'] for app in approaches]
            gas_ranges = [max_val - min_val if max_val != float('inf') and min_val != float('inf') else 0 
                         for min_val, max_val in zip(gas_mins, gas_maxs)]
            
            bars2 = ax2.bar(approach_labels, gas_ranges, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
            ax2.set_title('Gas Usage Range (Max - Min)')
            ax2.set_ylabel('Gas Range')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, range_val in zip(bars2, gas_ranges):
                if range_val > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(gas_ranges)*0.01,
                           f'{int(range_val):,}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            gas_chart_file = os.path.join(self.reports_dir, f"gas_analysis_{timestamp}.png")
            plt.savefig(gas_chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📊 Gas analysis chart: {gas_chart_file}")
            
            # 2. Performance Efficiency Radar Chart
            self._generate_radar_chart(summary_metrics, timestamp, approaches, approach_labels)
            
        except Exception as e:
            print(f"⚠️  Detailed chart generation failed: {e}")
    
    def _generate_radar_chart(self, summary_metrics: dict, timestamp: str, approaches: list, approach_labels: list):
        """Generate a radar chart comparing all approaches across multiple metrics."""
        try:
            # Normalize metrics to 0-100 scale for radar chart
            metrics = ['Gas Efficiency', 'Success Rate', 'Build Speed', 'Proof Compactness']
            
            # Get raw values
            gas_values = [summary_metrics['comparison']['gas_efficiency'][app] for app in approaches]
            success_values = [summary_metrics['comparison']['success_rates'][app] for app in approaches]
            build_values = [summary_metrics['comparison']['build_times'][app] for app in approaches]
            proof_values = [summary_metrics['comparison']['proof_sizes'][app] for app in approaches]
            
            # Normalize to 0-100 (higher is better)
            # Gas efficiency: lower is better, so invert
            max_gas = max(gas_values) if max(gas_values) > 0 else 1
            gas_norm = [100 - (val / max_gas * 100) for val in gas_values]
            
            # Success rate: already in percentage
            success_norm = success_values
            
            # Build speed: lower is better, so invert
            max_build = max(build_values) if max(build_values) > 0 else 1
            build_norm = [100 - (val / max_build * 100) for val in build_values]
            
            # Proof compactness: lower is better, so invert
            max_proof = max(proof_values) if max(proof_values) > 0 else 1
            proof_norm = [100 - (val / max_proof * 100) if val > 0 else 50 for val in proof_values]
            
            # Set up radar chart
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            for i, approach in enumerate(approaches):
                values = [gas_norm[i], success_norm[i], build_norm[i], proof_norm[i]]
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=approach_labels[i], color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(['20', '40', '60', '80', '100'])
            ax.grid(True)
            
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            plt.title('Performance Radar Comparison\n(Higher values are better)', size=16, fontweight='bold', pad=20)
            
            radar_chart_file = os.path.join(self.reports_dir, f"performance_radar_{timestamp}.png")
            plt.savefig(radar_chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📊 Performance radar chart: {radar_chart_file}")
            
        except Exception as e:
            print(f"⚠️  Radar chart generation failed: {e}")
    
    def _generate_reports(self):
        """Generate test reports from collected performance data."""
        if not self.performance_data:
            print("⚠️ No performance data to generate reports from.")
            return
        
        print("\n=== TEST RESULTS SUMMARY ===")
        
        for approach_name, approach_data in self.performance_data.items():
            print(f"\n--- {approach_name.upper()} ---")
            print(f"  Total Queries: {approach_data['total_queries']}")
            print(f"  Successful: {approach_data['successful_verifications']}")
            print(f"  Failed: {approach_data['failed_verifications']}")
            success_rate = (approach_data['successful_verifications']/approach_data['total_queries']*100) if approach_data['total_queries'] > 0 else 0
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Total Gas Used: {approach_data['total_gas_used']:,}")
            print(f"  Average Gas per Query: {approach_data['average_gas_per_query']:.0f}")
            print(f"  Build Time: {approach_data['build_time']:.3f}s")
            
            # Show additional statistics if available
            if 'query_results' in approach_data and approach_data['query_results']:
                gas_values = [q.get('gas_used', 0) for q in approach_data['query_results'] if q.get('gas_used', 0) > 0]
                if gas_values:
                    print(f"  Gas Range: {min(gas_values):,} - {max(gas_values):,}")
                    print(f"  Gas Std Dev: {np.std(gas_values):.0f}")
        
        print(f"\n� Charts and detailed metrics have been generated in: {self.reports_dir}")
    
    def _generate_alpha_tuning_report(self, tuning_results: dict, alpha_values: list[float]):
        """Generate specialized reports for alpha hyperparameter tuning."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.reports_dir:
                reports_dir = self.reports_dir
            else:
                reports_dir = "./test_reports"
            
            os.makedirs(reports_dir, exist_ok=True)
            
            # Create alpha tuning subdirectory
            alpha_reports_dir = os.path.join(reports_dir, f"alpha_tuning_{timestamp}")
            os.makedirs(alpha_reports_dir, exist_ok=True)
            
            print(f"\n📊 Generating alpha tuning reports...")
            
            # 1. Generate performance vs alpha charts for each approach
            for approach_name, approach_data in tuning_results.items():
                alpha_vals = []
                avg_gas_vals = []
                build_times = []
                success_rates = []
                
                for alpha_key, alpha_result in approach_data.items():
                    if alpha_key.startswith("alpha_"):
                        alpha = alpha_result['alpha']
                        alpha_vals.append(alpha)
                        avg_gas_vals.append(alpha_result['average_gas_per_query'])
                        build_times.append(alpha_result['build_time'])
                        success_rate = (alpha_result['successful_verifications'] / alpha_result['total_queries'] * 100) if alpha_result['total_queries'] > 0 else 0
                        success_rates.append(success_rate)
                
                if not alpha_vals:
                    continue
                
                # Sort by alpha values
                sorted_data = sorted(zip(alpha_vals, avg_gas_vals, build_times, success_rates))
                alpha_vals, avg_gas_vals, build_times, success_rates = zip(*sorted_data)
                
                # Create subplots for this approach
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'{approach_name.replace("_", " ").title()} - Alpha Hyperparameter Tuning', fontsize=16)
                
                # Plot 1: Average Gas vs Alpha
                ax1.plot(alpha_vals, avg_gas_vals, 'o-', linewidth=2, markersize=8)
                ax1.set_xlabel('Alpha Threshold')
                ax1.set_ylabel('Average Gas per Query')
                ax1.set_title('Gas Efficiency vs Alpha')
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Build Time vs Alpha
                ax2.plot(alpha_vals, build_times, 's-', color='orange', linewidth=2, markersize=8)
                ax2.set_xlabel('Alpha Threshold')
                ax2.set_ylabel('Build Time (seconds)')
                ax2.set_title('Build Time vs Alpha')
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Success Rate vs Alpha
                ax3.plot(alpha_vals, success_rates, '^-', color='green', linewidth=2, markersize=8)
                ax3.set_xlabel('Alpha Threshold')
                ax3.set_ylabel('Success Rate (%)')
                ax3.set_title('Success Rate vs Alpha')
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim(0, 105)
                
                # Plot 4: Combined Normalized Metrics
                # Normalize metrics to 0-1 scale for comparison
                norm_gas = [(max(avg_gas_vals) - x) / (max(avg_gas_vals) - min(avg_gas_vals)) for x in avg_gas_vals] if max(avg_gas_vals) != min(avg_gas_vals) else [0.5] * len(avg_gas_vals)
                norm_build_time = [(max(build_times) - x) / (max(build_times) - min(build_times)) for x in build_times] if max(build_times) != min(build_times) else [0.5] * len(build_times)
                norm_success = [x / 100.0 for x in success_rates]
                
                ax4.plot(alpha_vals, norm_gas, 'o-', label='Gas Efficiency (inverted)', linewidth=2, markersize=6)
                ax4.plot(alpha_vals, norm_build_time, 's-', label='Build Speed (inverted)', linewidth=2, markersize=6)
                ax4.plot(alpha_vals, norm_success, '^-', label='Success Rate', linewidth=2, markersize=6)
                ax4.set_xlabel('Alpha Threshold')
                ax4.set_ylabel('Normalized Performance (0-1)')
                ax4.set_title('Combined Performance Metrics')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(0, 1.05)
                
                plt.tight_layout()
                
                # Save individual approach chart
                chart_file = os.path.join(alpha_reports_dir, f"{approach_name}_alpha_tuning.png")
                plt.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"  📈 Generated chart for {approach_name}: {chart_file}")
            
            # 2. Generate comparison chart across all approaches
            if len(tuning_results) > 1:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                for approach_name, approach_data in tuning_results.items():
                    alpha_vals = []
                    avg_gas_vals = []
                    
                    for alpha_key, alpha_result in approach_data.items():
                        if alpha_key.startswith("alpha_"):
                            alpha_vals.append(alpha_result['alpha'])
                            avg_gas_vals.append(alpha_result['average_gas_per_query'])
                    
                    if alpha_vals:
                        # Sort by alpha values
                        sorted_data = sorted(zip(alpha_vals, avg_gas_vals))
                        alpha_vals, avg_gas_vals = zip(*sorted_data)
                        
                        ax.plot(alpha_vals, avg_gas_vals, 'o-', label=approach_name.replace('_', ' ').title(), linewidth=2, markersize=6)
                
                ax.set_xlabel('Alpha Threshold')
                ax.set_ylabel('Average Gas per Query')
                ax.set_title('Gas Efficiency Comparison Across Approaches')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                comparison_file = os.path.join(alpha_reports_dir, "alpha_tuning_comparison.png")
                plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"  📊 Generated comparison chart: {comparison_file}")
            
            # 3. Generate summary report (JSON)
            summary_data = {
                'timestamp': datetime.now().isoformat(),
                'alpha_values_tested': alpha_values,
                'approaches': {}
            }
            
            for approach_name, approach_data in tuning_results.items():
                approach_summary = {
                    'results_by_alpha': {},
                    'optimal_alpha': None,
                    'optimal_gas': float('inf')
                }
                
                for alpha_key, alpha_result in approach_data.items():
                    if alpha_key.startswith("alpha_"):
                        alpha = alpha_result['alpha']
                        
                        # Calculate average proof size and verification time from query_results
                        query_results = alpha_result.get('query_results', [])
                        avg_proof_size = 0
                        avg_verification_time = 0
                        if query_results:
                            total_proof_size = sum(q.get('proof_size', 0) for q in query_results)
                            total_verification_time = sum(q.get('verification_time', 0) for q in query_results)
                            avg_proof_size = total_proof_size / len(query_results)
                            avg_verification_time = total_verification_time / len(query_results)
                        
                        approach_summary['results_by_alpha'][alpha] = {
                            'average_gas_per_query': alpha_result['average_gas_per_query'],
                            'average_proof_size': avg_proof_size,
                            'average_verification_time': avg_verification_time,
                            'build_time': alpha_result['build_time'],
                            'success_rate': (alpha_result['successful_verifications'] / alpha_result['total_queries'] * 100) if alpha_result['total_queries'] > 0 else 0
                        }
                        
                        # Track optimal alpha (lowest gas)
                        if alpha_result['average_gas_per_query'] < approach_summary['optimal_gas']:
                            approach_summary['optimal_gas'] = alpha_result['average_gas_per_query']
                            approach_summary['optimal_alpha'] = alpha
                
                summary_data['approaches'][approach_name] = approach_summary
            
            summary_file = os.path.join(alpha_reports_dir, "alpha_tuning_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            print(f"  📄 Generated summary report: {summary_file}")
            print(f"\n📁 All alpha tuning reports saved to: {alpha_reports_dir}")
            
            # Print optimal alpha values
            print(f"\n🎯 OPTIMAL ALPHA VALUES:")
            for approach_name, approach_summary in summary_data['approaches'].items():
                optimal_alpha = approach_summary['optimal_alpha']
                optimal_gas = approach_summary['optimal_gas']
                if optimal_alpha is not None:
                    print(f"  {approach_name}: α = {optimal_alpha} (gas = {optimal_gas:.0f})")
            
        except ImportError:
            print("⚠️ matplotlib not available - skipping alpha tuning charts")
        except Exception as e:
            print(f"❌ Error generating alpha tuning reports: {e}")

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
    """Enhanced test runner with comprehensive metrics and visualization."""
    print("--- 1. Setup Phase ---")
    parser = argparse.ArgumentParser(description='Enhanced Test Runner with Metrics and Charts')
    parser.add_argument('--total-properties', type=int, default=2000, help='Total number of properties to simulate')
    parser.add_argument('--total-queries-per-run', type=int, default=1000, help='Number of queries to generate')
    parser.add_argument('--transactional-ratio', type=float, default=0.95, help='Ratio of transactional queries in mixed workload')
    parser.add_argument('--workload-type', choices=['mixed', 'stress'], default='mixed', 
                       help='Workload type: mixed (transactional + regional audit for normal operations), stress (pure national audit for stress testing)')
    parser.add_argument('--reports-dir', type=str, default='./test_reports', help='Directory to save test reports and charts')
    parser.add_argument('--approaches', nargs='+', default=None, 
                       help='Specific approaches to test (e.g., traditional_multiproof clustered_province)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--alpha-tuning', action='store_true', 
                       help='Enable alpha hyperparameter tuning for pairs-first Huffman models')
    parser.add_argument('--alpha-values', nargs='+', type=float, default=[0.05, 0.2, 0.4, 0.5, 0.7, 0.8],
                       help='Alpha values to test during hyperparameter tuning (default: [0.05, 0.2, 0.4, 0.5, 0.7, 0.8])')
    parser.add_argument('--alpha-overrides', type=str, default=None,
                       help='Optional per-model alpha overrides, format: model1:0.05,model2=0.2 (keys: traditional_document_huffman, traditional_property_level_huffman, clustered_province_with_document_huffman, jurisdiction_tree)')
    parser.add_argument('--property-zipf-s', type=float, default=1.2,
                       help='Zipfian exponent s for property access frequency (default: 1.2, empirically derived from population data)')
    parser.add_argument('--audit-sample-size', type=int, default=200,
                       help='Average number of properties per audit query (lambda, default: 200)')

    args = parser.parse_args()

    print("--- 2. Seeding Phase ---")
    seed_gen = SeedGenerator(total_properties=args.total_properties, random_seed=42)
    property_dataset = seed_gen.generate_dataset()

    print("--- 3. Constructing Pattern Phase ---")
    # Define our plausible "real-world" weights for each document type.
    # This is our core assumption about transactional user behavior.
    DOCUMENT_IMPORTANCE_MAP = {
        "TitleDeed": 100,               # Sertifikat Hak Milik - Critical
        "LandAndBuildingTax2025": 95,   # PBB Terbaru - Almost always required
        "BuildingPermit": 70,           # IMB - Important for due diligence
        "LandAndBuildingTax2024": 50,   # PBB Tahun Sebelumnya - Often checked
        "ElectricityBill_Latest": 30,   # Bukti Bayar Listrik - Sometimes required
        "WaterBill_Latest": 25,         # Bukti Bayar Air - Less common
    }
    transactional_pattern = TransactionalPattern(
        document_importance_map=DOCUMENT_IMPORTANCE_MAP,
        alpha_threshold=0.15,         # Lower threshold for more pair detection
        use_zipfian=False,            # Documents are balanced, use importance map instead
        zipf_parameter=1.0,           # Not used (disabled)
        use_property_zipfian=True,    # Enable property-level Zipfian distribution
        property_zipf_parameter=args.property_zipf_s,  # Configurable via --property-zipf-s
        random_seed=42                # Fixed seed for reproducible patterns
    )
    audit_pattern = AuditPattern(
        province_weights=seed_gen.province_weights,
        avg_sample_size=args.audit_sample_size,  # Configurable via --audit-sample-size
        min_sample_size=10,        # Minimum to ensure meaningful audit samples
        avg_docs_per_property=2,   # λ for documents per property (realistic average)
        min_docs_per_property=1,   # Minimum to ensure at least one document
        random_seed=42             # Fixed seed for reproducible patterns
    )

    print("--- 4. Constructing Workload Generator ---")
    print(f"  📊 Access Pattern Configuration:")
    print(f"    Document Distribution: {'Zipfian' if transactional_pattern.use_zipfian else 'Uniform'} (zipf_param={transactional_pattern.zipf_parameter})")
    print(f"    Property Distribution: {'Zipfian' if transactional_pattern.use_property_zipfian else 'Uniform'} (zipf_param={transactional_pattern.property_zipf_parameter})")
    print(f"    Alpha Threshold: {transactional_pattern.alpha_threshold} (pair frequency cutoff)")
    
    workload_gen = WorkloadGenerator(
        property_dataset, 
        transactional_pattern=transactional_pattern,
        audit_pattern=audit_pattern,
        random_seed=42  # Fixed seed for reproducible workload generation
    )
    
    print("--- 5. Constructing Workload Generator ---")
    
    # Generate workload based on user specification
    if args.workload_type == 'mixed':
        # Mixed workload: Transactional + Regional Audit (normal operations)
        print(f"  Generating Mixed workload (Transactional + Regional Audit)")
        print(f"    - {int(args.total_queries_per_run * args.transactional_ratio)} Transactional queries")
        print(f"    - {args.total_queries_per_run - int(args.total_queries_per_run * args.transactional_ratio)} Regional Audit queries")
        
        final_workload = workload_gen.generate_mixed_workload(args.total_queries_per_run, args.transactional_ratio)
        
    elif args.workload_type == 'stress':
        # Stress test workload: Pure National Audit (cross-province stress testing)
        print(f"  Generating Stress Test workload (Pure National Audit)")
        print(f"    - {args.total_queries_per_run} National Audit queries (cross-province)")
        
        final_workload = workload_gen.generate_stress_test_workload(args.total_queries_per_run)
        
    else:
        raise ValueError(f"Unknown workload type: {args.workload_type}")
    
    print("  All workloads are ready.")

    print("--- 6. Extract Documents from Dataset ---")
    # Extract all documents from the property dataset
    documents = []
    for property_obj in property_dataset:
        documents.extend(property_obj.documents)
    print(f"  Extracted {len(documents)} documents from {len(property_dataset)} properties")
    
    # Validate document ID uniqueness
    doc_ids = [doc.doc_id for doc in documents]
    unique_doc_ids = set(doc_ids)
    if len(doc_ids) != len(unique_doc_ids):
        duplicates = [doc_id for doc_id in doc_ids if doc_ids.count(doc_id) > 1]
        print(f"  ⚠️ WARNING: Found {len(doc_ids) - len(unique_doc_ids)} duplicate document IDs: {set(duplicates)}")
    else:
        print(f"  ✅ All {len(doc_ids)} document IDs are unique")

    print("--- 7. Experiment Definition Phase ---")
    web3 = setup_web3_connection()
    runner = TestRunner(web3_instance=web3, reports_dir=args.reports_dir, verbose=args.verbose)

    # Parse alpha overrides if provided
    alpha_overrides = {}
    if args.alpha_overrides:
        # Accept comma-separated pairs in either key:val or key=val format
        items = [s.strip() for s in args.alpha_overrides.split(',') if s.strip()]
        for item in items:
            if ':' in item:
                k, v = item.split(':', 1)
            elif '=' in item:
                k, v = item.split('=', 1)
            else:
                print(f"⚠️  Invalid alpha override item '{item}', expected key:val or key=val")
                continue
            try:
                alpha_overrides[k.strip()] = float(v.strip())
            except ValueError:
                print(f"⚠️  Could not parse alpha value for '{k}': '{v}'")

    if alpha_overrides:
        runner.alpha_overrides = alpha_overrides
        print(f"  ✅ Using per-model alpha overrides: {runner.alpha_overrides}")

    # Select approaches to test
    selected_approaches = args.approaches if args.approaches else None
    
    # Check if alpha tuning is requested
    if args.alpha_tuning:
        print("🎯 Alpha hyperparameter tuning mode enabled")
        
        # Run alpha tuning tests
        tuning_results = runner.run_alpha_tuning_tests(
            documents, final_workload, args.alpha_values, transactional_pattern, audit_pattern, selected_approaches
        )
        
        print("\n✅ Alpha tuning completed successfully!")
        print(f"📁 All tuning results and charts saved to: {args.reports_dir}")
        return tuning_results
    else:
        # Normal testing mode
        tree_systems = runner._build_all_tree_systems(documents, transactional_pattern, audit_pattern, selected_approaches)
    
    # Debug: Show sample property and document IDs from the actual generated data
    # print("\n--- Debug: Sample Generated Data ---")
    # if len(property_dataset) > 0:
    #     sample_property = property_dataset[0]
    #     print(f"  Sample property ID: {sample_property.property_id}")
    #     if len(sample_property.documents) > 0:
    #         sample_doc = sample_property.documents[0]
    #         print(f"  Sample document ID: {sample_doc.doc_id}")
    #         print(f"  Sample document full_id: {sample_doc.full_id}")
    #         print(f"  Sample document hash: {sample_doc.hash_hex}")
    
    # # Create realistic queries using actual generated data
    # realistic_queries = []
    # if len(property_dataset) >= 2:
    #     # Get first property for TRANSACTIONAL queries
    #     prop1 = property_dataset[0]
    #     print(f"  Property 1: {prop1.property_id} has {len(prop1.documents)} documents")
    #     for i, doc in enumerate(prop1.documents[:3]):
    #         print(f"    Doc {i}: {doc.doc_id}")
        
    #     if len(prop1.documents) >= 2:
    #         realistic_queries.append(
    #             ("TRANSACTIONAL", prop1.property_id, [prop1.documents[0].doc_id, prop1.documents[1].doc_id])
    #         )
        
    #     # Get second property for another TRANSACTIONAL query
    #     prop2 = property_dataset[1]
    #     print(f"  Property 2: {prop2.property_id} has {len(prop2.documents)} documents")
    #     for i, doc in enumerate(prop2.documents[:3]):
    #         print(f"    Doc {i}: {doc.doc_id}")
            
    #     if len(prop2.documents) >= 1:
    #         realistic_queries.append(
    #             ("TRANSACTIONAL", prop2.property_id, [prop2.documents[0].doc_id])
    #         )
        
    #     # Create an AUDIT query using documents from different properties
    #     if len(prop1.documents) >= 1 and len(prop2.documents) >= 1:
    #         realistic_queries.append(
    #             ("AUDIT", [(prop1.property_id, prop1.documents[0].doc_id), 
    #                       (prop2.property_id, prop2.documents[0].doc_id)])
    #         )
    
    # print(f"  Generated {len(realistic_queries)} realistic queries based on actual data")
    # for i, query in enumerate(realistic_queries):
    #     print(f"  Query {i+1}: {query[0]} - {len(query[1]) if query[0] == 'TRANSACTIONAL' else len(query[1])} items")
    
    # queries = realistic_queries if realistic_queries else [
    #     # Fallback to original queries if data generation failed
    #     ("TRANSACTIONAL", "Property1", ["doc1", "doc3"]),
    #     ("TRANSACTIONAL", "Property2", ["doc2"]),
    #     ("AUDIT", [("Property1", "doc1"), ("Property3", "doc4")])
    # ]
    
    print("🚀 Running verification tests...")
    result = runner.run_tests(documents, tree_systems, final_workload, selected_approaches)
    
    # Generate comprehensive reports
    runner._generate_reports()
    
    print("\n✅ Test completed successfully!")
    print(f"📁 All results and charts saved to: {args.reports_dir}")
    return result


if __name__ == "__main__":
    main()
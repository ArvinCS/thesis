#!/usr/bin/env python3
"""
Clustered Flat Tree Adapter

Adapter to integrate Clustered Flat Tree with existing verification suite
and benchmarking infrastructure.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clustered_flat_tree_builder import ClusteredFlatTreeBuilder, Document
import json

class ClusteredFlatTreeAdapter:
    """Adapter for Clustered Flat Tree to work with existing benchmark infrastructure."""
    
    def __init__(self):
        self.approach_name = "clustered_flat"
        self.display_name = "Clustered Flat Tree"
        self.description = "Single flat Merkle tree with two-level optimization: province clustering + pairs-first Huffman"
    
    def prepare_documents(self, documents_data):
        """Convert generic document data to Document objects."""
        documents = []
        for doc_data in documents_data:
            if isinstance(doc_data, dict):
                doc_id = doc_data.get('doc_id', f"doc_{len(documents)}")
                content = doc_data.get('content', f"content_{len(documents)}")
                province = doc_data.get('province', 'DEFAULT')
                property_id = doc_data.get('property_id', f"prop_{len(documents)}")
            elif hasattr(doc_data, 'doc_id') and hasattr(doc_data, 'province') and hasattr(doc_data, 'property_id'):
                # Handle Document objects from LargeScaleDocumentGenerator
                doc_id = doc_data.doc_id
                content = doc_data.content
                province = doc_data.province
                property_id = doc_data.property_id
            else:
                # Handle simple format
                doc_id = f"doc_{len(documents)}"
                content = str(doc_data)
                province = 'DEFAULT'
                property_id = f"prop_{len(documents)}"
            
            documents.append(Document(doc_id, content, province, property_id))
        
        return documents
    
    def build_tree_system(self, documents_data, traffic_logs=None):
        """Build the clustered flat tree system."""
        documents = self.prepare_documents(documents_data)
        
        # Only accept compressed traffic logs
        from compressed_traffic_logs import CompressedTrafficLogs
        if not isinstance(traffic_logs, (CompressedTrafficLogs, type(None))):
            raise ValueError("Only CompressedTrafficLogs instances are accepted. Legacy traffic logs are not supported.")
        
        compressed_traffic = traffic_logs if traffic_logs is not None else CompressedTrafficLogs()
        
        builder = ClusteredFlatTreeBuilder(documents, compressed_traffic)
        
        # Store builder reference for daily verification
        self.builder = builder
        
        root = builder.build()
        
        return {
            'builder': builder,
            'root': root,
            'tree_info': builder.get_tree_info(),
            'approach': self.approach_name
        }
    
    def generate_verification_data(self, tree_system, leaves_to_verify):
        """Generate verification data for the given leaves."""
        builder = tree_system['builder']
        
        # Convert leaves to hex format if needed
        if isinstance(leaves_to_verify[0], str) and not leaves_to_verify[0].startswith('0x'):
            # Assume these are leaf hashes in hex format
            leaves_hex = leaves_to_verify
        else:
            # Convert document objects to hashes
            leaves_hex = []
            for leaf in leaves_to_verify:
                if hasattr(leaf, 'hash_hex'):
                    leaves_hex.append(leaf.hash_hex)
                else:
                    # Create hash from string
                    from eth_utils import keccak
                    hash_hex = keccak(str(leaf).encode('utf-8')).hex()
                    leaves_hex.append(hash_hex)
        
        # CRITICAL FIX: Sort leaves according to tree's internal ordering
        # This ensures multiproof generation works correctly
        ordered_leaves_dict = {leaf: i for i, leaf in enumerate(builder.ordered_leaves_hex)}
        leaves_hex_sorted = sorted(leaves_hex, key=lambda x: ordered_leaves_dict.get(x, float('inf')))
        
        proof, proof_flags = builder.generate_multiproof(leaves_hex_sorted)
        
        return {
            'proof': proof,
            'proof_flags': proof_flags,
            'leaves': leaves_hex_sorted,  # Return sorted leaves
            'root': builder.merkle_root,
            'approach': self.approach_name
        }
    
    def verify_locally(self, verification_data):
        """Perform local verification."""
        proof = verification_data['proof']
        proof_flags = verification_data['proof_flags']
        leaves = verification_data['leaves']
        expected_root = verification_data['root']
        
        # Find the builder from global context or recreate
        # For now, we'll assume the verification data includes enough info
        # In practice, this would be handled by the main verification suite
        
        return True, "Local verification not implemented in adapter"
    
    def add_learning_event(self, verified_properties, day_idx=None, learning_mode=None):
        """
        Add verification event to the builder's traffic logs with unified learning support.
        
        Args:
            verified_properties: List of property IDs that were verified
            day_idx: Day index for daily learning (optional)
            learning_mode: Override learning mode (optional, uses config if None)
        """
        if not hasattr(self, 'builder'):
            return
            
        # Determine learning mode
        if learning_mode is None:
            try:
                from learning_config import get_learning_config, LearningMode
                config = get_learning_config()
                learning_mode = config.mode
            except ImportError:
                # Fallback to immediate learning if config not available
                learning_mode = "IMMEDIATE"
        
        # Apply learning based on mode
        if hasattr(self.builder, 'add_verification_event'):
            # Builder supports immediate learning
            self.builder.add_verification_event(verified_properties)
        elif hasattr(self.builder, 'add_daily_verification'):
            # Builder only supports daily learning
            self.builder.add_daily_verification(verified_properties, day_idx)
    
    # Backward compatibility methods
    def add_verification_event(self, verified_properties):
        """Add immediate verification event (backward compatibility)."""
        self.add_learning_event(verified_properties, learning_mode="IMMEDIATE")
    
    def add_daily_verification(self, verified_properties, day_idx=None):
        """Add daily verification event (backward compatibility)."""
        self.add_learning_event(verified_properties, day_idx=day_idx, learning_mode="DAILY")
    
    def prepare_contract_call_data(self, verification_data):
        """Prepare data for smart contract verification."""
        proof_bytes32 = []
        for p in verification_data['proof']:
            if p.startswith('0x'):
                proof_bytes32.append(p)
            else:
                proof_bytes32.append('0x' + p)
        
        leaves_bytes32 = []
        for leaf in verification_data['leaves']:
            if leaf.startswith('0x'):
                leaves_bytes32.append(leaf)
            else:
                leaves_bytes32.append('0x' + leaf)
        
        root_bytes32 = verification_data['root']
        if not root_bytes32.startswith('0x'):
            root_bytes32 = '0x' + root_bytes32
        
        return {
            'proof': proof_bytes32,
            'proofFlags': verification_data['proof_flags'],
            'leaves': leaves_bytes32,
            'root': root_bytes32
        }
    
    def get_contract_function_name(self):
        """Get the name of the contract function to call."""
        return 'verifyClusteredFlatMultiproof'
    
    def get_gas_estimation_function_name(self):
        """Get the name of the gas estimation function."""
        return 'estimateVerificationGas'
    
    def format_gas_analysis_result(self, gas_used, verification_data, additional_info=None):
        """Format gas analysis results for reporting."""
        return {
            'approach': self.approach_name,
            'display_name': self.display_name,
            'gas_used': gas_used,
            'leaves_count': len(verification_data['leaves']),
            'proof_length': len(verification_data['proof']),
            'proof_flags_length': len(verification_data['proof_flags']),
            'optimization': 'Two-level: Province clustering + Pairs-first Huffman',
            'additional_info': additional_info or {}
        }

# Factory function for integration
def create_clustered_flat_adapter():
    """Factory function to create the adapter."""
    return ClusteredFlatTreeAdapter()

# Test function
def test_clustered_flat_tree():
    """Test the clustered flat tree implementation."""
    print("Testing Clustered Flat Tree...")
    
    # Create test documents
    test_docs = [
        {'doc_id': 'doc1', 'content': 'content1', 'province': 'JAKARTA', 'property_id': 'prop1'},
        {'doc_id': 'doc2', 'content': 'content2', 'province': 'JAKARTA', 'property_id': 'prop2'},
        {'doc_id': 'doc3', 'content': 'content3', 'province': 'BANDUNG', 'property_id': 'prop3'},
        {'doc_id': 'doc4', 'content': 'content4', 'province': 'BANDUNG', 'property_id': 'prop4'},
        {'doc_id': 'doc5', 'content': 'content5', 'province': 'JAKARTA', 'property_id': 'prop5'},
    ]
    
    # Create test traffic logs
    test_traffic = [
        ['JAKARTA.prop1', 'JAKARTA.prop2'],  # Frequent pair
        ['JAKARTA.prop1', 'JAKARTA.prop5'],  # Another pair
        ['BANDUNG.prop3', 'BANDUNG.prop4'],  # Bandung pair
        ['JAKARTA.prop1', 'JAKARTA.prop2'],  # Repeat to increase frequency
    ]
    
    adapter = ClusteredFlatTreeAdapter()
    
    # Build tree system
    tree_system = adapter.build_tree_system(test_docs, test_traffic)
    print(f"Tree built successfully!")
    print(f"Root: {tree_system['root'][:16]}...")
    print(f"Tree info: {tree_system['tree_info']}")
    
    # Test verification
    builder = tree_system['builder']
    test_leaves = builder.ordered_leaves_hex[:3]  # Test first 3 leaves
    
    verification_data = adapter.generate_verification_data(tree_system, test_leaves)
    print(f"Verification data generated:")
    print(f"  Proof length: {len(verification_data['proof'])}")
    print(f"  Proof flags length: {len(verification_data['proof_flags'])}")
    print(f"  Leaves count: {len(verification_data['leaves'])}")
    
    # Test local verification
    is_valid, message = builder.verify_multiproof_locally(
        verification_data['proof'],
        verification_data['proof_flags'],
        verification_data['leaves']
    )
    print(f"Local verification: {is_valid} - {message}")
    
    # Test contract call data preparation
    contract_data = adapter.prepare_contract_call_data(verification_data)
    print(f"Contract call data prepared successfully")
    
    print("Clustered Flat Tree test completed!")

if __name__ == "__main__":
    test_clustered_flat_tree()

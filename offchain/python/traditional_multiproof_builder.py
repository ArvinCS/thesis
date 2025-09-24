"""
Traditional Multiproof Merkle Tree Builder

This module provides a traditional flat Merkle tree implementation with multiproof
generation for comparison against hierarchical approaches.
"""

from optimized_tree_builder import combine_and_hash
from eth_utils import keccak


class TraditionalMerkleTreeBuilder:
    """Traditional flat Merkle tree with multiproof for comparison."""
    
    def __init__(self, all_documents):
        self.all_documents = all_documents
        self.ordered_leaves_hex = []
        self.merkle_root = None
        self.layers = []
    
    def build(self):
        """Build traditional flat Merkle tree."""
        # Simple alphabetical ordering (no optimization)
        sorted_docs = sorted(self.all_documents, key=lambda d: d.full_id)
        self.ordered_leaves_hex = [doc.hash_hex for doc in sorted_docs]
        
        if not self.ordered_leaves_hex:
            self.merkle_root = keccak(b'').hex()
            return self.merkle_root
        
        # Build tree layers
        self.layers = [list(self.ordered_leaves_hex)]
        nodes = list(self.ordered_leaves_hex)
        
        while len(nodes) > 1:
            if len(nodes) % 2 != 0:
                nodes.append(nodes[-1])
            
            parents = []
            for i in range(0, len(nodes), 2):
                parent_hash = combine_and_hash(nodes[i], nodes[i+1])
                parents.append(parent_hash)
            
            self.layers.append(parents)
            nodes = parents
        
        self.merkle_root = nodes[0] if nodes else keccak(b'').hex()
        return self.merkle_root
    
    def generate_proof_for_documents(self, document_hashes):
        """Generate proof for specific documents using the exact working algorithm from optimized_tree_builder."""
        # Use the working multiproof generation algorithm
        return self.generate_batched_proof_with_flags(document_hashes)
    
    def generate_batched_proof_with_flags(self, leaves_to_prove_hex):
        """Generate multiproof using the correct OpenZeppelin-compatible algorithm."""
        if not self.ordered_leaves_hex:
            return [], []
        
        # Use the working algorithm from optimized_tree_builder
        # This ensures compatibility with OpenZeppelin's multiproof format
        return self._generate_multiproof_openzeppelin_compatible(leaves_to_prove_hex)
    
    def _generate_multiproof_openzeppelin_compatible(self, leaves_to_prove_hex):
        """Generate multiproof compatible with OpenZeppelin's multiProofVerify."""
        from openzeppelin_multiproof import generate_multiproof_openzeppelin, build_tree_layers
        
        if not leaves_to_prove_hex:
            return [], []
        
        # Sort leaves to ensure consistent ordering
        # Ensure uniqueness to avoid duplicated leaves in multiproof
        sorted_leaves = sorted(set(leaves_to_prove_hex))
        
        # Find indices of leaves to prove
        leaf_indices = []
        for leaf in sorted_leaves:
            if leaf in self.ordered_leaves_hex:
                leaf_indices.append(self.ordered_leaves_hex.index(leaf))
        
        if not leaf_indices:
            return [], []
        
        # Build the tree layers
        tree_layers = build_tree_layers(self.ordered_leaves_hex)
        
        # Generate multiproof using the correct OpenZeppelin algorithm
        proof, proof_flags = generate_multiproof_openzeppelin(
            self.ordered_leaves_hex, leaf_indices, tree_layers
        )
        
        return proof, proof_flags

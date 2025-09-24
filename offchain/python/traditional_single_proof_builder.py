"""
Traditional Single Proof Merkle Tree Builder

This module provides a traditional flat Merkle tree implementation with single proof
generation for individual document verification.
"""

from optimized_tree_builder import combine_and_hash
from eth_utils import keccak


class TraditionalSingleProofMerkleTreeBuilder:
    """Traditional flat Merkle tree with single proof per document."""
    
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
    
    def generate_single_proof_for_document(self, document_hash):
        """Generate a single proof for one document."""
        if not self.layers:
            return [], []
        
        # Find index of document to prove
        if document_hash not in self.ordered_leaves_hex:
            return [], []
        
        leaf_index = self.ordered_leaves_hex.index(document_hash)
        proof = []
        
        # Generate proof by traversing up the tree
        current_index = leaf_index
        for layer in self.layers[:-1]:  # Exclude root layer
            if len(layer) % 2 != 0:
                layer = layer + [layer[-1]]  # Duplicate last element if odd
            
            # Determine sibling
            if current_index % 2 == 0:  # Left child
                sibling_index = current_index + 1
            else:  # Right child
                sibling_index = current_index - 1
            
            # Add sibling to proof
            if sibling_index < len(layer):
                proof.append(layer[sibling_index])
            
            # Move to parent
            current_index = current_index // 2
        
        return proof
    
    def generate_single_proofs_for_documents(self, document_hashes):
        """Generate individual proofs for each document."""
        all_proofs = []
        for doc_hash in document_hashes:
            proof = self.generate_single_proof_for_document(doc_hash)
            all_proofs.append({
                'document_hash': doc_hash,
                'proof': proof,
                'proof_size_bytes': len(proof) * 32  # 32 bytes per hash
            })
        return all_proofs
    
    def generate_proofs_for_documents(self, document_hashes):
        """Alias for generate_single_proofs_for_documents for compatibility."""
        return self.generate_single_proofs_for_documents(document_hashes)

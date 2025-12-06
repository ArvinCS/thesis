"""
Traditional Multiproof Merkle Tree Builder

This module provides a traditional flat Merkle tree implementation with multiproof
generation for comparison against hierarchical approaches.

Uses OpenZeppelin-compatible multiproof format for balanced trees.
"""

import random
from optimized_tree_builder import combine_and_hash
from eth_utils import keccak


class TraditionalMerkleTreeBuilder:
    """Traditional flat Merkle tree with multiproof for comparison."""
    
    def __init__(self, all_documents, random_seed=42):
        self.all_documents = all_documents
        self.ordered_leaves_hex = []
        self.merkle_root = None
        self.layers = []
        self.random_seed = random_seed
    
    def build(self):
        """Build traditional flat Merkle tree with random leaf ordering (no optimization)."""
        # Random shuffling for true baseline comparison (no artificial clustering)
        shuffled_docs = list(self.all_documents)
        random.seed(self.random_seed)  # Deterministic randomness for reproducible results
        random.shuffle(shuffled_docs)
        self.ordered_leaves_hex = [doc.hash_hex for doc in shuffled_docs]
        
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
        """
        Generate multiproof using the correct OpenZeppelin-compatible algorithm.
        
        Returns:
            tuple: (proof, proof_flags, leaves_in_order) where leaves_in_order
                   is the leaves in the exact order expected by verification.
        """
        if not self.ordered_leaves_hex:
            return [], [], []
        
        # Use the working algorithm from optimized_tree_builder
        # This ensures compatibility with OpenZeppelin's multiproof format
        return self._generate_multiproof_openzeppelin_compatible(leaves_to_prove_hex)
    
    def generate_multiproof(self, leaves_to_prove_hex):
        """
        Generate multiproof using OpenZeppelin-compatible format for balanced trees.
        
        Returns:
            tuple: (proof, proof_flags, leaves_in_order) where leaves_in_order
                   is the leaves in the exact order expected by verification.
        """
        return self._generate_multiproof_openzeppelin_compatible(leaves_to_prove_hex)
    
    def _generate_multiproof_openzeppelin_compatible(self, leaves_to_prove_hex):
        """Generate multiproof compatible with OpenZeppelin's multiProofVerify."""
        from openzeppelin_multiproof import generate_multiproof_openzeppelin
        
        if not leaves_to_prove_hex:
            return [], [], []
        
        # Remove duplicates while preserving order
        unique_leaves = list(set(leaves_to_prove_hex))
        
        # Find indices of leaves to prove in the tree
        leaf_indices = []
        for leaf in unique_leaves:
            if leaf in self.ordered_leaves_hex:
                leaf_indices.append(self.ordered_leaves_hex.index(leaf))
        
        if not leaf_indices:
            return [], [], []
        
        # Use the stored tree layers (already built in self.build())
        # This ensures consistency with how the tree was originally built
        tree_layers = self.layers
        
        # Generate multiproof using the correct OpenZeppelin algorithm
        # This returns proof, flags, AND the leaves in the correct order for verification
        proof, proof_flags, leaves_in_order = generate_multiproof_openzeppelin(
            self.ordered_leaves_hex, leaf_indices, tree_layers
        )
        
        return proof, proof_flags, leaves_in_order
    
    def generate_pathmap_proof(self, leaves_to_prove_hex):
        """Generate proof using new pathMap format for bottom-up reconstruction."""
        if not leaves_to_prove_hex:
            return {'leaves': [], 'proofHashes': [], 'pathMap': []}
        
        # Sort and deduplicate leaves
        sorted_leaves = sorted(set(leaves_to_prove_hex))
        
        # Find indices of leaves to prove in the tree
        leaf_indices = []
        for leaf in sorted_leaves:
            if leaf in self.ordered_leaves_hex:
                leaf_indices.append(self.ordered_leaves_hex.index(leaf))
        
        if not leaf_indices:
            return {'leaves': [], 'proofHashes': [], 'pathMap': []}
        
        # Step 1: Gather all nodes in proof subtree
        proof_nodes = set()
        node_positions = {}  # Maps (layer, index) to node hash
        
        # Add leaves being proven
        for idx in leaf_indices:
            proof_nodes.add((0, idx))  # (layer, index)
            node_positions[(0, idx)] = self.ordered_leaves_hex[idx]
        
        # Traverse up from each leaf, collecting all nodes and siblings
        for layer_idx in range(len(self.layers) - 1):
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]
            
            # For each node in current layer that's in our proof subtree
            nodes_to_process = [(l, i) for l, i in proof_nodes if l == layer_idx]
            
            for layer, idx in nodes_to_process:
                # Add parent
                parent_idx = idx // 2
                proof_nodes.add((layer_idx + 1, parent_idx))
                if (layer_idx + 1, parent_idx) not in node_positions:
                    node_positions[(layer_idx + 1, parent_idx)] = next_layer[parent_idx]
                
                # Add sibling
                sibling_idx = idx ^ 1  # XOR with 1 to get sibling
                if sibling_idx < len(current_layer):
                    proof_nodes.add((layer_idx, sibling_idx))
                    if (layer_idx, sibling_idx) not in node_positions:
                        node_positions[(layer_idx, sibling_idx)] = current_layer[sibling_idx]
        
        # Step 2: Separate leaves from proof hashes
        leaves = []
        proof_hashes = []
        working_set_indices = {}  # Maps (layer, index) to position in working set
        
        # First, add sorted leaves to working set
        for i, leaf_hash in enumerate(sorted_leaves):
            leaves.append(leaf_hash)
            # Find this leaf's position in the tree
            leaf_tree_idx = self.ordered_leaves_hex.index(leaf_hash)
            working_set_indices[(0, leaf_tree_idx)] = i
        
        # Then, add proof hashes (siblings and other internal nodes)
        proof_hash_nodes = []
        for layer, idx in proof_nodes:
            if (layer, idx) not in working_set_indices:  # Not already in leaves
                if layer == 0:  # Leaf siblings
                    if node_positions[(layer, idx)] not in sorted_leaves:
                        proof_hash_nodes.append((layer, idx, node_positions[(layer, idx)]))
                else:  # Internal nodes
                    proof_hash_nodes.append((layer, idx, node_positions[(layer, idx)]))
        
        # Sort proof hashes for deterministic ordering
        proof_hash_nodes.sort(key=lambda x: x[2])  # Sort by hash value
        
        for i, (layer, idx, hash_val) in enumerate(proof_hash_nodes):
            proof_hashes.append(hash_val)
            working_set_indices[(layer, idx)] = len(leaves) + i
        
        # Step 3: Build pathMap instructions (bottom-up)
        path_map = []
        computed_indices = {}  # Maps (layer, index) to computed position
        next_computed_idx = len(leaves) + len(proof_hashes)
        
        # Process layers bottom-up (excluding leaf layer and root)
        for layer_idx in range(1, len(self.layers)):
            layer_nodes = [(l, i) for l, i in proof_nodes if l == layer_idx]
            layer_nodes.sort()  # Process in consistent order
            
            for layer, idx in layer_nodes:
                if (layer, idx) not in working_set_indices:  # Needs to be computed
                    # Find left and right children
                    left_child = (layer - 1, idx * 2)
                    right_child = (layer - 1, idx * 2 + 1)
                    
                    # Get indices in working set
                    left_idx = working_set_indices.get(left_child)
                    right_idx = working_set_indices.get(right_child)
                    
                    if left_idx is not None and right_idx is not None:
                        path_map.extend([left_idx, right_idx])
                        computed_indices[(layer, idx)] = next_computed_idx
                        working_set_indices[(layer, idx)] = next_computed_idx
                        next_computed_idx += 1
        
        return {
            'leaves': leaves,
            'proofHashes': proof_hashes,
            'pathMap': path_map
        }

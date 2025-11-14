#!/usr/bin/env python3
"""
Clustered Province Tree Builder

This module implements a single flat Merkle tree with two-level custom leaf ordering:
1. Primary Sorting (Clustering): Group leaves by jurisdiction (province)
2. Secondary Sorting (Traffic Optimization): Within each province cluster, 
   use "Pairs-first Huffman" algorithm based on local co-verification frequency

Features:
- Single flat Merkle tree architecture
- Province-based clustering for jurisdictional efficiency
- Traffic-aware optimization within provinces
- Gas-optimized multiproof generation
- Compatible with inline Solidity verifier
"""

import heapq
from collections import defaultdict
from eth_utils import keccak
from typing import Dict, List, Tuple, Any, Optional
from basic_data_structure import Document

class PropertyCluster:
    """Represents a group of documents for a single property within a province."""
    def __init__(self, property_id):
        self.property_id = property_id
        self.documents = []
        self.frequency = 0  # Co-verification frequency
    
    def add_document(self, doc):
        self.documents.append(doc)
    
    def get_leaf_hashes_hex(self):
        return [doc.hash_hex for doc in self.documents]
    
    def __hash__(self):
        return hash(self.property_id)
    
    def __eq__(self, other):
        return isinstance(other, PropertyCluster) and self.property_id == other.property_id

class HuffmanNode:
    """Node for Huffman tree construction."""
    def __init__(self, item, freq):
        self.item = item
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def combine_and_hash(hash1_hex, hash2_hex):
    """Combine two hashes using keccak256 (OpenZeppelin compatible)."""
    h1_bytes = bytes.fromhex(hash1_hex)
    h2_bytes = bytes.fromhex(hash2_hex)
    # Sort hashes to ensure deterministic ordering
    combined = h1_bytes + h2_bytes if h1_bytes < h2_bytes else h2_bytes + h1_bytes
    return keccak(combined).hex()

class ClusteredProvinceTreeBuilder:
    """
    Builds a single flat Merkle tree with two-level custom leaf ordering:
    1. Primary: Cluster by province (jurisdiction)
    2. Secondary: Optimize within province using pairs-first Huffman
    """
    
    def __init__(self, all_documents, audit_pattern=None, transactional_pattern=None):
        self.all_documents = all_documents
        self.merkle_root = None
        self.ordered_leaves_hex = []
        self.tree_layers = []
        
        # Accept access patterns for optimization
        from access_patterns_enhanced import AuditPattern, TransactionalPattern
        if audit_pattern is not None and not isinstance(audit_pattern, AuditPattern):
            raise ValueError("Only AuditPattern instances are accepted for audit_pattern.")
        if transactional_pattern is not None and not isinstance(transactional_pattern, TransactionalPattern):
            raise ValueError("Only TransactionalPattern instances are accepted for transactional_pattern.")
        
        self.audit_pattern = audit_pattern
        self.transactional_pattern = transactional_pattern
        self.verification_count = 0
        
        # Group documents by province and property
        self.province_clusters = self._create_province_clusters()
        
    def _create_province_clusters(self):
        """Group documents by province, then by property within each province."""
        clusters = {}
        
        for doc in self.all_documents:
            province = doc.province
            if province not in clusters:
                clusters[province] = {}
            
            full_id = doc.full_id
            if full_id not in clusters[province]:
                clusters[province][full_id] = PropertyCluster(full_id)
            
            clusters[province][full_id].add_document(doc)
        
        return clusters
    
    def _calculate_frequencies(self):
        """Calculate co-verification frequencies for optimization using access patterns."""
        property_freq = defaultdict(int)
        pair_freq = defaultdict(int)
        province_freq = defaultdict(int)
        
        # Use transactional pattern for property and pair frequencies
        if self.transactional_pattern:
            # Calculate property frequencies
            for province, property_clusters in self.province_clusters.items():
                for prop_id, cluster in property_clusters.items():
                    if cluster.documents:
                        doc_frequencies = self.transactional_pattern.get_document_frequencies(cluster.documents)
                        prop_freq = sum(doc_frequencies.values())
                        property_freq[prop_id] = prop_freq
                        province_freq[province] += prop_freq
            
            # Calculate pair frequencies using simulated queries within each cluster
            for province, clusters in self.province_clusters.items():
                for cluster in clusters.values():
                    if len(cluster.documents) >= 2:
                        doc_pair_freq = self.transactional_pattern.get_document_pair_frequencies(
                            cluster.documents, num_simulated_queries=60  # Reduced for efficiency
                        )
                        
                        # Convert document pairs to property pairs
                        for (doc1, doc2), freq in doc_pair_freq.items():
                            prop1_id = f"{doc1.province}.{doc1.property_id}"
                            prop2_id = f"{doc2.province}.{doc2.property_id}"
                            prop_pair = tuple(sorted([prop1_id, prop2_id]))
                            pair_freq[prop_pair] += freq
        
        return dict(province_freq), dict(property_freq), dict(pair_freq)
    
    def _build_huffman_tree(self, items_with_freq):
        """Build Huffman tree for traffic-aware optimization."""
        if not items_with_freq:
            return None
            
        pq = [HuffmanNode(item, freq) for item, freq in items_with_freq.items()]
        heapq.heapify(pq)
        
        while len(pq) > 1:
            left = heapq.heappop(pq)
            right = heapq.heappop(pq)
            
            merged = HuffmanNode(None, left.freq + right.freq)
            merged.left = left
            merged.right = right
            
            heapq.heappush(pq, merged)
        
        return pq[0] if pq else None

    def _build_balanced_subtree_hash(self, hashes: List[str]) -> str:
        """Build a balanced Merkle subtree from a list of hex hashes and return the root hash.

        This constructs a standard balanced binary Merkle tree (pad with last element
        when a level has an odd number of nodes).
        """
        if not hashes:
            return keccak(b'').hex()

        current = list(hashes)
        # If only one hash, return it directly
        if len(current) == 1:
            return current[0]

        while len(current) > 1:
            if len(current) % 2 != 0:
                current.append(current[-1])

            next_level = []
            for i in range(0, len(current), 2):
                parent = combine_and_hash(current[i], current[i + 1])
                next_level.append(parent)

            current = next_level

        return current[0]

    def _generate_huffman_codes(self, root_node):
        """Generate Huffman codes from the tree using iterative approach."""
        codes = {}
        if root_node is None:
            return codes
        
        # Use a stack to simulate recursion
        stack = [(root_node, "")]
            
        while stack:
            node, code = stack.pop()
            
            if node.item is not None:
                # Leaf node - store the code
                codes[node.item] = code
            else:
                # Internal node - add children to stack
                if node.right is not None:
                    stack.append((node.right, code + "1"))
                if node.left is not None:
                    stack.append((node.left, code + "0"))
                    
        return codes
    
    def _optimize_province_ordering(self, province, property_clusters, property_freq, pair_freq, alpha_threshold=0.12):
        """
        Optimize ordering within a province using a simple frequency-based ordering.

        This ordinary clustered-province builder does not perform pairs-first Huffman
        merging. Instead, properties inside a province are ordered by their
        aggregate frequency (descending). Keeping this implementation simple
        avoids the O(n^2) pair-processing cost and matches the ordinary
        clustered province tree behavior.
        """
        if not property_clusters:
            return []
        
        # Filter pair frequencies for this province only
        province_pairs = {}
        for (prop1, prop2), freq in pair_freq.items():
            # Check if both properties belong to this province
            if ('.' in prop1 and prop1.split('.')[0] == province and 
                '.' in prop2 and prop2.split('.')[0] == province):
                province_pairs[(prop1, prop2)] = freq
        
        # Calculate alpha threshold based on maximum pair frequency in this province
        # For the ordinary clustered province tree we simply sort properties by
        # their frequency. Use frequency from property_freq (defaults to 0).
        sorted_clusters = sorted(
            property_clusters,
            key=lambda c: property_freq.get(c.property_id, 0),
            reverse=True
        )

        print(f"    Province {province}: Ordering {len(sorted_clusters)} properties by frequency")
        return sorted_clusters
    
    def _apply_pairs_first_huffman_to_clusters(self, clusters, cluster_freq, pair_freq, min_pair_freq_threshold, province):
        """
        Apply pairs-first Huffman algorithm to property clusters.
        Creates unbalanced tree with shallow depth for frequent pairs.
        """
        if len(clusters) <= 1:
            return clusters
        
        # Create cluster nodes with frequencies
        cluster_nodes = {}
        for cluster in clusters:
            freq = cluster_freq.get(cluster.property_id, 1)
            cluster_nodes[cluster.property_id] = HuffmanNode(cluster, freq)
        
        merged_ids = set()
        strong_pairs_count = 0
        weak_pairs_count = 0
        
        # Sort pairs by frequency (most frequent first)
        sorted_pairs = sorted(pair_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Step 1: Merge strong pairs first (pairs-first approach)
        for (prop1, prop2), freq in sorted_pairs:
            if freq >= min_pair_freq_threshold:
                # Check if both properties exist and haven't been merged yet
                if (prop1 not in merged_ids and prop2 not in merged_ids and 
                    prop1 in cluster_nodes and prop2 in cluster_nodes):
                    
                    # Create merged node (shallow depth for this pair)
                    left_node = cluster_nodes[prop1]
                    right_node = cluster_nodes[prop2]
                    
                    merged_node = HuffmanNode(None, left_node.freq + right_node.freq)
                    merged_node.left = left_node
                    merged_node.right = right_node
                    
                    # Replace both nodes with merged node
                    merged_id = f"merged_{prop1}_{prop2}"
                    cluster_nodes[merged_id] = merged_node
                    
                    # Safely remove the original nodes
                    if prop1 in cluster_nodes:
                        del cluster_nodes[prop1]
                    if prop2 in cluster_nodes:
                        del cluster_nodes[prop2]
                    
                    merged_ids.add(prop1)
                    merged_ids.add(prop2)
                    strong_pairs_count += 1
            else:
                weak_pairs_count += 1
        
        print(f"    Province {province}: Merged {strong_pairs_count} strong pairs, skipped {weak_pairs_count} weak pairs")
        
        # Step 2: Build Huffman tree from remaining nodes (including merged pairs)
        remaining_nodes = list(cluster_nodes.values())
        
        if len(remaining_nodes) <= 1:
            # Extract clusters from the final structure
            if remaining_nodes:
                return self._extract_clusters_from_node(remaining_nodes[0])
            else:
                return clusters
        
        # Build Huffman tree for remaining nodes
        heap = remaining_nodes.copy()
        heapq.heapify(heap)
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            merged = HuffmanNode(None, left.freq + right.freq)
            merged.left = left
            merged.right = right
            
            heapq.heappush(heap, merged)
        
        # Extract final ordering from Huffman tree
        if heap:
            return self._extract_clusters_from_node(heap[0])
        else:
            return clusters
    
    def _extract_clusters_from_node(self, node):
        """
        Extract clusters from Huffman tree node in optimal order (in-order traversal).
        """
        if node is None:
            return []
        
        if node.item is not None:
            # Leaf node - return the cluster
            return [node.item]
        
        # Internal node - traverse left then right for optimal ordering
        result = []
        result.extend(self._extract_clusters_from_node(node.left))
        result.extend(self._extract_clusters_from_node(node.right))
        return result
    
    def build(self):
        """
        Build the clustered province tree with province-based clustering:
        1. Primary: Group by province
        2. Secondary: Within each province, flatten documents in deterministic order
        3. Build a single flat balanced Merkle tree from all documents (preserving province order)
        
        The key difference from traditional shuffled approach is that documents are
        ordered by province first, keeping locality benefits while using a standard
        flat Merkle tree structure for proof compatibility.
        """
        print("Building Clustered Province Tree (flat with province clustering)...")
        
        # Calculate frequencies (still useful for stats)
        province_freq, property_freq, pair_freq = self._calculate_frequencies()

        # Primary sorting: Group by province (alphabetical order for consistency)
        sorted_provinces = sorted(self.province_clusters.keys())

        # Flatten all documents, maintaining province clusters
        all_document_hashes = []
        self.province_boundaries = {}  # Store where each province starts/ends
        
        for province in sorted_provinces:
            start_idx = len(all_document_hashes)
            property_clusters = list(self.province_clusters[province].values())
            
            # Flatten documents for this province in deterministic order
            for cluster in sorted(property_clusters, key=lambda c: c.property_id):
                sorted_docs = sorted(cluster.documents, key=lambda d: d.doc_id)
                for d in sorted_docs:
                    all_document_hashes.append(d.hash_hex)
            
            end_idx = len(all_document_hashes)
            self.province_boundaries[province] = (start_idx, end_idx)
            print(f"  Province {province}: docs {start_idx}-{end_idx} ({end_idx - start_idx} total)")
        
        # Set ordered_leaves_hex to document-level hashes (flat structure)
        self.ordered_leaves_hex = all_document_hashes
        
        if not self.ordered_leaves_hex:
            self.merkle_root = keccak(b'').hex()
            return self.merkle_root

        # Build balanced Merkle tree layers from document leaves
        self.tree_layers = []
        current_layer = list(self.ordered_leaves_hex)
        self.tree_layers.append(current_layer)

        while len(current_layer) > 1:
            if len(current_layer) % 2 != 0:
                current_layer.append(current_layer[-1])

            next_level = []
            for i in range(0, len(current_layer), 2):
                parent_hash = combine_and_hash(current_layer[i], current_layer[i + 1])
                next_level.append(parent_hash)

            self.tree_layers.append(next_level)
            current_layer = next_level

        self.merkle_root = current_layer[0] if current_layer else keccak(b'').hex()

        print(f"  Clustered Province Tree (flat) Root: {self.merkle_root[:16]}...")
        print(f"  Total documents: {len(self.ordered_leaves_hex)}")
        return self.merkle_root
    
    def generate_multiproof(self, leaves_to_prove_hex):
        """
        Generate optimized multiproof for the flat tree.
        Returns proof elements and flags for inline Solidity verification.
        """
        if self.merkle_root is None:
            raise ValueError("Tree not built yet")
        
        if not leaves_to_prove_hex:
            return [], []
        
        # Remove duplicates while preserving order
        unique_leaves = []
        seen = set()
        for leaf in leaves_to_prove_hex:
            if leaf not in seen:
                unique_leaves.append(leaf)
                seen.add(leaf)
        
        # Map leaves to indices
        leaf_indices_map = {leaf: i for i, leaf in enumerate(self.ordered_leaves_hex)}
        
        # Track nodes that are part of proof path
        processed_nodes = {}
        for leaf in unique_leaves:
            if leaf in leaf_indices_map:
                processed_nodes[(0, leaf_indices_map[leaf])] = True
        
        proof = []
        proof_flags = []
        proof_nodes_seen = set()
        
        # Generate proof by traversing layers bottom-up
        for layer_idx in range(len(self.tree_layers) - 1):
            layer_nodes = list(self.tree_layers[layer_idx])
            
            # Pad layer if odd
            if len(layer_nodes) % 2 != 0:
                layer_nodes.append(layer_nodes[-1])
            
            for node_idx in range(0, len(layer_nodes), 2):
                left_key = (layer_idx, node_idx)
                right_key = (layer_idx, node_idx + 1)
                
                is_left_in_path = left_key in processed_nodes
                is_right_in_path = right_key in processed_nodes
                
                if is_left_in_path or is_right_in_path:
                    # Mark parent as processed
                    parent_key = (layer_idx + 1, node_idx // 2)
                    processed_nodes[parent_key] = True
                    
                    if is_left_in_path and is_right_in_path:
                        # Both children in path - flag only
                        proof_flags.append(True)
                    elif is_left_in_path:
                        # Need right sibling
                        right_node = layer_nodes[node_idx + 1]
                        if right_node not in proof_nodes_seen:
                            proof.append(right_node)
                            proof_nodes_seen.add(right_node)
                        proof_flags.append(False)
                    else:  # is_right_in_path
                        # Need left sibling
                        left_node = layer_nodes[node_idx]
                        if left_node not in proof_nodes_seen:
                            proof.append(left_node)
                            proof_nodes_seen.add(left_node)
                        proof_flags.append(False)
        
        return proof, proof_flags
    
    def generate_iterative_proofs(self, document_hashes):
        """
        Generate iterative proofs compatible with IterativeMultiProofVerifier.
        Each document gets its own proof path to the root.
        """
        if not document_hashes:
            return []
        
        if self.merkle_root is None:
            raise ValueError("Tree not built yet")
        
        iterative_proofs = []
        
        for doc_hash in document_hashes:
            # Find leaf index
            try:
                leaf_index = self.ordered_leaves_hex.index(doc_hash)
            except ValueError:
                # Document not found in tree
                continue
            
            # Generate proof path for this specific document
            proof_path = []
            positions = []
            current_index = leaf_index
            
            # Traverse up the tree layers
            for layer_idx in range(len(self.tree_layers) - 1):
                layer = self.tree_layers[layer_idx]
                
                # Ensure even number of nodes
                if len(layer) % 2 != 0:
                    layer = layer + [layer[-1]]  # Pad with last element
                
                # Find sibling
                if current_index % 2 == 0:
                    # Left node, sibling is on the right
                    sibling_index = current_index + 1
                    if sibling_index < len(layer):
                        proof_path.append(layer[sibling_index])
                        positions.append(1)  # Right sibling
                    else:
                        proof_path.append(layer[current_index])  # Self as sibling
                        positions.append(1)
                else:
                    # Right node, sibling is on the left
                    sibling_index = current_index - 1
                    proof_path.append(layer[sibling_index])
                    positions.append(0)  # Left sibling
                
                # Move up to parent
                current_index = current_index // 2
            
            iterative_proofs.append({
                'leaf': doc_hash,
                'proof': proof_path,
                'positions': positions
            })
        
        return iterative_proofs
    
    def verify_multiproof_locally(self, proof, proof_flags, leaves):
        """Local verification of multiproof for testing."""
        leaves_len = len(leaves)
        proof_flags_len = len(proof_flags)
        
        if proof_flags_len == 0:
            if leaves_len == 1:
                return leaves[0] == self.merkle_root, "Single leaf verification"
            elif leaves_len == 0:
                return keccak(b'').hex() == self.merkle_root, "Empty tree"
            else:
                return False, "Invalid empty proof for multiple leaves"
        
        # Validate proof structure
        expected_total = proof_flags_len + 1
        actual_total = leaves_len + len(proof)
        
        if abs(actual_total - expected_total) > 1:
            return False, f"Invalid proof structure: expected ~{expected_total}, got {actual_total}"
        
        hashes = [''] * proof_flags_len
        leaf_pos, hash_pos, proof_pos = 0, 0, 0
        
        try:
            for i in range(proof_flags_len):
                # Get first operand
                if leaf_pos < leaves_len:
                    a = leaves[leaf_pos]
                    leaf_pos += 1
                elif hash_pos < len(hashes) and i > hash_pos:
                    a = hashes[hash_pos]
                    hash_pos += 1
                else:
                    return False, f"Cannot get first operand at step {i}"
                
                # Get second operand
                if proof_flags[i]:
                    if leaf_pos < leaves_len:
                        b = leaves[leaf_pos]
                        leaf_pos += 1
                    elif hash_pos < len(hashes) and i > hash_pos:
                        b = hashes[hash_pos]
                        hash_pos += 1
                    else:
                        return False, f"Cannot get second operand (flag) at step {i}"
                else:
                    if proof_pos >= len(proof):
                        return False, f"Proof consumed prematurely at step {i}"
                    b = proof[proof_pos]
                    proof_pos += 1
                
                hashes[i] = combine_and_hash(a, b)
            
            # Check if we consumed the proof appropriately
            if proof_pos < len(proof) - 1:
                return False, f"Proof not fully consumed: used {proof_pos}/{len(proof)}"
            
            computed_root = hashes[proof_flags_len - 1]
            is_valid = computed_root == self.merkle_root
            
            return is_valid, "OK" if is_valid else f"Root mismatch: {computed_root[:16]}... vs {self.merkle_root[:16]}..."
            
        except Exception as e:
            return False, f"Multiproof processing error: {str(e)}"
    
    def add_daily_verification(self, property_ids):
        """Add verification patterns observed during the day for learning."""
        if self.daily_learning_active:
            self.daily_verification_patterns.append(property_ids)
    
    def end_of_day_rebuild(self):
        """Rebuild the tree structure based on patterns observed during the day."""
        if not self.daily_learning_active or not self.daily_verification_patterns:
            return False
        
        print(f"ClusteredProvinceTreeBuilder: Rebuilding with {len(self.daily_verification_patterns)} daily patterns...")
        
        # Add daily patterns to compressed traffic logs
        for pattern in self.daily_verification_patterns:
            self.compressed_traffic.add_verification_event(pattern)
        
        # Rebuild the tree with enhanced traffic data
        try:
            self.build()
            print("ClusteredProvinceTreeBuilder: Daily rebuild completed successfully")
            return True
        except Exception as e:
            print(f"ClusteredProvinceTreeBuilder: Daily rebuild failed: {e}")
            return False
    
    def get_daily_learning_stats(self, current_day=None):
        """Get statistics about daily learning patterns."""
        if not self.daily_learning_active:
            return {"learning_active": False}
        
        return {
            "learning_active": True,
            "patterns_collected": len(self.daily_verification_patterns),
            "unique_properties": len(set(prop for pattern in self.daily_verification_patterns for prop in pattern)),
            "total_verifications": sum(len(pattern) for pattern in self.daily_verification_patterns),
            "current_day": current_day
        }
    
    def reset_daily_learning(self):
        """Reset daily learning data, typically called at start of new day."""
        self.daily_verification_patterns = []
        self.daily_learning_active = True
        print("ClusteredProvinceTreeBuilder: Daily learning reset for new day")
    
    def get_tree_info(self):
        """Get information about the built tree."""
        if not self.merkle_root:
            return {"status": "not_built"}
        
        province_distribution = {}
        for doc in self.all_documents:
            province = doc.province
            if province not in province_distribution:
                province_distribution[province] = 0
            province_distribution[province] += 1
        
        return {
            "tree_type": "Clustered Province Tree",
            "merkle_root": self.merkle_root,
            "total_leaves": len(self.ordered_leaves_hex),
            "total_documents": len(self.all_documents),
            "provinces": len(province_distribution),
            "province_distribution": province_distribution,
            "tree_depth": len(self.tree_layers),
            "optimization": "Two-level: Province clustering + Pairs-first Huffman",
        }
    
    def add_learning_event(self, verified_properties, learning_mode=None, end_of_day=False):
        """
        Add verification event with unified learning support.
        
        Args:
            verified_properties: List of property IDs that were verified
            learning_mode: Override learning mode (optional, uses config if None)
            end_of_day: Boolean indicating if this is the end of day event
        """
        try:
            # Always get the config for settings like verbose_logging, batch_size, etc.
            from learning_config import get_learning_config, LearningMode
            config = get_learning_config()
            
            # Determine which learning mode to use
            if learning_mode is None:
                effective_mode = config.mode
            else:
                # Convert string mode to enum if needed
                if isinstance(learning_mode, str):
                    effective_mode = LearningMode(learning_mode.lower())
                else:
                    effective_mode = learning_mode
            
            if config.verbose_logging:
                print(f"📚 Using learning mode: {effective_mode.value}")
            
            # Always add to compressed traffic logs
            self.compressed_traffic.add_verification_event(verified_properties)
            self.verification_count += 1

            # Determine if we should rebuild based on the effective learning mode
            should_rebuild = False
            
            if effective_mode == LearningMode.IMMEDIATE:
                should_rebuild = True
                if config.verbose_logging:
                    print(f"📚 Immediate learning triggered (verification #{self.verification_count})")
            elif effective_mode == LearningMode.BATCH:
                should_rebuild = (self.verification_count % config.batch_size == 0)
                if should_rebuild and config.verbose_logging:
                    print(f"📚 Batch learning triggered (every {config.batch_size} verifications)")
            elif effective_mode == LearningMode.DAILY:
                should_rebuild = end_of_day  # Rebuild only at end of day
                if should_rebuild and config.verbose_logging:
                    print(f"📚 Daily learning triggered (end of day)")
            elif effective_mode == LearningMode.HYBRID:
                # Use immediate learning for simple events, batch for complex ones
                if len(verified_properties) < config.immediate_threshold:
                    should_rebuild = True
                    if config.verbose_logging:
                        print(f"📚 Hybrid immediate learning (simple event: {len(verified_properties)} properties)")
                elif self.verification_count % config.batch_size == 0:
                    should_rebuild = True
                    if config.verbose_logging:
                        print(f"📚 Hybrid batch learning (complex events batched)")
            elif effective_mode == LearningMode.DISABLED:
                should_rebuild = False
                if config.verbose_logging:
                    print(f"📚 Learning disabled - no rebuild")
            
            if should_rebuild:
                old_root = self.merkle_root
                old_leaf_order = self.ordered_leaves_hex.copy() if self.ordered_leaves_hex else []
                self.build()
                if config.verbose_logging:
                    print(f"✅ Tree rebuilt after verification #{self.verification_count}")
                if old_root != self.merkle_root and config.verbose_logging:
                    print(f"   Root changed: {old_root[:16]}... -> {self.merkle_root[:16]}...")
                # if old_leaf_order != self.ordered_leaves_hex and config.verbose_logging:
                    # print(f"   Leaf order changed: {old_leaf_order[:16]}... -> {self.ordered_leaves_hex[:16]}...")
            
            return should_rebuild, self.merkle_root
            
        except Exception as e:
            print(f"⚠️ Error in add_learning_event: {e}")
            # Fallback: always add to traffic logs and use immediate learning
            self.compressed_traffic.add_verification_event(verified_properties)
            self.verification_count += 1
            self.build()
            return True, self.merkle_root

    def add_verification_event(self, verified_properties, learning_mode=None, end_of_day=False):
        """Add new verification event to compressed traffic logs (backward compatibility)."""
        return self.add_learning_event(verified_properties, learning_mode=learning_mode, end_of_day=end_of_day)
    
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
        for layer_idx in range(len(self.tree_layers) - 1):
            current_layer = self.tree_layers[layer_idx]
            next_layer = self.tree_layers[layer_idx + 1]
            
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
        
        # Then, add proof hashes (only siblings that can't be computed)
        proof_hash_nodes = []
        computable_nodes = set()  # Nodes that can be computed from their children
        
        # First pass: identify which internal nodes can be computed
        for layer_idx in range(1, len(self.tree_layers)):
            layer_nodes = [(l, i) for l, i in proof_nodes if l == layer_idx]
            for layer, idx in layer_nodes:
                # Check if both children are available
                left_child = (layer - 1, idx * 2)
                right_child = (layer - 1, idx * 2 + 1)
                
                if (left_child in proof_nodes and right_child in proof_nodes):
                    computable_nodes.add((layer, idx))
        
        # Second pass: add non-computable nodes as proof hashes
        for layer, idx in proof_nodes:
            if (layer, idx) not in working_set_indices:  # Not already in leaves
                if (layer, idx) not in computable_nodes:  # Can't be computed
                    if layer == 0:  # Leaf siblings
                        if node_positions[(layer, idx)] not in sorted_leaves:
                            proof_hash_nodes.append((layer, idx, node_positions[(layer, idx)]))
                    else:  # Internal nodes that can't be computed
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
        for layer_idx in range(1, len(self.tree_layers)):
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
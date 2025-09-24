#!/usr/bin/env python3
"""
Clustered Flat Tree Builder

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

class Document:
    """Document entity with hierarchical province.property structure."""
    def __init__(self, doc_id, content, province, property_id):
        self.doc_id = doc_id
        self.content = content
        self.province = province
        self.property_id = property_id
        self.full_id = f"{province}.{property_id}"
        self.hash_hex = keccak(self.content.encode('utf-8')).hex()

    def __repr__(self):
        return f"Doc({self.doc_id}, {self.full_id})"

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

class ClusteredFlatTreeBuilder:
    """
    Builds a single flat Merkle tree with two-level custom leaf ordering:
    1. Primary: Cluster by province (jurisdiction)
    2. Secondary: Optimize within province using pairs-first Huffman
    """
    
    def __init__(self, all_documents, traffic_logs):
        self.all_documents = all_documents
        self.merkle_root = None
        self.ordered_leaves_hex = []
        self.tree_layers = []
        
        # Only accept compressed traffic logs
        from compressed_traffic_logs import CompressedTrafficLogs
        if not isinstance(traffic_logs, (CompressedTrafficLogs, type(None))):
            raise ValueError("Only CompressedTrafficLogs instances are accepted. Legacy traffic logs are not supported.")
        
        self.compressed_traffic = traffic_logs if traffic_logs is not None else CompressedTrafficLogs()
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
        """Calculate co-verification frequencies for optimization using compressed traffic logs."""
        # Use pre-calculated frequencies from compressed traffic
        property_freq = dict(self.compressed_traffic.property_frequencies)
        pair_freq = dict(self.compressed_traffic.pair_frequencies)
        
        # Calculate province frequencies from property frequencies
        province_freq = defaultdict(int)
        for prop_id, freq in property_freq.items():
            if '.' in prop_id:
                province = prop_id.split('.')[0]
                province_freq[province] += freq
        
        return province_freq, property_freq, pair_freq
    
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
    
    def _optimize_province_ordering(self, province, property_clusters, property_freq, pair_freq):
        """
        Apply pairs-first Huffman optimization within a province.
        Returns optimally ordered list of property clusters.
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
        
        # Apply pairs-first optimization
        clusters_map = {cluster.property_id: cluster for cluster in property_clusters}
        merged_ids = set()
        merged_clusters = []
        
        # Sort pairs by frequency (most frequent first)
        sorted_pairs = sorted(province_pairs.items(), key=lambda x: x[1], reverse=True)
        
        for (prop1, prop2), freq in sorted_pairs:
            if prop1 not in merged_ids and prop2 not in merged_ids:
                if prop1 in clusters_map and prop2 in clusters_map:
                    # Create merged virtual cluster
                    merged_cluster = [clusters_map[prop1], clusters_map[prop2]]
                    merged_clusters.append(merged_cluster)
                    merged_ids.add(prop1)
                    merged_ids.add(prop2)
        
        # Add remaining unmerged clusters
        remaining_clusters = []
        for prop_id, cluster in clusters_map.items():
            if prop_id not in merged_ids:
                remaining_clusters.append([cluster])
        
        # Combine merged and remaining clusters
        all_cluster_groups = merged_clusters + remaining_clusters
        
        # Apply Huffman coding to cluster groups
        group_frequencies = {}
        for i, group in enumerate(all_cluster_groups):
            total_freq = sum(property_freq.get(cluster.property_id, 1) for cluster in group)
            group_frequencies[i] = total_freq
        
        if len(group_frequencies) > 1:
            huffman_root = self._build_huffman_tree(group_frequencies)
            if huffman_root:
                codes = self._generate_huffman_codes(huffman_root)
                # Sort groups by Huffman codes
                sorted_indices = sorted(range(len(all_cluster_groups)), 
                                      key=lambda i: codes.get(i, ""))
                optimized_groups = [all_cluster_groups[i] for i in sorted_indices]
            else:
                optimized_groups = all_cluster_groups
        else:
            optimized_groups = all_cluster_groups
        
        # Flatten optimized groups back to individual clusters
        optimized_clusters = []
        for group in optimized_groups:
            optimized_clusters.extend(group)
        
        return optimized_clusters
    
    def build(self):
        """
        Build the clustered flat tree with two-level optimization:
        1. Primary: Group by province
        2. Secondary: Optimize within province using pairs-first Huffman
        """
        print("Building Clustered Flat Tree with two-level optimization...")
        
        # Calculate frequencies for optimization
        province_freq, property_freq, pair_freq = self._calculate_frequencies()
        
        # Primary sorting: Group by province (alphabetical order for consistency)
        sorted_provinces = sorted(self.province_clusters.keys())
        
        final_ordered_clusters = []
        
        for province in sorted_provinces:
            property_clusters = list(self.province_clusters[province].values())
            
            print(f"  Optimizing {province}: {len(property_clusters)} properties")
            
            # Secondary sorting: Apply pairs-first Huffman within province
            optimized_clusters = self._optimize_province_ordering(
                province, property_clusters, property_freq, pair_freq
            )
            
            final_ordered_clusters.extend(optimized_clusters)
        
        # Extract ordered leaves from optimized clusters
        self.ordered_leaves_hex = []
        for cluster in final_ordered_clusters:
            self.ordered_leaves_hex.extend(cluster.get_leaf_hashes_hex())
        
        # Remove duplicates while preserving order
        unique_leaves = []
        seen = set()
        for leaf in self.ordered_leaves_hex:
            if leaf not in seen:
                unique_leaves.append(leaf)
                seen.add(leaf)
        
        self.ordered_leaves_hex = unique_leaves
        
        if not self.ordered_leaves_hex:
            self.merkle_root = keccak(b'').hex()
            return self.merkle_root
        
        print(f"  Total unique leaves: {len(self.ordered_leaves_hex)}")
        
        # Build flat Merkle tree bottom-up
        self.tree_layers = []
        current_layer = list(self.ordered_leaves_hex)
        self.tree_layers.append(current_layer)
        
        while len(current_layer) > 1:
            # Pad with last element if odd number
            if len(current_layer) % 2 != 0:
                current_layer.append(current_layer[-1])
            
            next_layer = []
            for i in range(0, len(current_layer), 2):
                parent_hash = combine_and_hash(current_layer[i], current_layer[i + 1])
                next_layer.append(parent_hash)
            
            self.tree_layers.append(next_layer)
            current_layer = next_layer
        
        self.merkle_root = current_layer[0] if current_layer else keccak(b'').hex()
        
        print(f"  Clustered Flat Tree Root: {self.merkle_root[:16]}...")
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
        
        print(f"ClusteredFlatTreeBuilder: Rebuilding with {len(self.daily_verification_patterns)} daily patterns...")
        
        # Add daily patterns to compressed traffic logs
        for pattern in self.daily_verification_patterns:
            self.compressed_traffic.add_verification_event(pattern)
        
        # Rebuild the tree with enhanced traffic data
        try:
            self.build()
            print("ClusteredFlatTreeBuilder: Daily rebuild completed successfully")
            return True
        except Exception as e:
            print(f"ClusteredFlatTreeBuilder: Daily rebuild failed: {e}")
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
        print("ClusteredFlatTreeBuilder: Daily learning reset for new day")
    
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
            "tree_type": "Clustered Flat Tree",
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
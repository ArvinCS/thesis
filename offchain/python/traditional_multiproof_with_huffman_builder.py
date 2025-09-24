"""
Traditional Multiproof Merkle Tree Builder with Pairs-First Huffman Optimization

This module provides a traditional flat Merkle tree implementation with multiproof
generation that incorporates the same pairs-first Huffman optimization as the 
hierarchical approach, but maintains a flat structure for comparison.
"""

import heapq
from collections import defaultdict
from optimized_tree_builder import combine_and_hash, Document, PropertyCluster, MergedCluster
from eth_utils import keccak
from learning_config import get_learning_config, LearningMode


class HuffmanNode:
    """Node for building Huffman tree for optimization."""
    def __init__(self, item, freq):
        self.item, self.freq, self.left, self.right = item, freq, None, None
    def __lt__(self, other): return self.freq < other.freq


def build_huffman_tree(items_with_freq):
    """Build Huffman tree from items and their frequencies."""
    pq = [HuffmanNode(item, freq) for item, freq in items_with_freq.items()]
    heapq.heapify(pq)
    while len(pq) > 1:
        left, right = heapq.heappop(pq), heapq.heappop(pq)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left, merged.right = left, right
        heapq.heappush(pq, merged)
    return pq[0]


def generate_codes_from_tree(root_node):
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


class TraditionalMultiproofWithHuffmanBuilder:
    """Traditional flat Merkle tree with multiproof and pairs-first Huffman optimization with incremental learning."""
    
    def __init__(self, all_documents, traffic_logs=None):
        self.all_documents = all_documents
        self.property_clusters = self._create_property_clusters()
        self.ordered_leaves_hex = []
        self.merkle_root = None
        self.layers = []
        self.last_rebuild_day = -1  # Track which day we last rebuilt
        
        # Enhanced learning tracking
        self.verification_count = 0
        self.rebuilds_today = 0
        self.learning_stats = {
            'total_rebuilds': 0,
            'successful_optimizations': 0,
            'average_improvement': 0.0
        }
        
        # Only accept compressed traffic logs
        from compressed_traffic_logs import CompressedTrafficLogs
        if not isinstance(traffic_logs, (CompressedTrafficLogs, type(None))):
            raise ValueError("Only CompressedTrafficLogs instances are accepted. Legacy traffic logs are not supported.")
        
        self.compressed_traffic = traffic_logs if traffic_logs is not None else CompressedTrafficLogs()
        self.daily_traffic_logs = []  # Accumulate traffic patterns during the day
    
    def _create_property_clusters(self):
        """Create property clusters using full hierarchical IDs."""
        clusters = {}
        
        # Handle both dictionary and list formats
        if isinstance(self.all_documents, dict):
            documents = self.all_documents.values()
        else:
            documents = self.all_documents
        
        for doc in documents:
            if doc.full_id not in clusters:
                clusters[doc.full_id] = PropertyCluster(doc.full_id)
            clusters[doc.full_id].add_document(doc)
        return clusters
    
    def build(self):
        """
        Build traditional flat Merkle tree with pairs-first Huffman optimization.
        This applies the same optimization as the hierarchical approach but in a flat structure.
        """
        if self.compressed_traffic.total_events == 0:
            # Fallback to simple alphabetical ordering if no traffic data
            return self._build_simple()
        
        # --- 1. Use Pre-calculated Frequencies from Compressed Traffic ---
        prop_freq = dict(self.compressed_traffic.property_frequencies)
        pair_freq = dict(self.compressed_traffic.pair_frequencies)
        # Ensure all properties have a base frequency
        for prop_id in self.property_clusters:
            if prop_id not in prop_freq: 
                prop_freq[prop_id] = 0
        
        # --- 2. Apply Pairs-First Optimization ---
        
        # Start with individual property clusters
        items_map = {prop_id: cluster for prop_id, cluster in self.property_clusters.items()}
        
        # pair_freq is already in the correct format from compressed traffic logs
        
        # Get pairs sorted by frequency, from most to least frequent
        sorted_pairs = sorted(pair_freq.items(), key=lambda item: item[1], reverse=True)
        merged_ids = set()
        # Merge most frequent pairs first
        for (prop1_id, prop2_id), freq in sorted_pairs:
            # If both properties in the pair haven't been merged yet, merge them
            if prop1_id not in merged_ids and prop2_id not in merged_ids:
                if prop1_id in items_map and prop2_id in items_map:
                    item1 = items_map[prop1_id]
                    item2 = items_map[prop2_id]
                    
                    new_merged_cluster = MergedCluster(item1, item2)
                    
                    # Update the items map to replace the individual items with the new merged cluster
                    items_map[prop1_id] = new_merged_cluster
                    items_map[prop2_id] = new_merged_cluster

                    # Mark these properties as merged
                    merged_ids.add(prop1_id)
                    merged_ids.add(prop2_id)
        # --- 3. Apply Huffman Coding for Final Ordering ---
        
        # Get the unique set of final items (merged and unmerged clusters)
        final_items = list({id(v): v for v in items_map.values()}.values())
        
        # Calculate the frequency for each final item
        final_items_with_freq = {}
        for item in final_items:
            if isinstance(item, PropertyCluster):
                final_items_with_freq[item] = prop_freq[item.property_id]
            elif isinstance(item, MergedCluster):
                # Frequency of a merged cluster is the sum of its parts
                total_freq = sum(prop_freq.get(p_id, 0) for p_id in item.id)
                final_items_with_freq[item] = total_freq
        # --- 4. Build Huffman Tree and Sort ---
        
        if len(final_items_with_freq) > 1:
            huffman_root = build_huffman_tree(final_items_with_freq)
            codebook = generate_codes_from_tree(huffman_root)
            # Sort items based on Huffman codes
            def get_sort_key(item):
                code = codebook.get(item, "")
                # Secondary sort by property name for consistency
                if isinstance(item, PropertyCluster):
                    return (code, item.property_id)
                else:
                    return (code, str(item))
            
            sorted_final_items = sorted(final_items, key=get_sort_key)
        else:
            # Single item case
            sorted_final_items = final_items
        
        # --- 5. Build the Flat Merkle Tree ---
        self.ordered_leaves_hex = []
        for item in sorted_final_items:
            self.ordered_leaves_hex.extend(item.get_leaf_hashes_hex())
        
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
    
    def _build_simple(self):
        """Fallback to simple alphabetical ordering if no traffic data."""
        # Simple alphabetical ordering
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
        return self.generate_batched_proof_with_flags(document_hashes)
    
    def generate_batched_proof_with_flags(self, leaves_to_prove_hex):
        """Generate multiproof using the correct OpenZeppelin-compatible algorithm."""
        if not self.ordered_leaves_hex:
            return [], []
        
        # Use the working algorithm from optimized_tree_builder
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
                #     print(f"   Leaf order changed: {old_leaf_order[:16]}... -> {self.ordered_leaves_hex[:16]}...")
            
            return should_rebuild, self.merkle_root
            
        except Exception as e:
            print(f"⚠️ Error in add_learning_event: {e}")
            # Fallback: always add to traffic logs and use immediate learning
            self.compressed_traffic.add_verification_event(verified_properties)
            self.verification_count += 1
            self.build()
            return True, self.merkle_root

    def add_verification_event(self, verified_properties, learning_mode=None, end_of_day=False):
        """Add new verification event to compressed traffic logs."""
        return self.add_learning_event(verified_properties, learning_mode, end_of_day=end_of_day)
    
    def _estimate_average_proof_size(self):
        """Estimate average proof size based on tree depth and common patterns."""
        if not self.merkle_root or not self.tree_structure:
            return 0
        
        # Simple estimation: average depth of leaves weighted by frequency
        total_weight = 0
        weighted_depth = 0
        
        for node_id, freq in self.property_frequencies.items():
            if node_id in self.tree_structure:
                depth = self._calculate_node_depth(node_id)
                weighted_depth += depth * freq
                total_weight += freq
        
        if total_weight == 0:
            return len(self.tree_structure.get('children', []))  # Fallback
        
        return weighted_depth / total_weight
    
    def _calculate_node_depth(self, node_id):
        """Calculate depth of a node in the tree."""
        if not self.tree_structure:
            return 1
        
        # Simple depth calculation - would need proper tree traversal for accuracy
        # For now, use a rough estimate based on tree balance
        total_nodes = len(self.property_frequencies)
        if total_nodes <= 1:
            return 1
        
        import math
        return max(1, int(math.log2(total_nodes)) + 1)
    
    def get_learning_stats(self):
        """Get comprehensive learning statistics."""
        config = get_learning_config()
        return {
            'mode': config.mode.value,
            'verification_count': self.verification_count,
            'rebuilds_today': self.rebuilds_today,
            'max_rebuilds_per_day': config.max_rebuilds_per_day,
            'batch_size': config.batch_size if config.mode == LearningMode.BATCH else None,
            'learning_stats': self.learning_stats.copy(),
            'daily_patterns': len(self.daily_traffic_logs),
            'total_patterns': self.compressed_traffic.total_events
        }
    
    def get_compression_stats(self):
        """Get compression statistics from the compressed traffic logs."""
        return self.compressed_traffic.get_statistics()

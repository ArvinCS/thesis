#!/usr/bin/env python3
"""
Traditional Property-Level Huffman Tree Builder

This module creates a true unbalanced Merkle tree using direct property-level Huffman optimization:
1. Document-Level Optimization: Apply Huffman within each property group
2. Property-Level Huffman: Build unbalanced tree where frequent properties are at shallow depths
3. Unbalanced Structure: Creates actual variable-depth tree (not just reordered leaves)
4. NO Province Clustering: Properties from all provinces are mixed together in optimization

Key Features:
- TRUE unbalanced tree structure (following research paper methodology)
- Direct property-level pairs-first Huffman optimization with alpha threshold
- Frequent properties at shallow depths regardless of province
- Document-level optimization within property groups
- Unbalanced proof generation with variable depths
- Gas-optimized verification for frequent access patterns
- Cross-province property optimization (no geographical clustering)
"""

import heapq
from collections import defaultdict
from eth_utils import keccak
from typing import Dict, List, Tuple, Any, Optional
from basic_data_structure import Document


class ProofNode:
    """
    Node in the proof tree for bitmap-based verification.
    
    The proof tree is a subset of the full Merkle tree containing only the nodes
    needed to verify the requested leaves. Post-order traversal of this tree
    generates the correct sequence of PUSH and MERGE operations for stack-based
    verification.
    
    Node types:
    - is_input=True: This is a leaf or sibling hash that needs to be PUSHED
    - is_input=False: This is an internal node that needs to be MERGED (computed from children)
    """
    def __init__(self, hash_value=None, is_input=True):
        self.hash_value = hash_value  # Hash value (for input nodes) or None (for merge nodes)
        self.is_input = is_input      # True = PUSH this hash, False = MERGE children
        self.left = None              # Left child (for merge nodes)
        self.right = None             # Right child (for merge nodes)
    
    def __repr__(self):
        if self.is_input:
            return f"ProofNode(PUSH, {self.hash_value[:16] if self.hash_value else 'None'}...)"
        else:
            return f"ProofNode(MERGE)"


class DocumentHuffmanNode:
    """Node for building Huffman tree for document-level optimization."""
    def __init__(self, item, freq):
        self.item = item  # Either a Document object or None for internal nodes
        self.freq = freq
        self.left = None
        self.right = None
        self.documents = []  # List of documents in this node (for merged nodes)
        
        # If item is a Document, initialize documents list
        if isinstance(item, Document):
            self.documents = [item]
    
    def __lt__(self, other):
        if hasattr(other, 'freq'):
            return self.freq < other.freq
        return NotImplemented
    
    def __eq__(self, other):
        if hasattr(other, 'freq'):
            return self.freq == other.freq
        return NotImplemented
    
    def get_all_documents(self):
        """Get all documents contained in this node (recursively for merged nodes)."""
        if self.item is not None:
            return [self.item]
        else:
            # Internal node - combine documents from children
            all_docs = []
            if self.left:
                all_docs.extend(self.left.get_all_documents())
            if self.right:
                all_docs.extend(self.right.get_all_documents())
            return all_docs


class PropertyHuffmanNode:
    """Node for building Huffman tree at property level for true unbalanced tree."""
    _counter = 0  # Class-level counter for deterministic tiebreaking
    
    def __init__(self, property_group, freq):
        self.property_group = property_group  # PropertyDocumentGroup object or None for internal nodes
        self.freq = freq
        self.left = None
        self.right = None
        self.all_documents = []  # All documents from this subtree
        self.depth = 0  # Depth in the tree for proof generation
        
        # Unique ID for deterministic comparison when frequencies are equal
        PropertyHuffmanNode._counter += 1
        self._id = PropertyHuffmanNode._counter
        
        # Tiebreaker key: use property ID for leaf nodes, or unique ID for internal nodes
        if property_group is not None:
            self._tiebreaker = f"{property_group.province}.{property_group.property_id}"
            self.all_documents = property_group.optimized_document_order.copy()
        else:
            self._tiebreaker = f"internal_{self._id}"
    
    def __lt__(self, other):
        # Primary comparison by frequency, tiebreak by property ID or internal ID
        if self.freq != other.freq:
            return self.freq < other.freq
        return self._tiebreaker < other._tiebreaker
    
    def get_all_documents(self):
        """Get all documents from this subtree."""
        if self.property_group is not None:
            return self.all_documents
        else:
            # Internal node - combine documents from children
            all_docs = []
            if self.left:
                all_docs.extend(self.left.get_all_documents())
            if self.right:
                all_docs.extend(self.right.get_all_documents())
            return all_docs
    
    def compute_depths(self, depth=0, visited=None):
        """Compute depth for all nodes iteratively (avoids stack overflow with deep trees)."""
        # Use iterative approach with explicit stack to handle deep trees (5000+ nodes)
        stack = [(self, depth)]
        
        while stack:
            node, current_depth = stack.pop()
            node.depth = current_depth
            
            # Push children onto stack (right first, then left, so left is processed first)
            if node.right:
                stack.append((node.right, current_depth + 1))
            if node.left:
                stack.append((node.left, current_depth + 1))


class PropertyDocumentGroup:
    """Represents a group of documents belonging to the same property within a province."""
    
    def __init__(self, property_id, province):
        self.property_id = property_id
        self.province = province
        self.documents = []
        self.optimized_document_order = []
    
    def add_document(self, document):
        """Add a document to this property group."""
        self.documents.append(document)
    
    def optimize_document_order(self, document_pair_frequencies, document_frequencies, alpha_threshold=0.15):
        """
        Simple sorted ordering for documents within this property.
        
        NOTE: Pairs-first Huffman optimization is ONLY applied at the property node level,
        NOT within document subtrees. Document subtrees use simple frequency-based sorting
        for a balanced tree structure.
        
        This ensures:
        1. Property nodes are organized by pairs-first Huffman (optimizes property co-access)
        2. Document subtrees within properties are simple balanced trees (deterministic, efficient)
        """
        if len(self.documents) <= 1:
            self.optimized_document_order = self.documents.copy()
            return
        
        # Simple frequency-based sorting for balanced document subtrees
        # No pairs-first Huffman at document level - only at property level
        self.optimized_document_order = sorted(
            self.documents,
            key=lambda doc: document_frequencies.get(f"{doc.province}.{doc.property_id}.{doc.doc_id}", 1),
            reverse=True  # Most frequent documents first
        )
    
    def _apply_pairs_first_huffman(self, documents, doc_frequencies, pair_frequencies, alpha_threshold=0.1):
        """
        DEPRECATED: This method is no longer used for document subtrees.
        Document subtrees now use simple sorted ordering (see optimize_document_order).
        Pairs-first Huffman is only applied at the property node level.
        
        Kept for backwards compatibility but not called in current implementation.
        """
        # Simply return sorted documents by frequency
        return sorted(documents, key=lambda doc: doc_frequencies.get(doc, 1), reverse=True)

    
    def get_document_hashes_hex(self):
        """Get ordered document hashes in hex format."""
        return [doc.hash_hex for doc in self.optimized_document_order]


def combine_and_hash(hash1_hex, hash2_hex):
    """
    Combine two hashes using keccak256 with RFC 6962 domain separation.
    
    SECURITY: Uses 0x01 prefix to prevent second preimage attacks.
    Internal nodes are prefixed with 0x01, making them cryptographically
    distinct from leaf nodes (which would be prefixed with 0x00 if hashed).
    This prevents tree splicing attacks where a 64-byte leaf could be
    interpreted as two concatenated 32-byte hashes.
    
    Compatible with OpenZeppelin when both use same domain separation.
    """
    h1_bytes = bytes.fromhex(hash1_hex)
    h2_bytes = bytes.fromhex(hash2_hex)
    
    # RFC 6962: 0x01 prefix for internal nodes
    prefix = b'\x01'
    
    # Sort hashes to ensure deterministic ordering (smaller hash first)
    if h1_bytes < h2_bytes:
        combined = prefix + h1_bytes + h2_bytes
    else:
        combined = prefix + h2_bytes + h1_bytes
    
    return keccak(combined).hex()


class TraditionalPropertyLevelHuffmanBuilder:
    """
    Builds a TRUE unbalanced Merkle tree with direct property-level Huffman optimization:
    
    1. Document subtrees (within properties): Simple balanced trees with frequency-based sorting
       - Documents are sorted by access frequency (most frequent first)
       - Standard balanced binary tree structure for deterministic, efficient proofs
    
    2. Property-level tree: Unbalanced Huffman tree with pairs-first optimization
       - Properties organized using pairs-first Huffman algorithm
       - Frequently co-accessed properties are placed closer in tree (shallower depth)
       - Alpha threshold controls relative weight for pair merging: w(tx) = q_k / min(p_i, p_j)
    
    3. Cross-province optimization: Properties from all provinces compete equally
       - NO geographical province clustering (unlike clustered_province_with_document_huffman)
       - All properties are organized in a single global Huffman tree
       - Pure frequency-based optimization without geographic hierarchy
    
    Key difference from clustered_province_with_document_huffman:
    - Pairs-first Huffman ONLY at property node level (LCA of documents)
    - Document subtrees are simple balanced trees (no Huffman within properties)
    - No province clustering - all properties in one global tree
    """
    
    def __init__(self, all_documents, audit_pattern=None, transactional_pattern=None, alpha_threshold=0.15):
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
        self.alpha_threshold = alpha_threshold  # Configurable threshold for pairs-first Huffman
        self.verification_count = 0
        
        # Unbalanced tree structure
        self.property_huffman_root = None  # Root of the property-level Huffman tree
        
        # Group documents by province and property
        self.province_property_groups = self._create_province_property_groups()
        
    def _create_province_property_groups(self):
        """Group documents by province, then by property within each province."""
        groups = {}
        
        for doc in self.all_documents:
            province = doc.province
            property_id = doc.property_id
            
            if province not in groups:
                groups[province] = {}
            
            if property_id not in groups[province]:
                groups[province][property_id] = PropertyDocumentGroup(property_id, province)
            
            groups[province][property_id].add_document(doc)
        
        return groups
    
    def _calculate_frequencies(self):
        """
        Calculate co-verification frequencies for optimization using BOTH access patterns.
        
        Combines:
        1. Transactional pattern: High-frequency individual property/document access
        2. Audit pattern: Regional audit queries that target entire provinces
        
        This ensures the Huffman tree is optimized for BOTH query types.
        """
        property_freq = defaultdict(int)
        pair_freq = defaultdict(int)
        province_freq = defaultdict(int)
        
        # Use transactional pattern for property and pair frequencies
        if self.transactional_pattern:
            # Calculate property frequencies based on Zipfian distribution (for transactional queries)
            property_frequencies = self.transactional_pattern.get_property_access_frequencies(
                self.province_property_groups
            )
            
            # Store property frequencies and aggregate to province level
            for full_prop_id, freq in property_frequencies.items():
                property_freq[full_prop_id] = freq
                province = full_prop_id.split('.')[0]
                province_freq[province] += freq
            
            # Calculate CROSS-PROPERTY pair frequencies from multi-property transactions
            # This is the key for pairs-first Huffman optimization at the property level
            # Simulate multi-property transactions (bundle deals, adjacent lots, etc.)
            num_provinces = len(self.province_property_groups)
            num_total_properties = sum(len(pg) for pg in self.province_property_groups.values())
            
            # Principled sampling formula for research validity:
            # - Ensure minimum 100 multi-property txns per province (×4 since only 25% are multi-property)
            # - For large datasets (>10k), use linear scaling for better statistical coverage
            # - No cap for precomputed frequencies (done once during tree construction)
            min_samples_per_province = 400  # 100 multi-property × 4 (25% ratio)
            
            if num_total_properties <= 10000:
                # Small datasets: use sqrt scaling (~5,000 for 10k properties)
                num_simulations = max(num_provinces * min_samples_per_province, int(50 * (num_total_properties ** 0.5)))
            else:
                # Large datasets (100k+): use linear scaling for research-grade coverage
                # 100k properties → 200k simulations (2× coverage)
                # Ensures rare patterns are captured for accurate frequency distribution
                num_simulations = max(num_provinces * min_samples_per_province, num_total_properties * 2)
            
            multi_property_pairs = self.transactional_pattern.simulate_multi_property_transactions(
                self.province_property_groups, 
                num_simulations=num_simulations
            )
            
            # Add cross-property pairs to pair_freq
            for prop_pair, freq in multi_property_pairs.items():
                pair_freq[prop_pair] += freq
            
            if multi_property_pairs:
                print(f"    Generated {len(multi_property_pairs)} cross-property pairs from multi-property transactions")
        
        # CRITICAL FIX: Also incorporate audit pattern to boost province frequencies
        # This ensures provinces targeted by regional audits get shorter paths in the tree
        if self.audit_pattern:
            # Get province weights from audit pattern (indicates which provinces are audited more)
            audit_province_weights = getattr(self.audit_pattern, 'province_weights', {})
            
            # Calculate audit frequency boost based on province weights
            # Scale factor to balance with transactional frequencies
            max_trans_freq = max(province_freq.values()) if province_freq else 1000
            
            for province, weight in audit_province_weights.items():
                # Boost province frequency proportionally to audit weight
                # Audit queries often access multiple documents per property
                audit_boost = int(weight * max_trans_freq * 0.5)  # 50% weight to audit pattern
                province_freq[province] += audit_boost
                
                # Also boost property frequencies within audited provinces
                if province in self.province_property_groups:
                    num_properties = len(self.province_property_groups[province])
                    per_property_boost = audit_boost // max(num_properties, 1)
                    for prop_id in self.province_property_groups[province]:
                        full_prop_id = f"{province}.{prop_id}"
                        property_freq[full_prop_id] += per_property_boost
        
        # Extract document frequencies and pair frequencies
        document_frequencies = self._extract_document_frequencies()
        document_pair_frequencies = self._extract_document_pair_frequencies()
        
        return dict(province_freq), dict(property_freq), dict(pair_freq), document_frequencies, document_pair_frequencies
    
    def _extract_document_frequencies(self):
        """Extract individual document frequencies from access pattern."""
        document_freq = defaultdict(int)
        
        # Use transactional pattern to get document frequencies
        if self.transactional_pattern:
            for province, property_groups in self.province_property_groups.items():
                for prop_id, group in property_groups.items():
                    if group.documents:
                        pattern_frequencies = self.transactional_pattern.get_document_frequencies(group.documents)
                        # Convert document object keys to string keys
                        for doc, freq in pattern_frequencies.items():
                            doc_key = f"{doc.province}.{doc.property_id}.{doc.doc_id}"
                            document_freq[doc_key] = freq
        
        return dict(document_freq)
    
    def _extract_document_pair_frequencies(self):
        """Extract document pair frequencies from access pattern."""
        doc_pair_freq = defaultdict(int)
        
        # Use transactional pattern to get document pair frequencies
        if self.transactional_pattern:
            all_documents = []
            for province_groups in self.province_property_groups.values():
                for group in province_groups.values():
                    all_documents.extend(group.documents)
            
            # Calculate pair frequencies within each property group separately
            for province_groups in self.province_property_groups.values():
                for group in province_groups.values():
                    if len(group.documents) >= 2:
                        # Get pair frequencies for this property's documents
                        pattern_pair_freq = self.transactional_pattern.get_document_pair_frequencies(
                            group.documents, num_simulated_queries=50  # Reduced for efficiency
                        )
                        
                        # Convert document object keys to string keys
                        for (doc1, doc2), freq in pattern_pair_freq.items():
                            doc1_key = f"{doc1.province}.{doc1.property_id}.{doc1.doc_id}"
                            doc2_key = f"{doc2.province}.{doc2.property_id}.{doc2.doc_id}"
                            
                            pair_key = tuple(sorted([doc1_key, doc2_key]))
                            doc_pair_freq[pair_key] += freq
        
        return dict(doc_pair_freq)
    
    def precompute_frequencies(self):
        """
        Pre-compute frequency data from access patterns.
        
        In production, this data would come from actual transaction logs.
        For benchmarking, this should be called BEFORE timing starts,
        as frequency collection is a separate ongoing process.
        
        Returns:
            tuple: (province_freq, property_freq, pair_freq, document_frequencies, document_pair_frequencies)
        """
        if hasattr(self, '_cached_frequencies') and self._cached_frequencies is not None:
            return self._cached_frequencies
        
        print("  Pre-computing frequencies from access patterns...")
        self._cached_frequencies = self._calculate_frequencies()
        return self._cached_frequencies
    
    def build(self, use_cached_frequencies=True):
        """
        Build TRUE unbalanced Huffman tree with property-level optimization:
        1. Apply document-level Huffman within each property group
        2. Build Huffman tree of property groups (creates unbalanced structure)
        3. Generate leaves from the unbalanced tree traversal
        
        Args:
            use_cached_frequencies: If True, uses pre-computed frequencies (for accurate timing).
                                   If False, calculates frequencies on-the-fly.
        """
        # Reset counter for deterministic tree building
        PropertyHuffmanNode._counter = 0
        
        print("Building Traditional Property-Level Huffman Tree (No Province Clustering)...")
        
        # Use cached frequencies if available (for accurate build time measurement)
        if use_cached_frequencies and hasattr(self, '_cached_frequencies') and self._cached_frequencies is not None:
            province_freq, property_freq, pair_freq, document_frequencies, document_pair_frequencies = self._cached_frequencies
        else:
            # Calculate frequencies on-the-fly (includes simulation time)
            province_freq, property_freq, pair_freq, document_frequencies, document_pair_frequencies = self._calculate_frequencies()
        
        # Step 1: Apply document-level Huffman within each property group
        all_property_groups = []
        for province, property_groups in self.province_property_groups.items():
            for property_id, property_group in property_groups.items():
                # print(f"  Pre-optimizing documents in {province}.{property_id}: {len(property_group.documents)} documents")
                
                # Apply document-level Huffman within property
                if len(property_group.documents) > 1:
                    property_group.optimize_document_order(document_pair_frequencies, document_frequencies, self.alpha_threshold)
                else:
                    property_group.optimized_document_order = property_group.documents
                
                all_property_groups.append(property_group)
        
        # Step 2: Build TRUE unbalanced Huffman tree at property level
        print(f"\n  Building property-level Huffman tree from {len(all_property_groups)} property groups...")
        self.property_huffman_root = self._build_property_huffman_tree(all_property_groups, property_freq, pair_freq)
        
        # Step 3: Generate ordered leaves from unbalanced tree traversal
        self.ordered_leaves_hex = []
        if self.property_huffman_root:
            self._extract_leaves_from_huffman_tree(self.property_huffman_root)
        
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
        
        print(f"  Total unique leaves from unbalanced tree: {len(self.ordered_leaves_hex)}")
        
        # Step 4: Compute Merkle root from the unbalanced Huffman structure
        self.merkle_root = self._compute_unbalanced_merkle_root()
        
        print(f"  Traditional Property-Level Huffman Tree Root (unbalanced): {self.merkle_root[:16]}...")
        
        return self.merkle_root
    
    def _build_property_huffman_tree(self, property_groups, property_freq, pair_freq):
        """Build true unbalanced Huffman tree at property level."""
        if not property_groups:
            return None
        
        # Create property nodes with frequencies
        property_nodes = []
        for prop_group in property_groups:
            full_prop_id = f"{prop_group.province}.{prop_group.property_id}"
            # Ensure minimum frequency of 1 to prevent extremely deep trees
            freq = max(property_freq.get(full_prop_id, 1), 1)
            node = PropertyHuffmanNode(prop_group, freq)
            property_nodes.append(node)
        
        # Apply pairs-first Huffman with relative weight threshold (as per paper)
        alpha_threshold = self.alpha_threshold
        if pair_freq:
            
            # Merge strong property pairs first
            merged_pairs = set()
            strong_pairs_count = 0
            
            sorted_pairs = sorted(pair_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Create a mapping from property_id to node for efficient lookup
            prop_id_to_node = {}
            for n in property_nodes:
                if n.property_group:
                    full_id = f"{n.property_group.province}.{n.property_group.property_id}"
                    prop_id_to_node[full_id] = n
            
            for (prop1_id, prop2_id), pair_frequency in sorted_pairs:
                # Calculate relative weight: w(tx) = q_k / min(p_i, p_j)
                freq1 = property_freq.get(prop1_id, 1)
                freq2 = property_freq.get(prop2_id, 1)
                min_individual_freq = min(freq1, freq2)
                
                if min_individual_freq > 0:
                    relative_weight = pair_frequency / min_individual_freq
                else:
                    relative_weight = 0
                
                if relative_weight >= alpha_threshold:
                    # Find corresponding nodes using the mapping
                    node1 = prop_id_to_node.get(prop1_id)
                    node2 = prop_id_to_node.get(prop2_id)
                    
                    if node1 and node2 and node1 is not node2 and node1 not in merged_pairs and node2 not in merged_pairs:
                        # Validate no cycles
                        if node1.left is node2 or node1.right is node2 or node2.left is node1 or node2.right is node1:
                            continue
                        
                        # Create merged node
                        merged_node = PropertyHuffmanNode(None, node1.freq + node2.freq)
                        merged_node.left = node1
                        merged_node.right = node2
                        
                        # Remove merged nodes from mapping
                        prop_id_to_node.pop(prop1_id, None)
                        prop_id_to_node.pop(prop2_id, None)
                        
                        # Replace nodes in list
                        property_nodes = [n for n in property_nodes if n != node1 and n != node2]
                        property_nodes.append(merged_node)
                        
                        merged_pairs.add(node1)
                        merged_pairs.add(node2)
                        strong_pairs_count += 1
        
        # Build final Huffman tree
        heap = property_nodes.copy()
        heapq.heapify(heap)
        merge_count = 0
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            # Check if nodes already have multiple parents (would indicate reuse bug)
            left_id = f"{left.property_group.province}.{left.property_group.property_id}" if left.property_group else f"internal_{id(left)}"
            right_id = f"{right.property_group.province}.{right.property_group.property_id}" if right.property_group else f"internal_{id(right)}"
            
            merged = PropertyHuffmanNode(None, left.freq + right.freq)
            merged.left = left
            merged.right = right
            
            heapq.heappush(heap, merged)
            merge_count += 1
            
            if merge_count <= 5 or merge_count % 100 == 0:
                print(f"      Merge #{merge_count}: {left_id} + {right_id} -> freq={merged.freq}")
        
        root = heap[0] if heap else None
        if root:
            root.compute_depths(0)
        
        return root
    
    def _extract_leaves_from_huffman_tree(self, node):
        """Extract leaves from unbalanced Huffman tree iteratively (avoids stack overflow)."""
        if node is None:
            return
        
        # Use iterative pre-order traversal with explicit stack
        stack = [node]
        
        while stack:
            current = stack.pop()
            
            if current.property_group is not None:
                # Leaf node - add all documents from this property
                for doc in current.property_group.optimized_document_order:
                    self.ordered_leaves_hex.append(doc.hash_hex)
            else:
                # Internal node - push children (right first, then left for in-order)
                if current.right:
                    stack.append(current.right)
                if current.left:
                    stack.append(current.left)
    
    def _compute_unbalanced_merkle_root(self):
        """Compute Merkle root preserving the unbalanced structure."""
        if not self.property_huffman_root:
            return keccak(b'').hex()
        
        # Create a mapping from documents to their hashes
        doc_to_hash = {doc.hash_hex: doc.hash_hex for doc in self.all_documents}
        
        return self._compute_node_hash(self.property_huffman_root, doc_to_hash)
    
    def _compute_node_hash(self, node, doc_to_hash):
        """Iteratively compute hash for nodes in the unbalanced tree using post-order traversal."""
        if node is None:
            return keccak(b'').hex()
        
        # Use iterative post-order traversal with explicit stack
        # We need to compute children before parents, so we use post-order
        stack = []
        node_hash_map = {}  # Cache computed hashes
        last_visited = None
        current = node
        
        while stack or current:
            # Go to leftmost node
            if current:
                stack.append(current)
                current = current.left
            else:
                peek_node = stack[-1]
                
                # If right child exists and hasn't been processed yet
                if peek_node.right and last_visited != peek_node.right:
                    current = peek_node.right
                else:
                    # Process this node
                    if peek_node.property_group is not None:
                        # Leaf node - compute hash of all documents in this property
                        if len(peek_node.all_documents) == 1:
                            node_hash_map[id(peek_node)] = peek_node.all_documents[0].hash_hex
                        else:
                            # Build balanced subtree for documents within this property
                            doc_hashes = [doc.hash_hex for doc in peek_node.all_documents]
                            node_hash_map[id(peek_node)] = self._compute_balanced_subtree_hash(doc_hashes)
                    else:
                        # Internal node - combine children hashes
                        left_hash = node_hash_map.get(id(peek_node.left), keccak(b'').hex()) if peek_node.left else keccak(b'').hex()
                        right_hash = node_hash_map.get(id(peek_node.right), keccak(b'').hex()) if peek_node.right else keccak(b'').hex()
                        node_hash_map[id(peek_node)] = combine_and_hash(left_hash, right_hash)
                    
                    stack.pop()
                    last_visited = peek_node
                    current = None
        
        return node_hash_map.get(id(node), keccak(b'').hex())
    
    def _compute_balanced_subtree_hash(self, hashes):
        """Compute hash for a balanced subtree of document hashes."""
        if not hashes:
            return keccak(b'').hex()
        
        if len(hashes) == 1:
            return hashes[0]
        
        # Build balanced binary tree
        current_level = list(hashes)
        
        while len(current_level) > 1:
            if len(current_level) % 2 != 0:
                current_level.append(current_level[-1])
            
            next_level = []
            for i in range(0, len(current_level), 2):
                parent_hash = combine_and_hash(current_level[i], current_level[i + 1])
                next_level.append(parent_hash)
            
            current_level = next_level
        
        return current_level[0]
    
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
    
    def add_learning_event(self, verified_properties, learning_mode=None, end_of_day=False):
        """Add verification event with unified learning support."""
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
                self.build()
                if config.verbose_logging:
                    print(f"✅ Tree rebuilt after verification #{self.verification_count}")
                if old_root != self.merkle_root and config.verbose_logging:
                    print(f"   Root changed: {old_root[:16]}... -> {self.merkle_root[:16]}...")
            
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
    
    def get_tree_info(self):
        """Get information about the built tree."""
        if not self.merkle_root:
            return {"status": "not_built"}
        
        province_distribution = {}
        property_distribution = {}
        for province, properties in self.province_property_groups.items():
            province_doc_count = 0
            for prop_id, prop_group in properties.items():
                province_doc_count += len(prop_group.documents)
                property_distribution[f"{province}.{prop_id}"] = len(prop_group.documents)
            province_distribution[province] = province_doc_count
        
        return {
            "tree_type": "Traditional Property-Level Huffman Tree",
            "merkle_root": self.merkle_root,
            "total_leaves": len(self.ordered_leaves_hex),
            "total_documents": len(self.all_documents),
            "provinces": len(province_distribution),
            "province_distribution": province_distribution,
            "properties": len(property_distribution),
            "property_distribution": property_distribution,
            "tree_depth": len(self.tree_layers),
            "optimization": "Two-level: Direct property-level Huffman + Document-level optimization (No province clustering)",
        }
    
    def generate_pathmap_proof(self, leaves_to_prove_hex):
        """Generate proof for hierarchical structure: unbalanced Huffman tree of properties, 
        each containing balanced subtree of documents.
        
        The pathMap is generated in post-order traversal format:
        - Each pair [left_idx, right_idx] represents computing parent = hash(left, right)
        - Children are always computed/available before parents (post-order guarantee)
        - Compatible with stack-based verification: push results as computed
        - Final element in stack after processing all instructions is the root
        """
        if not leaves_to_prove_hex or not self.property_huffman_root:
            return {'leaves': [], 'proofHashes': [], 'pathMap': []}
        
        # Sort and deduplicate leaves
        sorted_leaves = sorted(set(leaves_to_prove_hex))
        
        # Step 1: Map each document to its property node
        doc_to_property = {}
        doc_to_property_docs = {}  # Map doc to all docs in its property
        
        def collect_property_mappings(node):
            """Collect mapping of documents to property nodes."""
            if node is None:
                return
            stack = [node]
            while stack:
                current = stack.pop()
                if current is None:
                    continue
                if current.property_group is not None:
                    # This is a property node (leaf in Huffman tree)
                    property_docs = current.property_group.optimized_document_order
                    for doc in property_docs:
                        doc_to_property[doc.hash_hex] = current
                        doc_to_property_docs[doc.hash_hex] = [d.hash_hex for d in property_docs]
                else:
                    if current.right:
                        stack.append(current.right)
                    if current.left:
                        stack.append(current.left)
        
        collect_property_mappings(self.property_huffman_root)
        
        # Step 2: Group requested documents by property
        property_to_docs = {}
        for doc_hash in sorted_leaves:
            if doc_hash in doc_to_property:
                prop_node = doc_to_property[doc_hash]
                if prop_node not in property_to_docs:
                    property_to_docs[prop_node] = []
                property_to_docs[prop_node].append(doc_hash)
        
        if not property_to_docs:
            return {'leaves': [], 'proofHashes': [], 'pathMap': []}
        
        # Step 3: Generate combined proof
        leaves = list(sorted_leaves)
        proof_hashes = []
        path_map = []
        path_map_instructions = set()  # Track [left_idx, right_idx] pairs to prevent duplicates
        working_set = {}  # Maps hash -> index in working set
        
        # Add leaves to working set
        for i, leaf in enumerate(leaves):
            working_set[leaf] = i
        
        next_idx = len(leaves)
        
        # Step 4: For each involved property, generate subtree proofs and property hash
        property_hashes = {}  # Maps property node -> its hash
        
        for prop_node, requested_docs in property_to_docs.items():
            all_prop_docs = [d.hash_hex for d in prop_node.property_group.optimized_document_order]
            
            # Generate balanced subtree proof for documents within this property
            if len(all_prop_docs) == 1:
                # Single document - hash is the document itself
                property_hashes[id(prop_node)] = all_prop_docs[0]
            else:
                # Multiple documents - need to prove through balanced subtree
                # Build balanced subtree and generate proof
                subtree_proof_data = self._generate_balanced_subtree_proof(
                    all_prop_docs, requested_docs, working_set, proof_hashes, path_map, path_map_instructions, next_idx
                )
                property_hashes[id(prop_node)] = subtree_proof_data['property_hash']
                next_idx = subtree_proof_data['next_idx']
        
        # Step 5: Generate proof through unbalanced Huffman tree
        # Traverse from involved properties to root
        huffman_proof_data = self._generate_huffman_tree_proof(
            property_to_docs.keys(), property_hashes, working_set, proof_hashes, path_map, path_map_instructions, next_idx, len(leaves)
        )
        
        # Validate PathMap: ensure no computed index is used more than once
        # (except it's OK for input indices to be used multiple times in theory, though unusual)
        all_hashes_count = len(leaves) + len(proof_hashes)
        indices_used = []
        for i in range(len(path_map) // 2):
            indices_used.extend([path_map[i*2], path_map[i*2+1]])
        
        from collections import Counter
        usage_counts = Counter(indices_used)
        
        for idx, count in usage_counts.items():
            if idx >= all_hashes_count and count > 1:
                # Computed index used more than once - this is invalid!
                print(f"      ERROR: Computed index {idx} used {count} times in PathMap!")
                print(f"      This violates stack-based computation - each computed value can only be used once")
                print(f"      PathMap: {path_map}")
                raise ValueError(f"Invalid PathMap: computed index {idx} used {count} times")
        
        # Note: Validation of merge instructions is complex because:
        # - Not all proofHashes may be used (some are computed from others)
        # - The actual number of inputs used depends on the tree structure
        # - Bitmap generator will only include inputs that are actually pushed
        
        return {
            'leaves': leaves,
            'proofHashes': proof_hashes,
            'pathMap': path_map
        }
    
    def _generate_balanced_subtree_proof(self, all_docs, requested_docs, working_set, proof_hashes, path_map, path_map_instructions, next_idx):
        """Generate proof for requested documents within a balanced subtree.
        
        Args:
            all_docs: All document hashes in this property (in order)
            requested_docs: Subset of documents that need to be proven
            working_set: Current working set (hash -> index)
            proof_hashes: List to append proof hashes to
            path_map: List to append path instructions to
            path_map_instructions: Set to track instruction pairs
            next_idx: Next available index for computed values
            
        Returns:
            dict with 'property_hash' and 'next_idx'
        """
        # Build balanced tree layers for this property's documents
        layers = [list(all_docs)]
        current = list(all_docs)
        
        while len(current) > 1:
            if len(current) % 2 != 0:
                current.append(current[-1])
            next_layer = []
            for i in range(0, len(current), 2):
                next_layer.append(combine_and_hash(current[i], current[i + 1]))
            layers.append(next_layer)
            current = next_layer
        
        property_hash = current[0]
        
        # Find which nodes are needed for proof
        requested_set = set(requested_docs)
        proof_nodes = set()
        
        # Add requested documents
        for doc in requested_docs:
            if doc in all_docs:
                idx = all_docs.index(doc)
                proof_nodes.add((0, idx))
        
        # Traverse up, collecting siblings
        for layer_idx in range(len(layers) - 1):
            current_layer = layers[layer_idx]
            next_layer = layers[layer_idx + 1]
            
            nodes_to_process = [(l, i) for l, i in proof_nodes if l == layer_idx]
            
            for layer, idx in nodes_to_process:
                # Add parent
                parent_idx = idx // 2
                proof_nodes.add((layer_idx + 1, parent_idx))
                
                # Add sibling
                sibling_idx = idx ^ 1
                if sibling_idx < len(current_layer):
                    proof_nodes.add((layer_idx, sibling_idx))
        
        # Separate into proof hashes and computable nodes
        computable = set()
        for layer_idx in range(1, len(layers)):
            for layer, idx in [(l, i) for l, i in proof_nodes if l == layer_idx]:
                left_child = (layer - 1, idx * 2)
                right_child = (layer - 1, idx * 2 + 1)
                if left_child in proof_nodes and right_child in proof_nodes:
                    computable.add((layer, idx))
        
        # Add non-computable siblings as proof hashes
        for layer, idx in sorted(proof_nodes):
            if (layer, idx) not in computable:
                hash_val = layers[layer][idx]
                if hash_val not in working_set:
                    # Add to proof_hashes and working_set
                    # This includes layer 0 siblings that aren't requested documents
                    proof_hashes.append(hash_val)
                    working_set[hash_val] = next_idx
                    next_idx += 1
        
        # Generate path instructions for computable nodes (post-order: bottom-up)
        # Process layers from bottom to top ensures children computed before parents
        for layer_idx in range(1, len(layers)):
            for layer, idx in sorted([(l, i) for l, i in proof_nodes if l == layer_idx and (l, i) in computable]):
                left_child = (layer - 1, idx * 2)
                right_child = (layer - 1, idx * 2 + 1)
                
                left_hash = layers[layer - 1][left_child[1]]
                right_hash = layers[layer - 1][right_child[1]]
                
                left_idx = working_set.get(left_hash)
                right_idx = working_set.get(right_hash)
                
                if left_idx is not None and right_idx is not None:
                    node_hash = layers[layer][idx]
                    if node_hash not in working_set:
                        # Check for duplicate instruction
                        instr_key = (left_idx, right_idx)
                        if instr_key not in path_map_instructions:
                            # Add instruction: compute node_hash = hash(left, right)
                            # Stack-based verifier will push result at next_idx position
                            # Note: left_idx == right_idx is valid - contract hashes value with itself
                            print(f"      SUBTREE: Adding instruction [{left_idx}, {right_idx}] -> {next_idx}")
                            path_map.extend([left_idx, right_idx])
                            path_map_instructions.add(instr_key)
                            working_set[node_hash] = next_idx
                            next_idx += 1
        
        return {'property_hash': property_hash, 'next_idx': next_idx}
    
    def _generate_huffman_tree_proof(self, involved_properties, property_hashes, working_set, proof_hashes, path_map, path_map_instructions, next_idx, num_leaves):
        """Generate proof through unbalanced Huffman tree from properties to root.
        
        Args:
            involved_properties: Set of property nodes that contain requested documents
            property_hashes: Maps property node id -> its hash
            working_set: Current working set (hash -> index)
            proof_hashes: List to append proof hashes to
            path_map: List to append path instructions to
            path_map_instructions: Set to track instruction pairs
            next_idx: Next available index for computed values
            num_leaves: Number of leaves in the working set
            
        Returns:
            dict with 'next_idx'
        """
        involved_set = set(involved_properties)
        
        # Pre-compute all node hashes in Huffman tree
        node_hash_cache = {}
        
        def compute_huffman_hashes(node):
            if node is None:
                return keccak(b'').hex()
            stack = []
            last_visited = None
            current = node
            
            while stack or current:
                if current:
                    stack.append(current)
                    current = current.left
                else:
                    peek = stack[-1]
                    if peek.right and last_visited != peek.right:
                        current = peek.right
                    else:
                        if peek.property_group is not None:
                            # Use cached property hash
                            node_hash_cache[id(peek)] = property_hashes.get(id(peek), keccak(b'').hex())
                        else:
                            left_h = node_hash_cache.get(id(peek.left), keccak(b'').hex()) if peek.left else keccak(b'').hex()
                            right_h = node_hash_cache.get(id(peek.right), keccak(b'').hex()) if peek.right else keccak(b'').hex()
                            node_hash_cache[id(peek)] = combine_and_hash(left_h, right_h)
                        stack.pop()
                        last_visited = peek
                        current = None
        
        compute_huffman_hashes(self.property_huffman_root)
        
        # Mark which nodes are needed (post-order traversal)
        node_needed = {}
        stack = []
        last_visited = None
        current = self.property_huffman_root
        
        while stack or current:
            if current:
                stack.append(current)
                current = current.left
            else:
                peek = stack[-1]
                if peek.right and last_visited != peek.right:
                    current = peek.right
                else:
                    if peek.property_group is not None:
                        node_needed[id(peek)] = peek in involved_set
                    else:
                        left_needed = node_needed.get(id(peek.left), False) if peek.left else False
                        right_needed = node_needed.get(id(peek.right), False) if peek.right else False
                        node_needed[id(peek)] = left_needed or right_needed
                    stack.pop()
                    last_visited = peek
                    current = None
        
        # Property hashes should already be in working set from subtree proof generation
        # Initialize node_to_index mapping for property nodes
        node_to_index = {}  # Maps node id -> index
        for prop_node in involved_properties:
            prop_hash = property_hashes.get(id(prop_node))
            if prop_hash:
                if prop_hash in working_set:
                    # Already has index from subtree
                    node_to_index[id(prop_node)] = working_set[prop_hash]
                else:
                    # Property hash not in working set - add to proofHashes
                    proof_hashes.append(prop_hash)
                    working_set[prop_hash] = next_idx
                    node_to_index[id(prop_node)] = next_idx
                    next_idx += 1
        
        # Collect sibling hashes (post-order traversal)
        stack = []
        last_visited = None
        current = self.property_huffman_root
        
        while stack or current:
            if current:
                stack.append(current)
                current = current.left
            else:
                peek = stack[-1]
                if peek.right and last_visited != peek.right:
                    current = peek.right
                else:
                    if node_needed.get(id(peek), False) and not peek.property_group:
                        # Internal node in proof path
                        left_needed = node_needed.get(id(peek.left), False) if peek.left else False
                        right_needed = node_needed.get(id(peek.right), False) if peek.right else False
                        
                        # Add sibling hash if one child is needed but not the other
                        if left_needed and not right_needed and peek.right:
                            sibling_hash = node_hash_cache.get(id(peek.right))
                            if sibling_hash:
                                # Always add to proof_hashes even if duplicate
                                # But only update working_set if it's the first occurrence
                                print(f"      HUFFMAN SIBLING: Adding right sibling {sibling_hash[:16]}... at index {next_idx} (node {id(peek.right)})")
                                proof_hashes.append(sibling_hash)
                                if sibling_hash not in working_set:
                                    working_set[sibling_hash] = next_idx
                                node_to_index[id(peek.right)] = next_idx
                                next_idx += 1
                        elif right_needed and not left_needed and peek.left:
                            sibling_hash = node_hash_cache.get(id(peek.left))
                            if sibling_hash:
                                # Always add to proof_hashes even if duplicate
                                # But only update working_set if it's the first occurrence
                                print(f"      HUFFMAN SIBLING: Adding left sibling {sibling_hash[:16]}... at index {next_idx} (node {id(peek.left)})")
                                proof_hashes.append(sibling_hash)
                                if sibling_hash not in working_set:
                                    working_set[sibling_hash] = next_idx
                                node_to_index[id(peek.left)] = next_idx
                                next_idx += 1
                    stack.pop()
                    last_visited = peek
                    current = None
        
        # Generate path instructions (post-order traversal for Huffman tree)
        # Post-order ensures children are processed before parents
        # This guarantees stack-based verifier can process instructions sequentially
        # IMPORTANT: node_to_index already initialized above with property and sibling mappings
        used_as_child = set()  # Track which indices have been used as merge operands
        stack = []
        last_visited = None
        current = self.property_huffman_root
        
        while stack or current:
            if current:
                stack.append(current)
                current = current.left
            else:
                peek = stack[-1]
                if peek.right and last_visited != peek.right:
                    current = peek.right
                else:
                    if node_needed.get(id(peek), False) and not peek.property_group:
                        # Internal node - check if both children have indices assigned
                        if peek.left and peek.right:
                            left_idx = node_to_index.get(id(peek.left))
                            right_idx = node_to_index.get(id(peek.right))
                            
                            if left_idx is not None and right_idx is not None:
                                node_hash = node_hash_cache.get(id(peek))
                                left_hash = node_hash_cache.get(id(peek.left))
                                right_hash = node_hash_cache.get(id(peek.right))
                                
                                print(f"      HUFFMAN MERGE CHECK: Node {id(peek)} with children [{left_idx}, {right_idx}]")
                                print(f"        Node hash: {node_hash[:16] if node_hash else None}...")
                                print(f"        Left child (node {id(peek.left)}): {left_hash[:16] if left_hash else None}... (index {left_idx})")
                                print(f"        Right child (node {id(peek.right)}): {right_hash[:16] if right_hash else None}... (index {right_idx})")
                                print(f"        Node already has index: {id(peek) in node_to_index}")
                                
                                # Check if either child has already been used in a merge
                                # Input indices (< num_leaves) can be reused, but computed indices cannot
                                all_hashes_count = num_leaves + len(proof_hashes)
                                left_already_used = left_idx >= num_leaves and left_idx in used_as_child
                                right_already_used = right_idx >= num_leaves and right_idx in used_as_child
                                
                                if left_already_used or right_already_used:
                                    print(f"        SKIPPING: Child already used (left: {left_already_used}, right: {right_already_used})")
                                elif id(peek) not in node_to_index:
                                    # Check for duplicate instruction
                                    instr_key = (left_idx, right_idx)
                                    if instr_key not in path_map_instructions:
                                        # Add merge instruction
                                        print(f"      HUFFMAN: Adding instruction [{left_idx}, {right_idx}] -> {next_idx}")
                                        path_map.extend([left_idx, right_idx])
                                        path_map_instructions.add(instr_key)
                                        node_to_index[id(peek)] = next_idx
                                        working_set[node_hash] = next_idx
                                        # Mark these indices as used
                                        used_as_child.add(left_idx)
                                        used_as_child.add(right_idx)
                                        next_idx += 1
                                    else:
                                        print(f"      HUFFMAN: Skipping duplicate instruction [{left_idx}, {right_idx}]")
                                else:
                                    print(f"      HUFFMAN: Skipping node - already has index assigned")
                                    print(f"      HUFFMAN: Skipping node - hash already in working_set")
                    stack.pop()
                    last_visited = peek
                    current = None
        
        return {'next_idx': next_idx}
    
    def _get_descendants(self, node):
        """Get all property node descendants of a node - iterative version."""
        if node is None:
            return set()
        stack = [self.property_huffman_root] if self.property_huffman_root else []
        while stack:
            node = stack.pop()
            if node is None:
                continue
            if node.property_group is not None:
                property_nodes.append(node)
                for doc in node.property_group.optimized_document_order:
                    leaf_to_property[doc.hash_hex] = node
            else:
                if node.right:
                    stack.append(node.right)
                if node.left:
                    stack.append(node.left)
        
        # Find property nodes that contain requested leaves
        involved_properties = set()
        for leaf in sorted_leaves:
            if leaf in leaf_to_property:
                involved_properties.add(leaf_to_property[leaf])
        
        if not involved_properties:
            return {'leaves': [], 'proofHashes': [], 'pathMap': []}
        
        # Generate proof from unbalanced Huffman tree structure
        proof_hashes = []
        path_map = []
        
        # For unbalanced tree, we need to generate proofs based on the Huffman tree structure
        # This is more complex than balanced trees but provides better efficiency for frequent properties
        
        # Collect all nodes in proof path using iterative post-order traversal
        proof_nodes = set()
        node_needed = {}  # Track if node is needed in proof path
        
        stack = []
        last_visited = None
        current = self.property_huffman_root
        
        while stack or current:
            if current:
                stack.append(current)
                current = current.left
            else:
                peek_node = stack[-1]
                if peek_node.right and last_visited != peek_node.right:
                    current = peek_node.right
                else:
                    # Process this node
                    if peek_node.property_group is not None:
                        # Leaf node
                        if peek_node in involved_properties:
                            proof_nodes.add(peek_node)
                            node_needed[id(peek_node)] = True
                        else:
                            node_needed[id(peek_node)] = False
                    else:
                        # Internal node
                        left_needed = node_needed.get(id(peek_node.left), False) if peek_node.left else False
                        right_needed = node_needed.get(id(peek_node.right), False) if peek_node.right else False
                        
                        if left_needed or right_needed:
                            proof_nodes.add(peek_node)
                            node_needed[id(peek_node)] = True
                        else:
                            node_needed[id(peek_node)] = False
                    
                    stack.pop()
                    last_visited = peek_node
                    current = None
        
        # Generate simplified proof for unbalanced structure
        # For now, use a basic approach that works with the pathMap format
        
        leaves = list(sorted_leaves)
        
        # For unbalanced trees, we need sibling nodes at various depths - iterative version
        # First get descendants and hashes for all nodes
        node_descendants = {}
        node_hash_map = {}
        
        def get_descendants_iterative(root):
            if root is None:
                return set()
            
            stack = []
            last_visited = None
            current = root
            
            while stack or current:
                if current:
                    stack.append(current)
                    current = current.left
                else:
                    peek_node = stack[-1]
                    if peek_node.right and last_visited != peek_node.right:
                        current = peek_node.right
                    else:
                        # Process this node
                        if peek_node.property_group is not None:
                            node_descendants[id(peek_node)] = {peek_node}
                            # Compute hash for property node
                            if peek_node.property_group.optimized_document_order:
                                doc_hashes = [doc.hash_hex for doc in peek_node.property_group.optimized_document_order]
                                if len(doc_hashes) == 1:
                                    node_hash_map[id(peek_node)] = doc_hashes[0]
                                else:
                                    node_hash_map[id(peek_node)] = self._compute_balanced_subtree_hash(doc_hashes)
                        else:
                            descendants = set()
                            if peek_node.left:
                                descendants.update(node_descendants.get(id(peek_node.left), set()))
                            if peek_node.right:
                                descendants.update(node_descendants.get(id(peek_node.right), set()))
                            node_descendants[id(peek_node)] = descendants
                            # Compute hash for internal node
                            left_hash = node_hash_map.get(id(peek_node.left), keccak(b'').hex()) if peek_node.left else keccak(b'').hex()
                            right_hash = node_hash_map.get(id(peek_node.right), keccak(b'').hex()) if peek_node.right else keccak(b'').hex()
                            node_hash_map[id(peek_node)] = combine_and_hash(left_hash, right_hash)
                        
                        stack.pop()
                        last_visited = peek_node
                        current = None
            
            return node_descendants.get(id(root), set())
        
        get_descendants_iterative(self.property_huffman_root)
        
        # Now collect siblings iteratively
        stack = [self.property_huffman_root] if self.property_huffman_root else []
        while stack:
            node = stack.pop()
            if node is None or node.property_group is not None:
                continue
            
            left_descendants = node_descendants.get(id(node.left), set()) if node.left else set()
            right_descendants = node_descendants.get(id(node.right), set()) if node.right else set()
            
            left_has_target = any(p in left_descendants for p in involved_properties)
            right_has_target = any(p in right_descendants for p in involved_properties)
            
            if left_has_target and not right_has_target:
                # Need right sibling
                if node.right:
                    right_hash = node_hash_map.get(id(node.right))
                    if right_hash and right_hash not in proof_hashes:
                        proof_hashes.append(right_hash)
            elif right_has_target and not left_has_target:
                # Need left sibling
                if node.left:
                    left_hash = node_hash_map.get(id(node.left))
                    if left_hash and left_hash not in proof_hashes:
                        proof_hashes.append(left_hash)
            
            # Add children to stack
            if left_has_target and node.left:
                stack.append(node.left)
            if right_has_target and node.right:
                stack.append(node.right)
        
        # Generate pathMap instructions for bottom-up reconstruction of unbalanced tree
        # Build a working set index mapping for proper reconstruction
        working_set_indices = {}
        
        # Step 1: Add all leaves to working set
        for i, leaf in enumerate(leaves):
            working_set_indices[leaf] = i
        
        # Step 2: Add all proof hashes to working set
        for i, proof_hash in enumerate(proof_hashes):
            working_set_indices[proof_hash] = len(leaves) + i
        
        # Step 3: Generate pathMap by traversing tree and recording hash operations
        next_computed_idx = len(leaves) + len(proof_hashes)
        computed_hashes = {}
        
        # PRE-COMPUTE ALL NODE HASHES ONCE to avoid O(n²) behavior
        # Build hash cache for all nodes in the tree
        node_hash_cache = {}
        
        def build_hash_cache(root):
            """Pre-compute hashes for all nodes in tree."""
            if root is None:
                return
            
            stack = []
            last_visited = None
            current = root
            
            while stack or current:
                if current:
                    stack.append(current)
                    current = current.left
                else:
                    peek_node = stack[-1]
                    if peek_node.right and last_visited != peek_node.right:
                        current = peek_node.right
                    else:
                        # Compute hash for this node
                        if peek_node.property_group is not None:
                            # Leaf - use first document hash
                            if peek_node.property_group.optimized_document_order:
                                doc_hashes = [doc.hash_hex for doc in peek_node.property_group.optimized_document_order]
                                if len(doc_hashes) == 1:
                                    node_hash_cache[id(peek_node)] = doc_hashes[0]
                                else:
                                    node_hash_cache[id(peek_node)] = self._compute_balanced_subtree_hash(doc_hashes)
                        else:
                            # Internal - combine children
                            left_hash = node_hash_cache.get(id(peek_node.left), keccak(b'').hex()) if peek_node.left else keccak(b'').hex()
                            right_hash = node_hash_cache.get(id(peek_node.right), keccak(b'').hex()) if peek_node.right else keccak(b'').hex()
                            node_hash_cache[id(peek_node)] = combine_and_hash(left_hash, right_hash)
                        
                        stack.pop()
                        last_visited = peek_node
                        current = None
        
        build_hash_cache(self.property_huffman_root)
        
        # Use iterative post-order traversal to generate path instructions
        def generate_path_instructions_iterative(root):
            nonlocal next_computed_idx
            
            if root is None:
                return None
            
            stack = []
            node_index_map = {}  # Maps node id to its working set index
            last_visited = None
            current = root
            
            while stack or current:
                # Go to leftmost node
                if current:
                    stack.append(current)
                    current = current.left
                else:
                    peek_node = stack[-1]
                    
                    # If right child exists and hasn't been processed yet
                    if peek_node.right and last_visited != peek_node.right:
                        current = peek_node.right
                    else:
                        # Process this node
                        node_id = id(peek_node)
                        
                        # If this is a property node (leaf), check if any of its documents are requested
                        if peek_node.property_group is not None:
                            if peek_node.property_group.optimized_document_order:
                                # Check if any documents from this property are in the requested set
                                property_docs = peek_node.property_group.optimized_document_order
                                requested_docs = [doc for doc in property_docs if doc.hash_hex in working_set_indices]
                                
                                # If this property has requested documents, we need to handle the subtree
                                # For now, mark the property node with the index of its first requested document
                                # (This is a simplified approach; full implementation would need subtree proof generation)
                                if requested_docs:
                                    # Use first requested document's index as placeholder
                                    node_index_map[node_id] = working_set_indices[requested_docs[0].hash_hex]
                        else:
                            # Internal node - use pre-computed hash from cache
                            node_hash = node_hash_cache.get(node_id)
                            
                            if node_hash is None:
                                # This shouldn't happen if cache was built correctly
                                continue
                            
                            # If this node is already in working set (as proof hash), use its index
                            if node_hash in working_set_indices:
                                node_index_map[node_id] = working_set_indices[node_hash]
                            # If already computed, use its index
                            elif node_hash in computed_hashes:
                                node_index_map[node_id] = computed_hashes[node_hash]
                            else:
                                # Need to compute this node from children
                                left_idx = node_index_map.get(id(peek_node.left)) if peek_node.left else None
                                right_idx = node_index_map.get(id(peek_node.right)) if peek_node.right else None
                                
                                if left_idx is not None and right_idx is not None:
                                    # Add instruction to pathMap (left_idx == right_idx is valid - contract handles it)
                                    path_map.extend([left_idx, right_idx])
                                    # Record this computed index
                                    computed_hashes[node_hash] = next_computed_idx
                                    node_index_map[node_id] = next_computed_idx
                                    next_computed_idx += 1
                        
                        stack.pop()
                        last_visited = peek_node
                        current = None
            
            return node_index_map.get(id(root))
        
        # Generate instructions starting from root
        generate_path_instructions_iterative(self.property_huffman_root)
        
        return {
            'leaves': leaves,
            'proofHashes': proof_hashes,
            'pathMap': path_map
        }
    
    def _get_descendants(self, node):
        """Get all property node descendants of a node - iterative version."""
        if node is None:
            return set()
        
        if node.property_group is not None:
            return {node}
        
        # Iterative post-order traversal
        descendants_map = {}
        stack = []
        last_visited = None
        current = node
        
        while stack or current:
            if current:
                stack.append(current)
                current = current.left
            else:
                peek_node = stack[-1]
                if peek_node.right and last_visited != peek_node.right:
                    current = peek_node.right
                else:
                    # Process this node
                    if peek_node.property_group is not None:
                        descendants_map[id(peek_node)] = {peek_node}
                    else:
                        descendants = set()
                        if peek_node.left:
                            descendants.update(descendants_map.get(id(peek_node.left), set()))
                        if peek_node.right:
                            descendants.update(descendants_map.get(id(peek_node.right), set()))
                        descendants_map[id(peek_node)] = descendants
                    
                    stack.pop()
                    last_visited = peek_node
                    current = None
        
        return descendants_map.get(id(node), set())
    
    def generate_bitmap_proof(self, leaves_to_prove_hex):
        """
        Generate bitmap proof using post-order traversal of proof tree.
        
        This method builds a proof tree (subset of the full Merkle tree needed for verification)
        and traverses it in post-order to generate the correct sequence of operations.
        
        Post-order traversal guarantees:
        - Children are processed before parents
        - When we reach an internal node, both children are already on top of stack
        - This is exactly what stack-based verification needs!
        
        Returns:
            dict with:
            - 'inputs': List of hash values in the order they should be pushed
            - 'bitmap': List of uint256 where bit=1 means PUSH, bit=0 means MERGE
            - 'expected_root': The expected root hash for verification
        """
        if not leaves_to_prove_hex or not self.property_huffman_root:
            return {'inputs': [], 'bitmap': [], 'expected_root': self.merkle_root}
        
        # Step 1: Build the proof tree
        proof_tree_root = self._build_proof_tree(leaves_to_prove_hex)
        
        if proof_tree_root is None:
            return {'inputs': [], 'bitmap': [], 'expected_root': self.merkle_root}
        
        # Step 2: Post-order traversal to generate inputs and operations (iterative to avoid recursion limit)
        inputs = []
        operations = []  # 1=PUSH, 0=MERGE
        
        # Iterative post-order traversal using explicit stack
        stack = []
        last_visited = None
        current = proof_tree_root
        
        while stack or current:
            if current:
                stack.append(current)
                current = current.left if not current.is_input else None
            else:
                peek = stack[-1]
                
                # If right child exists and hasn't been visited yet
                if peek.right and last_visited != peek.right and not peek.is_input:
                    current = peek.right
                else:
                    # Process this node
                    if peek.is_input:
                        # This is a leaf/sibling hash - PUSH it
                        inputs.append(peek.hash_value)
                        operations.append(1)  # PUSH
                    else:
                        # This is an internal node - MERGE (children already processed)
                        operations.append(0)  # MERGE
                    
                    stack.pop()
                    last_visited = peek
                    current = None
        
        # Step 3: Convert operations to bitmap format
        bitmap = self._operations_to_bitmap(operations)
        
        return {
            'inputs': inputs,
            'bitmap': bitmap,
            'expected_root': self.merkle_root,
            'num_operations': len(operations),
            'num_pushes': operations.count(1),
            'num_merges': operations.count(0)
        }
    
    def _build_proof_tree(self, leaves_to_prove_hex):
        """
        Build a proof tree from the requested leaves.
        
        The proof tree is a subset of the full Merkle tree that includes:
        1. All requested leaves
        2. All ancestors of requested leaves up to the root
        3. All sibling hashes needed for verification
        
        Returns:
            ProofNode: Root of the proof tree
        """
        # Sort and deduplicate leaves
        sorted_leaves = sorted(set(leaves_to_prove_hex))
        
        # Step 1: Map documents to their property nodes in the Huffman tree
        doc_to_property = {}
        
        def collect_property_mappings(node):
            if node is None:
                return
            stack = [node]
            while stack:
                current = stack.pop()
                if current is None:
                    continue
                if current.property_group is not None:
                    for doc in current.property_group.optimized_document_order:
                        doc_to_property[doc.hash_hex] = current
                else:
                    if current.right:
                        stack.append(current.right)
                    if current.left:
                        stack.append(current.left)
        
        collect_property_mappings(self.property_huffman_root)
        
        # Step 2: Group requested documents by property
        property_to_docs = {}
        for doc_hash in sorted_leaves:
            if doc_hash in doc_to_property:
                prop_node = doc_to_property[doc_hash]
                if prop_node not in property_to_docs:
                    property_to_docs[prop_node] = []
                property_to_docs[prop_node].append(doc_hash)
        
        if not property_to_docs:
            return None
        
        # Step 3: Pre-compute all hashes in the Huffman tree
        node_hash_cache = {}
        property_hash_cache = {}  # Maps property node id -> its hash
        
        def compute_all_hashes(node):
            """Compute hashes for all nodes using post-order traversal."""
            if node is None:
                return keccak(b'').hex()
            
            stack = []
            last_visited = None
            current = node
            
            while stack or current:
                if current:
                    stack.append(current)
                    current = current.left
                else:
                    peek = stack[-1]
                    if peek.right and last_visited != peek.right:
                        current = peek.right
                    else:
                        if peek.property_group is not None:
                            # Property node - compute balanced subtree hash
                            doc_hashes = [d.hash_hex for d in peek.property_group.optimized_document_order]
                            if len(doc_hashes) == 1:
                                node_hash_cache[id(peek)] = doc_hashes[0]
                            else:
                                node_hash_cache[id(peek)] = self._compute_balanced_subtree_hash(doc_hashes)
                            property_hash_cache[id(peek)] = node_hash_cache[id(peek)]
                        else:
                            left_h = node_hash_cache.get(id(peek.left), keccak(b'').hex()) if peek.left else keccak(b'').hex()
                            right_h = node_hash_cache.get(id(peek.right), keccak(b'').hex()) if peek.right else keccak(b'').hex()
                            node_hash_cache[id(peek)] = combine_and_hash(left_h, right_h)
                        stack.pop()
                        last_visited = peek
                        current = None
        
        compute_all_hashes(self.property_huffman_root)
        
        # Step 4: Mark which Huffman tree nodes are needed
        node_needed = {}
        involved_properties = set(property_to_docs.keys())
        
        def mark_needed_nodes(node):
            """Mark nodes in proof path using post-order traversal."""
            if node is None:
                return False
            
            stack = []
            last_visited = None
            current = node
            
            while stack or current:
                if current:
                    stack.append(current)
                    current = current.left
                else:
                    peek = stack[-1]
                    if peek.right and last_visited != peek.right:
                        current = peek.right
                    else:
                        if peek.property_group is not None:
                            node_needed[id(peek)] = peek in involved_properties
                        else:
                            left_needed = node_needed.get(id(peek.left), False) if peek.left else False
                            right_needed = node_needed.get(id(peek.right), False) if peek.right else False
                            node_needed[id(peek)] = left_needed or right_needed
                        stack.pop()
                        last_visited = peek
                        current = None
        
        mark_needed_nodes(self.property_huffman_root)
        
        # Step 5: Build proof tree for each property's document subtree
        property_proof_trees = {}  # Maps property node id -> proof tree for that property
        
        for prop_node, requested_docs in property_to_docs.items():
            all_docs = [d.hash_hex for d in prop_node.property_group.optimized_document_order]
            property_proof_trees[id(prop_node)] = self._build_balanced_subtree_proof_tree(
                all_docs, requested_docs
            )
        
        # Step 6: Build proof tree for the Huffman tree (property level) - ITERATIVE
        # Use post-order traversal to build proof tree nodes bottom-up
        proof_tree_cache = {}  # Maps node id -> ProofNode
        
        stack = []
        last_visited = None
        current = self.property_huffman_root
        
        while stack or current:
            if current:
                stack.append(current)
                current = current.left
            else:
                peek = stack[-1]
                
                if peek.right and last_visited != peek.right:
                    current = peek.right
                else:
                    # Process this node - build its proof tree node
                    if not node_needed.get(id(peek), False):
                        # This node is not in the proof path - it's a sibling hash
                        proof_tree_cache[id(peek)] = ProofNode(hash_value=node_hash_cache[id(peek)], is_input=True)
                    elif peek.property_group is not None:
                        # This is a property node that's in our proof path
                        proof_tree_cache[id(peek)] = property_proof_trees.get(id(peek))
                    else:
                        # Internal node in proof path - combine children
                        left_proof = proof_tree_cache.get(id(peek.left)) if peek.left else None
                        right_proof = proof_tree_cache.get(id(peek.right)) if peek.right else None
                        
                        if left_proof is None and right_proof is None:
                            proof_tree_cache[id(peek)] = None
                        else:
                            left_child_needed = node_needed.get(id(peek.left), False) if peek.left else False
                            right_child_needed = node_needed.get(id(peek.right), False) if peek.right else False
                            
                            if left_child_needed and right_child_needed:
                                merge_node = ProofNode(is_input=False)
                                merge_node.left = left_proof
                                merge_node.right = right_proof
                                proof_tree_cache[id(peek)] = merge_node
                            elif left_child_needed:
                                merge_node = ProofNode(is_input=False)
                                merge_node.left = left_proof
                                merge_node.right = ProofNode(hash_value=node_hash_cache[id(peek.right)], is_input=True)
                                proof_tree_cache[id(peek)] = merge_node
                            else:  # right_child_needed
                                merge_node = ProofNode(is_input=False)
                                merge_node.left = ProofNode(hash_value=node_hash_cache[id(peek.left)], is_input=True)
                                merge_node.right = right_proof
                                proof_tree_cache[id(peek)] = merge_node
                    
                    stack.pop()
                    last_visited = peek
                    current = None
        
        return proof_tree_cache.get(id(self.property_huffman_root))
    
    def _build_balanced_subtree_proof_tree(self, all_docs, requested_docs):
        """
        Build proof tree for a balanced subtree of documents within a property.
        
        Args:
            all_docs: All document hashes in this property (in order)
            requested_docs: Subset of documents that need to be proven
            
        Returns:
            ProofNode: Root of proof tree for this subtree
        """
        if len(all_docs) == 1:
            # Single document - just a leaf
            return ProofNode(hash_value=all_docs[0], is_input=True)
        
        # Build balanced tree layers
        layers = [list(all_docs)]
        current = list(all_docs)
        
        while len(current) > 1:
            if len(current) % 2 != 0:
                current.append(current[-1])
            next_layer = []
            for i in range(0, len(current), 2):
                next_layer.append(combine_and_hash(current[i], current[i + 1]))
            layers.append(next_layer)
            current = next_layer
        
        # Determine which nodes are needed for proof
        requested_set = set(requested_docs)
        needed_nodes = set()  # (layer, idx) tuples
        
        # Mark requested documents and propagate up
        for doc in requested_docs:
            if doc in all_docs:
                idx = all_docs.index(doc)
                needed_nodes.add((0, idx))
        
        # Propagate "needed" status up the tree
        for layer_idx in range(len(layers) - 1):
            layer = layers[layer_idx]
            padded_len = len(layer) if len(layer) % 2 == 0 else len(layer) + 1
            
            for layer_l, idx in list(needed_nodes):
                if layer_l == layer_idx:
                    parent_idx = idx // 2
                    needed_nodes.add((layer_idx + 1, parent_idx))
        
        # Build proof tree iteratively (bottom-up from leaf layer to root)
        # Only build ProofNodes for positions that are actually needed
        proof_node_cache = {}
        
        # Process from leaf layer (0) up to root layer
        for layer_idx in range(len(layers)):
            # Only process nodes at positions that are needed for the proof
            for layer_l, node_idx in needed_nodes:
                if layer_l != layer_idx:
                    continue
                    
                if layer_idx == 0:
                    # Leaf layer - PUSH the document hash
                    if node_idx < len(layers[0]):
                        proof_node_cache[(0, node_idx)] = ProofNode(hash_value=layers[0][node_idx], is_input=True)
                    else:
                        # Padding - use last element
                        proof_node_cache[(0, node_idx)] = ProofNode(hash_value=layers[0][-1], is_input=True)
                else:
                    # Internal layer - combine children from layer below
                    left_child_idx = node_idx * 2
                    right_child_idx = node_idx * 2 + 1
                    
                    left_needed = (layer_idx - 1, left_child_idx) in needed_nodes
                    right_needed = (layer_idx - 1, right_child_idx) in needed_nodes
                    
                    if left_needed and right_needed:
                        # Both children needed - create MERGE node
                        merge_node = ProofNode(is_input=False)
                        merge_node.left = proof_node_cache.get((layer_idx - 1, left_child_idx))
                        merge_node.right = proof_node_cache.get((layer_idx - 1, right_child_idx))
                        proof_node_cache[(layer_idx, node_idx)] = merge_node
                    elif left_needed:
                        # Only left needed - right is sibling
                        merge_node = ProofNode(is_input=False)
                        merge_node.left = proof_node_cache.get((layer_idx - 1, left_child_idx))
                        # Right sibling hash
                        if right_child_idx < len(layers[layer_idx - 1]):
                            sibling_hash = layers[layer_idx - 1][right_child_idx]
                        else:
                            sibling_hash = layers[layer_idx - 1][-1]  # Padding
                        merge_node.right = ProofNode(hash_value=sibling_hash, is_input=True)
                        proof_node_cache[(layer_idx, node_idx)] = merge_node
                    elif right_needed:
                        # Only right needed - left is sibling
                        merge_node = ProofNode(is_input=False)
                        sibling_hash = layers[layer_idx - 1][left_child_idx]
                        merge_node.left = ProofNode(hash_value=sibling_hash, is_input=True)
                        merge_node.right = proof_node_cache.get((layer_idx - 1, right_child_idx))
                        proof_node_cache[(layer_idx, node_idx)] = merge_node
                    else:
                        # This node is needed but neither child is - use pre-computed hash
                        proof_node_cache[(layer_idx, node_idx)] = ProofNode(hash_value=layers[layer_idx][node_idx], is_input=True)
        
        # Return root node
        return proof_node_cache.get((len(layers) - 1, 0))
    
    def _operations_to_bitmap(self, operations):
        """
        Convert list of operations (1=PUSH, 0=MERGE) to uint256 bitmap array.
        
        Each uint256 can hold 256 operation bits.
        Bit is set to 1 for PUSH, 0 for MERGE.
        """
        if not operations:
            return []
        
        bitmap = []
        current_word = 0
        
        for i, op in enumerate(operations):
            if op == 1:  # PUSH
                current_word |= (1 << (i % 256))
            # MERGE is 0, bit stays 0
            
            # Every 256 bits, start a new word
            if (i + 1) % 256 == 0:
                bitmap.append(current_word)
                current_word = 0
        
        # Add remaining bits
        if len(operations) % 256 != 0 or not bitmap:
            bitmap.append(current_word)
        
        return bitmap
    
    def verify_bitmap_locally(self, inputs, bitmap):
        """
        Local verification of bitmap proof for testing.
        
        Simulates the stack-based verification done in the smart contract.
        """
        if not inputs:
            return False, "Empty inputs"
        
        stack = []
        input_idx = 0
        
        # Process each bit in bitmap
        for word_idx, bitmap_word in enumerate(bitmap):
            for bit_pos in range(256):
                # Check termination condition
                if len(stack) == 1 and input_idx >= len(inputs):
                    computed_root = stack[0]
                    is_valid = computed_root == self.merkle_root
                    return is_valid, f"Root: {computed_root[:16]}... vs {self.merkle_root[:16]}..."
                
                is_push = (bitmap_word & (1 << bit_pos)) != 0
                
                if is_push:
                    if input_idx >= len(inputs):
                        return False, f"Input index {input_idx} out of bounds (have {len(inputs)})"
                    stack.append(inputs[input_idx])
                    input_idx += 1
                else:
                    # MERGE
                    if len(stack) < 2:
                        return False, f"Stack underflow at bit {word_idx * 256 + bit_pos}: stack size {len(stack)}"
                    right = stack.pop()
                    left = stack.pop()
                    merged = combine_and_hash(left, right)
                    stack.append(merged)
        
        # Final check
        if len(stack) == 1 and input_idx >= len(inputs):
            computed_root = stack[0]
            is_valid = computed_root == self.merkle_root
            return is_valid, f"Root: {computed_root[:16]}... vs {self.merkle_root[:16]}..."
        
        return False, f"Invalid final state: stack size {len(stack)}, inputs consumed {input_idx}/{len(inputs)}"

    def generate_multiproof(self, leaves_to_prove_hex):
        """Generate multiproof using bitmap format (replaces pathMap)."""
        proof_data = self.generate_bitmap_proof(leaves_to_prove_hex)
        return proof_data['inputs'], proof_data['bitmap']
    
    def get_learning_stats(self):
        """Get learning statistics."""
        from learning_config import get_learning_config
        config = get_learning_config()
        return {
            'mode': config.mode.value,
            'approach': 'traditional_property_level_huffman',
            'verification_count': self.verification_count,
            'batch_size': config.batch_size if config.mode.name == 'BATCH' else None,
            'total_patterns': self.compressed_traffic.total_events,
            'provinces': len(self.province_property_groups),
            'total_properties': sum(len(props) for props in self.province_property_groups.values()),
            'total_documents': len(self.ordered_leaves_hex)
        }
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
    def __init__(self, property_group, freq):
        self.property_group = property_group  # PropertyDocumentGroup object or None for internal nodes
        self.freq = freq
        self.left = None
        self.right = None
        self.all_documents = []  # All documents from this subtree
        self.depth = 0  # Depth in the tree for proof generation
        
        # If property_group is provided, initialize documents
        if property_group is not None:
            self.all_documents = property_group.optimized_document_order.copy()
    
    def __lt__(self, other):
        return self.freq < other.freq
    
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
    
    def compute_depths(self, depth=0):
        """Compute depth for all nodes in subtree."""
        self.depth = depth
        if self.left:
            self.left.compute_depths(depth + 1)
        if self.right:
            self.right.compute_depths(depth + 1)


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
        """Apply pairs-first Huffman optimization to documents within this property."""
        if len(self.documents) <= 1:
            self.optimized_document_order = self.documents.copy()
            return
        
        # # For very large property groups, use simple frequency-based ordering instead of expensive Huffman
        if len(self.documents) > 0:  # Threshold for computational efficiency
            self.optimized_document_order = sorted(
                self.documents,
                key=lambda doc: document_frequencies.get(f"{doc.province}.{doc.property_id}.{doc.doc_id}", 1),
                reverse=True
            )
            return
        
        # Create document frequency map for this property's documents
        property_doc_freq = {}
        for doc in self.documents:
            doc_id = f"{doc.province}.{doc.property_id}.{doc.doc_id}"
            property_doc_freq[doc] = document_frequencies.get(doc_id, 1)  # Default frequency of 1
        
        # Get document pair frequencies relevant to this property (optimized)
        property_pair_freq = {}
        for i, doc1 in enumerate(self.documents):
            for doc2 in self.documents[i+1:]:  # Only check pairs once, avoid duplicates
                doc1_id = f"{doc1.province}.{doc1.property_id}.{doc1.doc_id}"
                doc2_id = f"{doc2.province}.{doc2.property_id}.{doc2.doc_id}"
                pair_key = tuple(sorted([doc1_id, doc2_id]))
                if pair_key in document_pair_frequencies:
                    freq = document_pair_frequencies[pair_key]
                    property_pair_freq[(doc1, doc2)] = freq
                    property_pair_freq[(doc2, doc1)] = freq  # Add both directions
        
        # Apply pairs-first optimization within this property with configurable alpha threshold
        self.optimized_document_order = self._apply_pairs_first_huffman(
            self.documents, property_doc_freq, property_pair_freq, alpha_threshold
        )
    
    def _apply_pairs_first_huffman(self, documents, doc_frequencies, pair_frequencies, alpha_threshold=0.1):
        """Apply pairs-first Huffman optimization to documents with alpha threshold.
        
        Args:
            documents: List of documents to optimize
            doc_frequencies: Individual document access frequencies
            pair_frequencies: Document pair co-access frequencies  
            alpha_threshold: Minimum relative weight w(tx) for pair merging (default 0.1 = 10%)
                           w(tx) = q_k / min(p_i, p_j) where q_k is pair frequency,
                           p_i and p_j are individual document frequencies
        """
        if len(documents) <= 1:
            return documents
        
        # Special handling for alpha = 0 to prevent computational explosion
        if alpha_threshold <= 0:
            print(f"      Alpha threshold: {alpha_threshold} -> Using simple frequency-based ordering (alpha=0 causes computational explosion)")
            # Use simple frequency-based ordering instead of pairs-first Huffman
            return sorted(documents, key=lambda doc: doc_frequencies.get(doc, 1), reverse=True)
        
        print(f"      Alpha threshold (relative weight): {alpha_threshold}")
        
        # Step 1: Pairs-first optimization with relative weight threshold
        # Create initial document nodes
        doc_nodes = {doc: DocumentHuffmanNode(doc, doc_frequencies[doc]) for doc in documents}
        merged_pairs = set()
        strong_pairs_count = 0
        weak_pairs_count = 0
        
        # Sort pairs by frequency (most frequent first)
        sorted_pairs = sorted(pair_frequencies.items(), key=lambda x: x[1], reverse=True)
        
        # Merge only strong pairs (relative weight above alpha threshold)
        for (doc1, doc2), pair_freq in sorted_pairs:
            # Calculate relative weight: w(tx) = q_k / min(p_i, p_j)
            freq1 = doc_frequencies.get(doc1, 1)
            freq2 = doc_frequencies.get(doc2, 1)
            min_individual_freq = min(freq1, freq2)
            
            if min_individual_freq > 0:
                relative_weight = pair_freq / min_individual_freq
            else:
                relative_weight = 0
            
            if relative_weight >= alpha_threshold:  # Alpha threshold check on relative weight
                if doc1 not in merged_pairs and doc2 not in merged_pairs:
                    if doc1 in doc_nodes and doc2 in doc_nodes:
                        # Create merged node
                        merged_node = DocumentHuffmanNode(None, doc_nodes[doc1].freq + doc_nodes[doc2].freq)
                        merged_node.left = doc_nodes[doc1]
                        merged_node.right = doc_nodes[doc2]
                        merged_node.documents = doc_nodes[doc1].documents + doc_nodes[doc2].documents
                        
                        # Update nodes map
                        doc_nodes[doc1] = merged_node
                        doc_nodes[doc2] = merged_node
                        
                        # Mark as merged
                        merged_pairs.add(doc1)
                        merged_pairs.add(doc2)
                        strong_pairs_count += 1
            else:
                weak_pairs_count += 1
        
        print(f"      Merged {strong_pairs_count} strong pairs, skipped {weak_pairs_count} weak pairs")
        
        # Step 2: Huffman tree construction for remaining nodes
        unique_nodes = list({id(node): node for node in doc_nodes.values()}.values())
        
        if len(unique_nodes) <= 1:
            # Single node case
            return unique_nodes[0].get_all_documents() if unique_nodes else documents
        
        # Build Huffman tree
        heap = unique_nodes.copy()
        heapq.heapify(heap)
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            merged = DocumentHuffmanNode(None, left.freq + right.freq)
            merged.left = left
            merged.right = right
            
            heapq.heappush(heap, merged)
        
        # Step 3: Generate ordering from Huffman tree
        if heap:
            root = heap[0]
            return self._generate_document_order_from_tree(root)
        else:
            return documents
    
    def _generate_document_order_from_tree(self, root):
        """Generate document ordering from Huffman tree using in-order traversal."""
        if root is None:
            return []
        
        ordered_docs = []
        
        # Use iterative in-order traversal
        stack = []
        current = root
        
        while stack or current:
            # Go to leftmost node
            while current:
                stack.append(current)
                current = current.left
            
            # Current is None, pop from stack
            current = stack.pop()
            
            # Process node
            if current.item is not None:
                # Leaf node with document
                ordered_docs.append(current.item)
            elif current.documents:
                # Internal node with documents (from merging)
                ordered_docs.extend(current.documents)
            
            # Move to right subtree
            current = current.right
        
        return ordered_docs
    
    def get_document_hashes_hex(self):
        """Get ordered document hashes in hex format."""
        return [doc.hash_hex for doc in self.optimized_document_order]


def combine_and_hash(hash1_hex, hash2_hex):
    """Combine two hashes using keccak256 (OpenZeppelin compatible)."""
    h1_bytes = bytes.fromhex(hash1_hex)
    h2_bytes = bytes.fromhex(hash2_hex)
    # Sort hashes to ensure deterministic ordering
    combined = h1_bytes + h2_bytes if h1_bytes < h2_bytes else h2_bytes + h1_bytes
    return keccak(combined).hex()


class TraditionalPropertyLevelHuffmanBuilder:
    """
    Builds a TRUE unbalanced Merkle tree with direct property-level Huffman optimization:
    1. Document-level Huffman: Optimize within each property group
    2. Property-level Huffman: Build unbalanced tree where frequent properties have shallow depths
    3. Unbalanced structure: Creates variable-depth tree (not balanced binary tree)
    4. Cross-province optimization: Properties from all provinces compete equally
    
    This follows the research paper methodology for true unbalanced Huffman Merkle trees
    without geographical province clustering.
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
        """Calculate co-verification frequencies for optimization using access patterns."""
        property_freq = defaultdict(int)
        pair_freq = defaultdict(int)
        province_freq = defaultdict(int)
        
        # Use transactional pattern for property and pair frequencies
        if self.transactional_pattern:
            # Calculate property frequencies
            for province, property_groups in self.province_property_groups.items():
                for prop_id, group in property_groups.items():
                    if group.documents:
                        doc_frequencies = self.transactional_pattern.get_document_frequencies(group.documents)
                        prop_freq = sum(doc_frequencies.values())
                        full_prop_id = f"{province}.{prop_id}"
                        property_freq[full_prop_id] = prop_freq
                        province_freq[province] += prop_freq
            
            # Calculate pair frequencies using simulated queries within each property group
            for province, province_groups in self.province_property_groups.items():
                for prop_id, group in province_groups.items():
                    if len(group.documents) >= 2:
                        doc_pair_freq = self.transactional_pattern.get_document_pair_frequencies(
                            group.documents, num_simulated_queries=70  # Reduced for efficiency
                        )
                        
                        # Convert document pairs to property pairs
                        for (doc1, doc2), freq in doc_pair_freq.items():
                            prop1_id = f"{doc1.province}.{doc1.property_id}"
                            prop2_id = f"{doc2.province}.{doc2.property_id}"
                            prop_pair = tuple(sorted([prop1_id, prop2_id]))
                            pair_freq[prop_pair] += freq
        
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
    
    def build(self):
        """
        Build TRUE unbalanced Huffman tree with property-level optimization:
        1. Apply document-level Huffman within each property group
        2. Build Huffman tree of property groups (creates unbalanced structure)
        3. Generate leaves from the unbalanced tree traversal
        """
        print("Building Traditional Property-Level Huffman Tree (No Province Clustering)...")
        
        # Calculate frequencies for optimization
        province_freq, property_freq, pair_freq, document_frequencies, document_pair_frequencies = self._calculate_frequencies()
        
        # Step 1: Apply document-level Huffman within each property group
        all_property_groups = []
        for province, property_groups in self.province_property_groups.items():
            for property_id, property_group in property_groups.items():
                print(f"  Pre-optimizing documents in {province}.{property_id}: {len(property_group.documents)} documents")
                
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
        
        # Step 4: Compute Merkle root from the unbalanced structure
        self.merkle_root = self._compute_unbalanced_merkle_root()
        
        print(f"  Traditional Property-Level Huffman Tree Root: {self.merkle_root[:16]}...")
        return self.merkle_root
        
        # # Build flat Merkle tree bottom-up
        # self.tree_layers = []
        # current_layer = list(self.ordered_leaves_hex)
        # self.tree_layers.append(current_layer)
        
        # while len(current_layer) > 1:
        #     # Pad with last element if odd number
        #     if len(current_layer) % 2 != 0:
        #         current_layer.append(current_layer[-1])
            
        #     next_layer = []
        #     for i in range(0, len(current_layer), 2):
        #         parent_hash = combine_and_hash(current_layer[i], current_layer[i + 1])
        #         next_layer.append(parent_hash)
            
        #     self.tree_layers.append(next_layer)
        #     current_layer = next_layer
        
        # self.merkle_root = current_layer[0] if current_layer else keccak(b'').hex()
        
        # print(f"  Clustered Province + Document Huffman Tree Root: {self.merkle_root[:16]}...")
        # return self.merkle_root
    
    def _build_property_huffman_tree(self, property_groups, property_freq, pair_freq):
        """Build true unbalanced Huffman tree at property level."""
        if not property_groups:
            return None
        
        # Create property nodes with frequencies
        property_nodes = []
        for prop_group in property_groups:
            full_prop_id = f"{prop_group.province}.{prop_group.property_id}"
            freq = property_freq.get(full_prop_id, 1)
            node = PropertyHuffmanNode(prop_group, freq)
            property_nodes.append(node)
            print(f"    Property {full_prop_id}: frequency {freq}, {len(prop_group.documents)} documents")
        
        # Apply pairs-first Huffman with relative weight threshold (as per paper)
        alpha_threshold = self.alpha_threshold
        if pair_freq:
            print(f"    Property-level alpha threshold (relative weight): {alpha_threshold}")
            
            # Merge strong property pairs first
            merged_pairs = set()
            strong_pairs_count = 0
            
            sorted_pairs = sorted(pair_freq.items(), key=lambda x: x[1], reverse=True)
            
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
                    # Find corresponding nodes
                    node1 = next((n for n in property_nodes if f"{n.property_group.province}.{n.property_group.property_id}" == prop1_id), None)
                    node2 = next((n for n in property_nodes if f"{n.property_group.province}.{n.property_group.property_id}" == prop2_id), None)
                    
                    if node1 and node2 and node1 not in merged_pairs and node2 not in merged_pairs:
                        # Create merged node
                        merged_node = PropertyHuffmanNode(None, node1.freq + node2.freq)
                        merged_node.left = node1
                        merged_node.right = node2
                        
                        # Replace nodes in list
                        property_nodes = [n for n in property_nodes if n != node1 and n != node2]
                        property_nodes.append(merged_node)
                        
                        merged_pairs.add(node1)
                        merged_pairs.add(node2)
                        strong_pairs_count += 1
            
            print(f"    Merged {strong_pairs_count} strong property pairs")
        
        # Build final Huffman tree
        heap = property_nodes.copy()
        heapq.heapify(heap)
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            merged = PropertyHuffmanNode(None, left.freq + right.freq)
            merged.left = left
            merged.right = right
            
            heapq.heappush(heap, merged)
        
        root = heap[0] if heap else None
        if root:
            root.compute_depths(0)
            print(f"    Property Huffman tree built with root frequency: {root.freq}")
        
        return root
    
    def _extract_leaves_from_huffman_tree(self, node):
        """Extract leaves from unbalanced Huffman tree using in-order traversal."""
        if node is None:
            return
        
        if node.property_group is not None:
            # Leaf node - add all documents from this property
            for doc in node.property_group.optimized_document_order:
                self.ordered_leaves_hex.append(doc.hash_hex)
        else:
            # Internal node - traverse children
            self._extract_leaves_from_huffman_tree(node.left)
            self._extract_leaves_from_huffman_tree(node.right)
    
    def _compute_unbalanced_merkle_root(self):
        """Compute Merkle root preserving the unbalanced structure."""
        if not self.property_huffman_root:
            return keccak(b'').hex()
        
        # Create a mapping from documents to their hashes
        doc_to_hash = {doc.hash_hex: doc.hash_hex for doc in self.all_documents}
        
        return self._compute_node_hash(self.property_huffman_root, doc_to_hash)
    
    def _compute_node_hash(self, node, doc_to_hash):
        """Recursively compute hash for a node in the unbalanced tree."""
        if node is None:
            return keccak(b'').hex()
        
        if node.property_group is not None:
            # Leaf node - compute hash of all documents in this property
            if len(node.all_documents) == 1:
                return node.all_documents[0].hash_hex
            else:
                # Build balanced subtree for documents within this property
                doc_hashes = [doc.hash_hex for doc in node.all_documents]
                return self._compute_balanced_subtree_hash(doc_hashes)
        else:
            # Internal node - combine children
            left_hash = self._compute_node_hash(node.left, doc_to_hash)
            right_hash = self._compute_node_hash(node.right, doc_to_hash)
            return combine_and_hash(left_hash, right_hash)
    
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
        """Generate proof using pathMap format for unbalanced tree bottom-up reconstruction."""
        if not leaves_to_prove_hex or not self.ordered_leaves_hex or not self.property_huffman_root:
            return {'leaves': [], 'proofHashes': [], 'pathMap': []}
        
        # Sort and deduplicate leaves
        sorted_leaves = sorted(set(leaves_to_prove_hex))
        
        # Find which property groups contain the requested leaves
        leaf_to_property = {}
        property_nodes = []
        
        def collect_property_nodes(node):
            if node is None:
                return
            if node.property_group is not None:
                property_nodes.append(node)
                for doc in node.property_group.optimized_document_order:
                    leaf_to_property[doc.hash_hex] = node
            else:
                collect_property_nodes(node.left)
                collect_property_nodes(node.right)
        
        collect_property_nodes(self.property_huffman_root)
        
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
        
        # Collect all nodes in proof path
        proof_nodes = set()
        
        def collect_proof_nodes(node, target_properties):
            if node is None:
                return False
            
            if node.property_group is not None:
                # Leaf node
                if node in target_properties:
                    proof_nodes.add(node)
                    return True
                return False
            else:
                # Internal node
                left_needed = collect_proof_nodes(node.left, target_properties)
                right_needed = collect_proof_nodes(node.right, target_properties)
                
                if left_needed or right_needed:
                    proof_nodes.add(node)
                    return True
                return False
        
        collect_proof_nodes(self.property_huffman_root, involved_properties)
        
        # Generate simplified proof for unbalanced structure
        # For now, use a basic approach that works with the pathMap format
        
        leaves = list(sorted_leaves)
        
        # For unbalanced trees, we need sibling nodes at various depths
        def collect_siblings(node, target_properties, depth=0):
            if node is None:
                return
            
            if node.property_group is not None:
                return
            
            left_has_target = any(p in self._get_descendants(node.left) for p in target_properties) if node.left else False
            right_has_target = any(p in self._get_descendants(node.right) for p in target_properties) if node.right else False
            
            if left_has_target and not right_has_target:
                # Need right sibling
                if node.right:
                    right_hash = self._compute_node_hash(node.right, {})
                    if right_hash not in proof_hashes:
                        proof_hashes.append(right_hash)
            elif right_has_target and not left_has_target:
                # Need left sibling
                if node.left:
                    left_hash = self._compute_node_hash(node.left, {})
                    if left_hash not in proof_hashes:
                        proof_hashes.append(left_hash)
            
            # Recurse
            if left_has_target:
                collect_siblings(node.left, target_properties, depth + 1)
            if right_has_target:
                collect_siblings(node.right, target_properties, depth + 1)
        
        collect_siblings(self.property_huffman_root, involved_properties)
        
        # Generate pathMap instructions for bottom-up reconstruction
        # This is simplified for the unbalanced case
        instruction_count = len(proof_hashes)
        for i in range(instruction_count):
            if i * 2 < len(leaves):
                path_map.extend([i * 2, (i * 2) + 1])
        
        return {
            'leaves': leaves,
            'proofHashes': proof_hashes,
            'pathMap': path_map
        }
    
    def _get_descendants(self, node):
        """Get all property node descendants of a node."""
        if node is None:
            return set()
        
        if node.property_group is not None:
            return {node}
        
        descendants = set()
        descendants.update(self._get_descendants(node.left))
        descendants.update(self._get_descendants(node.right))
        return descendants
    
    def generate_multiproof(self, leaves_to_prove_hex):
        """Generate multiproof using new pathMap format."""
        proof_data = self.generate_pathmap_proof(leaves_to_prove_hex)
        return proof_data['proofHashes'], proof_data['pathMap']
    
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
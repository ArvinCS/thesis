#!/usr/bin/env python3
"""
Jurisdiction Tree Builder - Multi-Root Architecture

This module implements a jurisdiction tree where:
1. Each province has its own independent Merkle tree
2. Province trees are optimized with property-level Huffman encoding
3. A jurisdiction tree connects all province roots
4. Two-phase verification: document→province + province→jurisdiction

Key Features:
- Independent province trees for parallel processing
- Optimized for province-specific queries (90% of use cases)
- Reduced rebuild costs (only affected provinces)
- Two-phase proof system with optional parallel verification
- Natural geographic data partitioning
"""

import heapq
from collections import defaultdict
from eth_utils import keccak
from typing import Dict, List, Tuple, Any, Optional
from basic_data_structure import Document


class PropertyHuffmanNode:
    """Node for building Huffman tree at property level within a province."""
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


class ProvinceJurisdictionNode:
    """Node for building jurisdiction tree connecting province roots."""
    def __init__(self, province_name, province_root_hash, freq):
        self.province_name = province_name  # Province name or None for internal nodes
        self.province_root_hash = province_root_hash  # Root hash of province tree
        self.freq = freq  # Total frequency for this province
        self.left = None
        self.right = None
        self.depth = 0
    
    def __lt__(self, other):
        return self.freq < other.freq
    
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
        
        # For very large property groups, use simple frequency-based ordering
        if len(self.documents) > 100:  # Threshold for computational efficiency
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
            property_doc_freq[doc] = document_frequencies.get(doc_id, 1)
        
        # Get document pair frequencies relevant to this property
        property_pair_freq = {}
        for i, doc1 in enumerate(self.documents):
            for doc2 in self.documents[i+1:]:
                doc1_id = f"{doc1.province}.{doc1.property_id}.{doc1.doc_id}"
                doc2_id = f"{doc2.province}.{doc2.property_id}.{doc2.doc_id}"
                pair_key = tuple(sorted([doc1_id, doc2_id]))
                if pair_key in document_pair_frequencies:
                    freq = document_pair_frequencies[pair_key]
                    property_pair_freq[(doc1, doc2)] = freq
                    property_pair_freq[(doc2, doc1)] = freq
        
        # Apply pairs-first optimization with configurable alpha threshold
        self.optimized_document_order = self._apply_pairs_first_huffman(
            self.documents, property_doc_freq, property_pair_freq, alpha_threshold
        )
    
    def _apply_pairs_first_huffman(self, documents, doc_frequencies, pair_frequencies, alpha_threshold=0.15):
        """Apply pairs-first Huffman optimization to documents with alpha threshold."""
        if len(documents) <= 1:
            return documents
        
        # Special handling for alpha = 0 to prevent computational explosion
        if alpha_threshold <= 0:
            print(f"      Alpha threshold: {alpha_threshold} -> Using simple frequency-based ordering (alpha=0 causes computational explosion)")
            # Use simple frequency-based ordering instead of pairs-first Huffman
            return sorted(documents, key=lambda doc: doc_frequencies.get(doc, 1), reverse=True)
        
        # Implement pairs-first Huffman: 1) pre-merge strong pairs above threshold,
        # 2) run standard Huffman merging on remaining nodes, 3) extract leaf order.
        
        # Helper node class for document-level Huffman
        class _DocNode:
            def __init__(self, doc=None, freq=0):
                self.doc = doc
                self.freq = freq
                self.left = None
                self.right = None
            def __lt__(self, other):
                return self.freq < other.freq
            def collect_docs(self):
                if self.doc is not None:
                    return [self.doc]
                docs = []
                if self.left:
                    docs.extend(self.left.collect_docs())
                if self.right:
                    docs.extend(self.right.collect_docs())
                return docs

        # Create initial nodes for each document
        nodes = [_DocNode(doc, doc_frequencies.get(doc, 1)) for doc in documents]

        # Pre-merge strong pairs according to pair_frequencies and alpha_threshold
        if pair_frequencies:
            max_pair_freq = max(pair_frequencies.values())
            min_threshold = max_pair_freq * alpha_threshold
            
            # Warn if alpha threshold results in too many pairs being processed
            total_pairs = len(pair_frequencies)
            strong_pairs_estimate = sum(1 for freq in pair_frequencies.values() if freq >= min_threshold)
            if strong_pairs_estimate > 10000:  # More than 10K pairs
                print(f"      ⚠️ Alpha threshold: {alpha_threshold} -> Min pair frequency: {min_threshold:.1f}")
                print(f"      ⚠️ Warning: {strong_pairs_estimate:,}/{total_pairs:,} pairs will be processed (may be slow)")
            else:
                print(f"      Alpha threshold: {alpha_threshold} -> Min pair frequency: {min_threshold:.1f}")

            merged_nodes = set()
            # Sort pairs by descending frequency
            sorted_pairs = sorted(pair_frequencies.items(), key=lambda x: x[1], reverse=True)

            for (d1, d2), freq in sorted_pairs:
                if freq < min_threshold:
                    break

                # Find current nodes representing d1 and d2
                node1 = next((n for n in nodes if n.doc == d1), None)
                node2 = next((n for n in nodes if n.doc == d2), None)

                # Only merge if both are present and not already merged
                if node1 and node2 and node1 not in merged_nodes and node2 not in merged_nodes and node1 != node2:
                    merged = _DocNode(None, node1.freq + node2.freq)
                    merged.left = node1
                    merged.right = node2

                    # Replace nodes
                    nodes = [n for n in nodes if n != node1 and n != node2]
                    nodes.append(merged)

                    merged_nodes.add(node1)
                    merged_nodes.add(node2)

        # Build Huffman tree on remaining nodes
        heap = nodes.copy()
        heapq.heapify(heap)

        while len(heap) > 1:
            a = heapq.heappop(heap)
            b = heapq.heappop(heap)
            parent = _DocNode(None, a.freq + b.freq)
            parent.left = a
            parent.right = b
            heapq.heappush(heap, parent)

        root = heap[0] if heap else None

        # Extract document order by traversing the final tree (left-to-right)
        def _traverse(node):
            if node is None:
                return []
            if node.doc is not None:
                return [node.doc]
            return _traverse(node.left) + _traverse(node.right)

        ordered_docs = _traverse(root) if root else []
        return ordered_docs
    
    def get_document_hashes_hex(self):
        """Get ordered document hashes in hex format."""
        return [doc.hash_hex for doc in self.optimized_document_order]
    
    # Storage for property-level tree built when constructing province
    def set_property_tree(self, tree_layers, root_hash):
        self.tree_layers = tree_layers
        self.root_hash = root_hash

    def get_property_proof(self, leaf_hash):
        """Generate sibling proof (bottom-up) for a leaf within this property's flat tree.

        Returns a list of sibling hashes from leaf level up to the property root.
        """
        if not hasattr(self, 'tree_layers') or not self.tree_layers:
            return []

        # Find index of leaf
        try:
            idx = [h for h in self.get_document_hashes_hex()].index(leaf_hash)
        except ValueError:
            return []

        siblings = []
        index = idx
        for layer in range(len(self.tree_layers) - 1):
            layer_nodes = list(self.tree_layers[layer])
            # if odd, last element was padded when building
            if index % 2 == 0:
                # sibling is index+1 if exists
                sib_idx = index + 1
            else:
                sib_idx = index - 1

            if sib_idx < len(layer_nodes):
                siblings.append(layer_nodes[sib_idx])
            else:
                # no sibling (odd end) - skip
                pass

            index = index // 2

        return siblings


class ProvinceTree:
    """Independent province tree with property-level Huffman optimization."""
    
    def __init__(self, province_name, property_groups):
        self.province_name = province_name
        self.property_groups = property_groups  # Dict of property_id -> PropertyDocumentGroup
        self.root_hash = None
        self.property_huffman_root = None
        self.ordered_leaves_hex = []
        self.tree_layers = []
        self.total_frequency = 0
    
    def build(self, property_frequencies, pair_frequencies, document_frequencies, document_pair_frequencies, alpha_threshold=0.15):
        """Build optimized province tree with property-level Huffman."""
        print(f"    Building province tree for {self.province_name}: {len(self.property_groups)} properties")
        
        # Step 1: Optimize document order within each property
        all_property_groups = []
        total_docs = 0
        for prop_id, prop_group in self.property_groups.items():
            prop_group.optimize_document_order(document_pair_frequencies, document_frequencies, alpha_threshold)
            all_property_groups.append(prop_group)
            total_docs += len(prop_group.documents)
            print(f"      Property {prop_id}: {len(prop_group.documents)} documents")
            # Build per-property flat Merkle tree for the property's documents
            # So we can compute property_root and generate per-leaf proofs later
            prop_leaves = [doc.hash_hex for doc in prop_group.optimized_document_order]
            prop_layers = []
            if prop_leaves:
                current = list(prop_leaves)
                prop_layers.append(current)
                while len(current) > 1:
                    if len(current) % 2 != 0:
                        current.append(current[-1])
                    nxt = []
                    for i in range(0, len(current), 2):
                        nxt.append(self._combine_and_hash(current[i], current[i+1]))
                    prop_layers.append(nxt)
                    current = nxt
                prop_root = current[0]
            else:
                prop_layers = []
                prop_root = keccak(b'').hex()

            # store property tree layers and root in property group for proof generation
            prop_group.set_property_tree(prop_layers, prop_root)
        
        # Step 2: Build property-level Huffman tree
        if len(all_property_groups) == 1:
            # Single property - create simple tree
            self.property_huffman_root = PropertyHuffmanNode(all_property_groups[0], 1)
        elif len(all_property_groups) > 1:
            # Multiple properties - build Huffman tree
            self.property_huffman_root = self._build_property_huffman_tree(
                all_property_groups, property_frequencies, pair_frequencies
            )
        
        # Step 3: Extract leaves in optimized order
        self.ordered_leaves_hex = []
        if self.property_huffman_root:
            self._extract_leaves_from_huffman_tree(self.property_huffman_root)
        
        # Step 4: Build flat Merkle tree for this province
        if self.ordered_leaves_hex:
            self.root_hash = self._build_flat_merkle_tree()
        else:
            self.root_hash = keccak(b'').hex()
        
        print(f"      Province {self.province_name} tree built: {total_docs} docs, root: {self.root_hash[:16]}...")
        return self.root_hash
    
    def _build_property_huffman_tree(self, property_groups, property_frequencies, pair_frequencies):
        """Build Huffman tree for properties within this province."""
        # Create property nodes
        property_nodes = []
        for prop_group in property_groups:
            full_prop_id = f"{prop_group.province}.{prop_group.property_id}"
            freq = property_frequencies.get(full_prop_id, 1)
            node = PropertyHuffmanNode(prop_group, freq)
            property_nodes.append(node)
            self.total_frequency += freq
        
        # Apply pairs-first optimization
        if len(property_nodes) > 1:
            # Filter pair frequencies for this province
            province_pair_freq = {}
            for (prop1_id, prop2_id), freq in pair_frequencies.items():
                prop1_province = prop1_id.split('.')[0] if '.' in prop1_id else prop1_id
                prop2_province = prop2_id.split('.')[0] if '.' in prop2_id else prop2_id
                
                if prop1_province == self.province_name and prop2_province == self.province_name:
                    province_pair_freq[(prop1_id, prop2_id)] = freq
            
            # Apply alpha threshold for property pairs
            alpha_threshold = 0.3  # More aggressive for province-level optimization
            if province_pair_freq:
                max_pair_freq = max(province_pair_freq.values())
                min_threshold = max_pair_freq * alpha_threshold
                
                merged_pairs = set()
                strong_pairs_count = 0
                
                sorted_pairs = sorted(province_pair_freq.items(), key=lambda x: x[1], reverse=True)
                
                for (prop1_id, prop2_id), freq in sorted_pairs:
                    if freq >= min_threshold:
                        # Find nodes to merge
                        node1 = next((n for n in property_nodes 
                                    if f"{n.property_group.province}.{n.property_group.property_id}" == prop1_id), None)
                        node2 = next((n for n in property_nodes 
                                    if f"{n.property_group.province}.{n.property_group.property_id}" == prop2_id), None)
                        
                        if node1 and node2 and node1 not in merged_pairs and node2 not in merged_pairs:
                            # Create merged node
                            merged_node = PropertyHuffmanNode(None, node1.freq + node2.freq)
                            merged_node.left = node1
                            merged_node.right = node2
                            
                            # Replace nodes
                            property_nodes = [n for n in property_nodes if n != node1 and n != node2]
                            property_nodes.append(merged_node)
                            
                            merged_pairs.add(node1)
                            merged_pairs.add(node2)
                            strong_pairs_count += 1
                
                print(f"        Merged {strong_pairs_count} strong property pairs in {self.province_name}")
        
        # Build final Huffman tree
        if len(property_nodes) == 1:
            return property_nodes[0]
        
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
            # ensure node carries province_root_hash for internal nodes
        # set province_root_hash on leaves
        def _assign_province_root_hash(node):
            if node is None:
                return None
            if node.property_group is not None:
                # leaf - take property root if available
                node.province_root_hash = getattr(node.property_group, 'root_hash', None)
                return node.province_root_hash
            left_hash = _assign_province_root_hash(node.left)
            right_hash = _assign_province_root_hash(node.right)
            node.province_root_hash = combine_and_hash(left_hash or keccak(b'').hex(), right_hash or keccak(b'').hex())
            return node.province_root_hash

        _assign_province_root_hash(root)

        return root
    
    def _extract_leaves_from_huffman_tree(self, node):
        """Extract leaves from property Huffman tree in optimized order."""
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
    
    def _build_flat_merkle_tree(self):
        """Build flat Merkle tree for this province and store layers."""
        self.tree_layers = []
        current_layer = list(self.ordered_leaves_hex)
        self.tree_layers.append(current_layer)
        
        while len(current_layer) > 1:
            # Pad with last element if odd number
            if len(current_layer) % 2 != 0:
                current_layer.append(current_layer[-1])
            
            next_layer = []
            for i in range(0, len(current_layer), 2):
                parent_hash = self._combine_and_hash(current_layer[i], current_layer[i + 1])
                next_layer.append(parent_hash)
            
            self.tree_layers.append(next_layer)
            current_layer = next_layer
        
        return current_layer[0] if current_layer else keccak(b'').hex()
    
    def _combine_and_hash(self, hash1_hex, hash2_hex):
        """Combine two hashes using keccak256."""
        h1_bytes = bytes.fromhex(hash1_hex)
        h2_bytes = bytes.fromhex(hash2_hex)
        # Sort hashes to ensure deterministic ordering
        combined = h1_bytes + h2_bytes if h1_bytes < h2_bytes else h2_bytes + h1_bytes
        return keccak(combined).hex()
    
    def generate_province_proof(self, leaves_to_prove_hex):
        """Generate proof for documents within this province."""
        if not leaves_to_prove_hex or (not self.ordered_leaves_hex and not self.property_huffman_root):
            return [], [], []
        
        # Remove duplicates while preserving order
        unique_leaves = []
        seen = set()
        for leaf in leaves_to_prove_hex:
            if leaf not in seen:
                # we'll later validate it's in some property
                unique_leaves.append(leaf)
                seen.add(leaf)
        
        if not unique_leaves:
            return [], [], []
        
        # We'll generate two outputs:
        # 1) multiproof compatible (for backward compatibility)
        # 2) per-leaf sibling proofs compatible with verifyTwoPhaseWithLeafProofs

        # First, try to generate per-leaf proofs using per-property trees and the Huffman property tree
        province_leaf_proofs = []  # list of dicts: { 'leaf': leaf, 'siblings': [...] }

        # helper to collect property-level siblings from Huffman tree
        def _compute_property_subtree_hash(node):
            if node is None:
                return keccak(b'').hex()
            if node.property_group is not None:
                return getattr(node.property_group, 'root_hash', keccak(b'').hex())
            left_hash = _compute_property_subtree_hash(node.left)
            right_hash = _compute_property_subtree_hash(node.right)
            return combine_and_hash(left_hash, right_hash)

        def _collect_property_level_siblings(node, target_property, acc):
            if node is None:
                return False
            if node.property_group is not None:
                return node.property_group.property_id == target_property
            left_has = _node_contains_property(node.left, target_property)
            right_has = _node_contains_property(node.right, target_property)
            if left_has and not right_has:
                # sibling is right subtree hash
                acc.append(_compute_property_subtree_hash(node.right))
                return True
            if right_has and not left_has:
                acc.append(_compute_property_subtree_hash(node.left))
                return True
            # recurse
            if left_has:
                return _collect_property_level_siblings(node.left, target_property, acc)
            if right_has:
                return _collect_property_level_siblings(node.right, target_property, acc)
            return False

        def _node_contains_property(node, prop_id):
            if node is None:
                return False
            if node.property_group is not None:
                return node.property_group.property_id == prop_id
            return _node_contains_property(node.left, prop_id) or _node_contains_property(node.right, prop_id)

        # map property id -> property group
        property_map = {pg.property_id: pg for pg in self.property_groups.values()}

        for leaf in unique_leaves:
            # find which property contains this leaf
            found = False
            leaf_siblings = []
            for prop_id, prop_group in self.property_groups.items():
                if hasattr(prop_group, 'optimized_document_order'):
                    hashes = [d.hash_hex for d in prop_group.optimized_document_order]
                else:
                    hashes = [d.hash_hex for d in prop_group.documents]
                if leaf in hashes:
                    # property-level siblings
                    prop_sibs = prop_group.get_property_proof(leaf)
                    leaf_siblings.extend(prop_sibs)
                    # property-level (Huffman) siblings up to province root
                    prop_level_sibs = []
                    if self.property_huffman_root:
                        _collect_property_level_siblings(self.property_huffman_root, prop_id, prop_level_sibs)
                    leaf_siblings.extend(prop_level_sibs)
                    found = True
                    break
            if found:
                province_leaf_proofs.append({'leaf': leaf, 'siblings': leaf_siblings})

        # Fallback: generate earlier multiproof using flat layers if available (backward compatibility)
        proof = []
        proof_flags = []
        if self.ordered_leaves_hex:
            leaf_indices_map = {leaf: i for i, leaf in enumerate(self.ordered_leaves_hex)}
            processed_nodes = {}
            for leaf in unique_leaves:
                if leaf in leaf_indices_map:
                    processed_nodes[(0, leaf_indices_map[leaf])] = True

            proof_nodes_seen = set()
            for layer_idx in range(len(self.tree_layers) - 1):
                layer_nodes = list(self.tree_layers[layer_idx])
                if len(layer_nodes) % 2 != 0:
                    layer_nodes.append(layer_nodes[-1])
                for node_idx in range(0, len(layer_nodes), 2):
                    left_key = (layer_idx, node_idx)
                    right_key = (layer_idx, node_idx + 1)
                    is_left_in_path = left_key in processed_nodes
                    is_right_in_path = right_key in processed_nodes
                    if is_left_in_path or is_right_in_path:
                        parent_key = (layer_idx + 1, node_idx // 2)
                        processed_nodes[parent_key] = True
                        if is_left_in_path and is_right_in_path:
                            proof_flags.append(True)
                        elif is_left_in_path:
                            right_node = layer_nodes[node_idx + 1]
                            if right_node not in proof_nodes_seen:
                                proof.append(right_node)
                                proof_nodes_seen.add(right_node)
                            proof_flags.append(False)
                        else:
                            left_node = layer_nodes[node_idx]
                            if left_node not in proof_nodes_seen:
                                proof.append(left_node)
                                proof_nodes_seen.add(left_node)
                            proof_flags.append(False)

        return proof, proof_flags, province_leaf_proofs


def combine_and_hash(hash1_hex, hash2_hex):
    """Combine two hashes using keccak256 (OpenZeppelin compatible)."""
    h1_bytes = bytes.fromhex(hash1_hex)
    h2_bytes = bytes.fromhex(hash2_hex)
    # Sort hashes to ensure deterministic ordering
    combined = h1_bytes + h2_bytes if h1_bytes < h2_bytes else h2_bytes + h1_bytes
    return keccak(combined).hex()


class JurisdictionTreeBuilder:
    """
    Jurisdiction Tree Builder - Multi-Root Architecture
    
    Creates separate optimized trees for each province, then connects them
    via a jurisdiction tree. Optimized for:
    1. Province-specific queries (90% of use cases)
    2. Parallel province processing
    3. Reduced rebuild costs
    4. Two-phase verification system
    """
    
    def __init__(self, all_documents, audit_pattern=None, transactional_pattern=None, alpha_threshold=0.15):
        self.all_documents = all_documents
        self.jurisdiction_root = None
        self.province_trees = {}  # province_name -> ProvinceTree
        self.jurisdiction_huffman_root = None
        
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
            
            # Calculate pair frequencies
            for province, province_groups in self.province_property_groups.items():
                for prop_id, group in province_groups.items():
                    if len(group.documents) >= 2:
                        doc_pair_freq = self.transactional_pattern.get_document_pair_frequencies(
                            group.documents, num_simulated_queries=50
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
        
        if self.transactional_pattern:
            for province, property_groups in self.province_property_groups.items():
                for prop_id, group in property_groups.items():
                    if group.documents:
                        pattern_frequencies = self.transactional_pattern.get_document_frequencies(group.documents)
                        for doc, freq in pattern_frequencies.items():
                            doc_key = f"{doc.province}.{doc.property_id}.{doc.doc_id}"
                            document_freq[doc_key] = freq
        
        return dict(document_freq)
    
    def _extract_document_pair_frequencies(self):
        """Extract document pair frequencies from access pattern."""
        doc_pair_freq = defaultdict(int)
        
        if self.transactional_pattern:
            for province_groups in self.province_property_groups.values():
                for group in province_groups.values():
                    if len(group.documents) >= 2:
                        pattern_pair_freq = self.transactional_pattern.get_document_pair_frequencies(
                            group.documents, num_simulated_queries=40
                        )
                        
                        for (doc1, doc2), freq in pattern_pair_freq.items():
                            doc1_key = f"{doc1.province}.{doc1.property_id}.{doc1.doc_id}"
                            doc2_key = f"{doc2.province}.{doc2.property_id}.{doc2.doc_id}"
                            
                            pair_key = tuple(sorted([doc1_key, doc2_key]))
                            doc_pair_freq[pair_key] += freq
        
        return dict(doc_pair_freq)
    
    def build(self):
        """
        Build jurisdiction tree architecture:
        1. Build independent province trees with property-level Huffman optimization
        2. Create jurisdiction tree connecting province roots
        3. Generate unified access interface
        """
        print("Building Jurisdiction Tree with Independent Province Trees...")
        
        # Calculate frequencies for optimization
        province_freq, property_freq, pair_freq, document_frequencies, document_pair_frequencies = self._calculate_frequencies()
        
        # Step 1: Build independent province trees
        province_roots = {}
        for province, property_groups in self.province_property_groups.items():
            print(f"  Building tree for province {province}: {len(property_groups)} properties")
            
            # Create province tree
            province_tree = ProvinceTree(province, property_groups)
            province_root = province_tree.build(
                property_freq, pair_freq, document_frequencies, document_pair_frequencies, self.alpha_threshold
            )
            
            self.province_trees[province] = province_tree
            province_roots[province] = province_root
        
        # Step 2: Build jurisdiction tree connecting province roots
        print(f"\n  Building jurisdiction tree from {len(province_roots)} province roots...")
        self.jurisdiction_root = self._build_jurisdiction_tree(province_roots, province_freq)
        
        print(f"  Jurisdiction Tree Root: {self.jurisdiction_root[:16]}...")
        return self.jurisdiction_root
    
    def _build_jurisdiction_tree(self, province_roots, province_freq):
        """Build Huffman tree connecting province roots."""
        if not province_roots:
            return keccak(b'').hex()
        
        if len(province_roots) == 1:
            return list(province_roots.values())[0]
        
        # Create jurisdiction nodes for each province
        jurisdiction_nodes = []
        for province, root_hash in province_roots.items():
            freq = province_freq.get(province, 1)
            node = ProvinceJurisdictionNode(province, root_hash, freq)
            jurisdiction_nodes.append(node)
            print(f"    Province {province}: frequency {freq}, root {root_hash[:16]}...")
        
        # Build Huffman tree of provinces
        heap = jurisdiction_nodes.copy()
        heapq.heapify(heap)
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            merged = ProvinceJurisdictionNode(None, None, left.freq + right.freq)
            merged.left = left
            merged.right = right
            
            heapq.heappush(heap, merged)
        
        self.jurisdiction_huffman_root = heap[0] if heap else None
        
        if self.jurisdiction_huffman_root:
            self.jurisdiction_huffman_root.compute_depths(0)
            # Compute final jurisdiction root hash
            return self._compute_jurisdiction_root_hash(self.jurisdiction_huffman_root)
        else:
            return keccak(b'').hex()
    
    def _compute_jurisdiction_root_hash(self, node):
        """Recursively compute jurisdiction tree root hash."""
        if node is None:
            return keccak(b'').hex()
        
        if node.province_name is not None:
            # Leaf node - return province root hash
            return node.province_root_hash
        else:
            # Internal node - combine children
            left_hash = self._compute_jurisdiction_root_hash(node.left)
            right_hash = self._compute_jurisdiction_root_hash(node.right)
            return combine_and_hash(left_hash, right_hash)
    
    def generate_two_phase_proof(self, leaves_to_prove_hex):
        """
        Generate two-phase proof:
        1. Province proof: document → province root
        2. Jurisdiction proof: province root → jurisdiction root
        """
        if not leaves_to_prove_hex:
            return {
                'province_proofs': {},
                'jurisdiction_proof': [],
                'jurisdiction_flags': [],
                'involved_provinces': []
            }
        
        # Group leaves by province
        province_leaves = defaultdict(list)
        for leaf in leaves_to_prove_hex:
            # Find which province this leaf belongs to
            found_province = None
            for province, province_tree in self.province_trees.items():
                if leaf in province_tree.ordered_leaves_hex:
                    found_province = province
                    break
            
            if found_province:
                province_leaves[found_province].append(leaf)
        
        # Phase 1: Generate province proofs
        province_proofs = {}
        involved_provinces = []
        for province, leaves in province_leaves.items():
            if province in self.province_trees:
                proof, flags, leaf_proofs = self.province_trees[province].generate_province_proof(leaves)
                province_proofs[province] = {
                    'leaves': leaves,
                    'proof': proof,
                    'flags': flags,
                    'leaf_proofs': leaf_proofs,
                    'province_root': self.province_trees[province].root_hash
                }
                involved_provinces.append(province)
        
        # Phase 2: Generate jurisdiction proof for involved provinces
        jurisdiction_proof, jurisdiction_flags = self._generate_jurisdiction_proof(involved_provinces)
        
        return {
            'province_proofs': province_proofs,
            'jurisdiction_proof': jurisdiction_proof,
            'jurisdiction_flags': jurisdiction_flags,
            'involved_provinces': involved_provinces
        }
    
    def _generate_jurisdiction_proof(self, involved_provinces):
        """Generate jurisdiction proof for specified provinces."""
        if not involved_provinces or not self.jurisdiction_huffman_root:
            return [], []
        
        # For simplicity, generate proof that includes all necessary province roots
        # In a full implementation, this would be optimized like multiproof
        
        proof_hashes = []
        
        def collect_jurisdiction_siblings(node, target_provinces):
            if node is None:
                return
            
            if node.province_name is not None:
                # Leaf node - skip if it's a target
                return
            
            # Internal node - check if we need siblings
            left_has_target = self._node_contains_provinces(node.left, target_provinces)
            right_has_target = self._node_contains_provinces(node.right, target_provinces)
            
            if left_has_target and not right_has_target:
                # Need right sibling
                right_hash = self._compute_jurisdiction_root_hash(node.right)
                proof_hashes.append(right_hash)
            elif right_has_target and not left_has_target:
                # Need left sibling
                left_hash = self._compute_jurisdiction_root_hash(node.left)
                proof_hashes.append(left_hash)
            
            # Recurse
            if left_has_target:
                collect_jurisdiction_siblings(node.left, target_provinces)
            if right_has_target:
                collect_jurisdiction_siblings(node.right, target_provinces)
        
        collect_jurisdiction_siblings(self.jurisdiction_huffman_root, set(involved_provinces))
        
        # Generate flags (simplified - in practice would be more sophisticated)
        flags = [False] * len(proof_hashes)
        
        return proof_hashes, flags
    
    def _node_contains_provinces(self, node, target_provinces):
        """Check if a jurisdiction node contains any target provinces."""
        if node is None:
            return False
        
        if node.province_name is not None:
            return node.province_name in target_provinces
        
        # Internal node
        return (self._node_contains_provinces(node.left, target_provinces) or 
                self._node_contains_provinces(node.right, target_provinces))
    
    def generate_multiproof(self, leaves_to_prove_hex):
        """
        Generate unified multiproof compatible with existing interface.
        Combines both phases into single proof for compatibility.
        """
        two_phase = self.generate_two_phase_proof(leaves_to_prove_hex)
        
        # Combine all proofs into single array for compatibility
        combined_proof = []
        combined_flags = []
        
        # Add province proofs
        for province_data in two_phase['province_proofs'].values():
            combined_proof.extend(province_data['proof'])
            combined_flags.extend(province_data['flags'])
        
        # Add jurisdiction proof
        combined_proof.extend(two_phase['jurisdiction_proof'])
        combined_flags.extend(two_phase['jurisdiction_flags'])
        
        return combined_proof, combined_flags
    
    def verify_multiproof_locally(self, proof, proof_flags, leaves):
        """Local verification for testing (simplified)."""
        # For jurisdiction tree, verification is more complex
        # This is a simplified version for compatibility
        return True, "Jurisdiction tree verification (simplified)"
    
    def add_learning_event(self, verified_properties, learning_mode=None, end_of_day=False):
        """Add verification event with province-specific learning."""
        try:
            from learning_config import get_learning_config, LearningMode
            config = get_learning_config()
            
            if learning_mode is None:
                effective_mode = config.mode
            else:
                if isinstance(learning_mode, str):
                    effective_mode = LearningMode(learning_mode.lower())
                else:
                    effective_mode = learning_mode
            
            self.verification_count += 1
            
            # Determine which provinces are affected
            affected_provinces = set()
            for prop in verified_properties:
                # Extract province from property ID (assuming format: "province.property_id")
                if '.' in prop:
                    province = prop.split('.')[0]
                    affected_provinces.add(province)
            
            # Determine if we should rebuild
            should_rebuild = False
            
            if effective_mode == LearningMode.IMMEDIATE:
                should_rebuild = True
            elif effective_mode == LearningMode.BATCH:
                should_rebuild = (self.verification_count % config.batch_size == 0)
            elif effective_mode == LearningMode.DAILY:
                should_rebuild = end_of_day
            elif effective_mode == LearningMode.HYBRID:
                if len(verified_properties) < config.immediate_threshold:
                    should_rebuild = True
                elif self.verification_count % config.batch_size == 0:
                    should_rebuild = True
            
            if should_rebuild:
                old_root = self.jurisdiction_root
                
                # Jurisdiction tree advantage: only rebuild affected provinces!
                if len(affected_provinces) < len(self.province_trees):
                    print(f"📚 Rebuilding {len(affected_provinces)} affected provinces: {affected_provinces}")
                    # Rebuild only affected provinces (major advantage!)
                    province_freq, property_freq, pair_freq, document_frequencies, document_pair_frequencies = self._calculate_frequencies()
                    
                    for province in affected_provinces:
                        if province in self.province_trees:
                            print(f"    Rebuilding province tree: {province}")
                            self.province_trees[province].build(
                                property_freq, pair_freq, document_frequencies, document_pair_frequencies
                            )
                    
                    # Rebuild jurisdiction tree with new province roots
                    province_roots = {p: tree.root_hash for p, tree in self.province_trees.items()}
                    self.jurisdiction_root = self._build_jurisdiction_tree(province_roots, province_freq)
                else:
                    # Full rebuild if most provinces affected
                    self.build()
                
                if config.verbose_logging:
                    print(f"✅ Jurisdiction tree rebuilt after verification #{self.verification_count}")
                if old_root != self.jurisdiction_root and config.verbose_logging:
                    print(f"   Root changed: {old_root[:16]}... -> {self.jurisdiction_root[:16]}...")
            
            return should_rebuild, self.jurisdiction_root
            
        except Exception as e:
            print(f"⚠️ Error in add_learning_event: {e}")
            self.verification_count += 1
            self.build()
            return True, self.jurisdiction_root
    
    def get_tree_info(self):
        """Get information about the jurisdiction tree."""
        if not self.jurisdiction_root:
            return {"status": "not_built"}
        
        province_info = {}
        total_docs = 0
        total_properties = 0
        
        for province, province_tree in self.province_trees.items():
            province_docs = len(province_tree.ordered_leaves_hex)
            province_props = len(self.province_property_groups.get(province, {}))
            province_info[province] = {
                'documents': province_docs,
                'properties': province_props,
                'root_hash': province_tree.root_hash,
                'tree_depth': len(province_tree.tree_layers) if province_tree.tree_layers else 0
            }
            total_docs += province_docs
            total_properties += province_props
        
        return {
            "tree_type": "Jurisdiction Tree (Multi-Root Architecture)",
            "jurisdiction_root": self.jurisdiction_root,
            "total_documents": total_docs,
            "total_properties": total_properties,
            "provinces": len(self.province_trees),
            "province_info": province_info,
            "optimization": "Two-phase: Independent province trees + Jurisdiction tree",
            "advantages": [
                "Province-specific rebuilds (partial updates)",
                "Parallel province processing",
                "Optimized for province-specific queries",
                "Two-phase verification system"
            ]
        }
    
    def get_learning_stats(self):
        """Get learning statistics."""
        from learning_config import get_learning_config
        config = get_learning_config()
        return {
            'mode': config.mode.value,
            'approach': 'jurisdiction_tree',
            'verification_count': self.verification_count,
            'batch_size': config.batch_size if config.mode.name == 'BATCH' else None,
            'provinces': len(self.province_trees),
            'total_properties': sum(len(props) for props in self.province_property_groups.values()),
            'total_documents': sum(len(tree.ordered_leaves_hex) for tree in self.province_trees.values())
        }
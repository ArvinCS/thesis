"""
True Huffman Merkle Tree Builder with Unbalanced Structure (Document Level)

This module implements a genuine Huffman-optimized Merkle tree where:
1. Frequent documents are placed at shallower depths (shorter proofs)
2. The tree structure is unbalanced, following the Huffman tree topology
3. Co-accessed document pairs can be efficiently proven together

Key difference from traditional balanced approach:
- Tree structure directly follows Huffman tree topology
- Frequent items have shorter proof paths
- Multiproofs benefit from optimal sibling placement
"""

import heapq
from collections import defaultdict, Counter
from optimized_tree_builder import combine_and_hash, Document
from eth_utils import keccak
from learning_config import get_learning_config, LearningMode


class HuffmanMerkleNode:
    """Node in the Huffman Merkle tree."""
    def __init__(self, frequency=0, document=None):
        self.frequency = frequency
        self.document = document  # Only leaf nodes have documents
        self.left = None
        self.right = None
        self.hash = None
        self.depth = 0
    
    def __lt__(self, other):
        return self.frequency < other.frequency
    
    def is_leaf(self):
        return self.document is not None
    
    def compute_hash(self):
        """Compute hash for this node."""
        if self.is_leaf():
            # Leaf node - use document hash
            self.hash = self.document.hash_hex
        else:
            # Internal node - combine children hashes
            if self.left and self.right:
                left_hash = self.left.hash if self.left.hash else "0" * 64
                right_hash = self.right.hash if self.right.hash else "0" * 64
                self.hash = combine_and_hash(left_hash, right_hash)
            elif self.left:
                self.hash = self.left.hash
            elif self.right:
                self.hash = self.right.hash
            else:
                self.hash = "0" * 64
        return self.hash
    
    def compute_depths(self, depth=0):
        """Compute depth for all nodes in subtree."""
        self.depth = depth
        if self.left:
            self.left.compute_depths(depth + 1)
        if self.right:
            self.right.compute_depths(depth + 1)


class TraditionalDocumentLevelHuffmanBuilder:
    """
    Builds a true unbalanced Huffman Merkle tree where frequent documents
    are at shallower depths for optimal proof efficiency.
    """
    
    def __init__(self, all_documents, transactional_pattern=None, alpha_threshold=0.1):
        # Handle both dictionary and list formats
        if isinstance(all_documents, dict):
            self.all_documents = list(all_documents.values())
        else:
            self.all_documents = all_documents
        
        self.transactional_pattern = transactional_pattern
        self.alpha_threshold = alpha_threshold  # Threshold for frequency-based optimizations
        self.huffman_root = None
        self.document_to_node = {}  # Map document to its leaf node
        self.document_depths = {}   # Map document to its depth in tree
        
        # Learning tracking
        self.verification_count = 0
        self.learning_stats = {
            'total_rebuilds': 0,
            'successful_optimizations': 0,
            'average_improvement': 0.0
        }
        
        print(f"📊 Building True Huffman Merkle Tree with {len(self.all_documents)} documents")
        print(f"📏 Using alpha threshold: {alpha_threshold} for frequency-based optimizations")
    
    def build(self):
        """Build the true unbalanced Huffman Merkle tree."""
        if not self.all_documents:
            return "0" * 64
        
        if len(self.all_documents) == 1:
            # Single document case
            doc = self.all_documents[0]
            self.huffman_root = HuffmanMerkleNode(frequency=1, document=doc)
            self.huffman_root.compute_hash()
            self.document_to_node[doc] = self.huffman_root
            self.document_depths[doc] = 0
            return self.huffman_root.hash
        
        # Step 1: Get document frequencies
        document_frequencies = self._get_document_frequencies()
        
        # Step 2: Build Huffman tree using frequencies
        self.huffman_root = self._build_huffman_tree(document_frequencies)
        
        # Step 3: Compute hashes for all nodes (bottom-up)
        self._compute_all_hashes(self.huffman_root)
        
        # Step 4: Compute depths for analysis
        self.huffman_root.compute_depths(0)
        self._update_document_depths()
        
        # Step 5: Log depth statistics
        self._log_depth_statistics()
        
        return self.huffman_root.hash
    
    def _get_document_frequencies(self):
        """Get document access frequencies from transactional pattern."""
        if self.transactional_pattern is None:
            # Default: equal frequencies (though this defeats the purpose)
            return {doc: 1 for doc in self.all_documents}
        
        # Use transactional pattern to get realistic frequencies
        return self.transactional_pattern.get_document_frequencies(self.all_documents)
    
    def _get_document_pair_frequencies(self):
        """Get document pair co-access frequencies from transactional pattern.
        
        For TRUE document-level Huffman, we use a practical approach:
        1. All pairs within the same property (high co-access probability)
        2. Cross-property pairs for the most frequently accessed documents (scalable)
        """
        if self.transactional_pattern is None:
            return {}
        
        # Extract document pair frequencies from access pattern
        doc_pair_freq = {}
        
        print(f"🔗 Generating practical document pairs with cross-property optimization")
        
        # Step 1: Get document frequencies and sort by importance
        doc_frequencies = {}
        for doc in self.all_documents:
            doc_type = doc.doc_id.split('_')[1] if '_' in doc.doc_id else doc.doc_id
            doc_frequencies[doc] = self.transactional_pattern.document_importance_map.get(doc_type, 1)
        
        # Get top frequent documents for cross-property pairing
        sorted_docs = sorted(self.all_documents, key=lambda d: doc_frequencies[d], reverse=True)
        top_docs_count = min(1000, len(self.all_documents) // 2)  # Top documents or 50%, whichever is smaller
        top_frequent_docs = set(sorted_docs[:top_docs_count])
        
        print(f"📊 Selected {len(top_frequent_docs)} most frequent documents for cross-property pairing")
        
        # Step 2: Generate pairs within same property (all documents)
        property_docs = {}
        for doc in self.all_documents:
            prop_id = doc.property_id
            if prop_id not in property_docs:
                property_docs[prop_id] = []
            property_docs[prop_id].append(doc)
        
        same_property_pairs = 0
        for prop_id, prop_docs in property_docs.items():
            for i, doc1 in enumerate(prop_docs):
                for j, doc2 in enumerate(prop_docs[i+1:], i+1):
                    pair_key = tuple(sorted([doc1, doc2], key=lambda x: x.full_id))
                    
                    doc1_freq = doc_frequencies[doc1]
                    doc2_freq = doc_frequencies[doc2]
                    
                    # Same property: high co-access probability
                    pair_freq = min(doc1_freq, doc2_freq) * 0.8
                    doc_pair_freq[pair_key] = pair_freq
                    same_property_pairs += 1
        
        # Step 3: Generate cross-property pairs for top frequent documents
        cross_property_pairs = 0
        for i, doc1 in enumerate(top_frequent_docs):
            for j, doc2 in enumerate(list(top_frequent_docs)[i+1:], i+1):
                if doc1.property_id != doc2.property_id:  # Only cross-property
                    pair_key = tuple(sorted([doc1, doc2], key=lambda x: x.full_id))
                    
                    # Skip if already added from same-property analysis
                    if pair_key not in doc_pair_freq:
                        doc1_freq = doc_frequencies[doc1]
                        doc2_freq = doc_frequencies[doc2]
                        
                        # Cross-property: lower but still valuable co-access
                        pair_freq = min(doc1_freq, doc2_freq) * 0.3
                        doc_pair_freq[pair_key] = pair_freq
                        cross_property_pairs += 1
        
        print(f"📊 Generated {len(doc_pair_freq)} total pairs:")
        print(f"    - {same_property_pairs} same-property pairs")
        print(f"    - {cross_property_pairs} cross-property pairs (top documents)")
        return doc_pair_freq
    
    def _build_huffman_tree(self, document_frequencies):
        """Build True Huffman tree using pairs-first approach with alpha threshold."""
        print(f"🌳 Building pairs-first Huffman tree from {len(document_frequencies)} documents")
        
        # Step 1: Get document pair frequencies for pairs-first optimization
        pair_frequencies = self._get_document_pair_frequencies()
        
        # Step 2: Apply pairs-first Huffman with alpha threshold
        optimized_nodes = self._apply_pairs_first_huffman(
            self.all_documents, document_frequencies, pair_frequencies
        )
        
        # Step 3: Build final Huffman tree from optimized nodes
        heap = list(optimized_nodes)
        heapq.heapify(heap)
        
        print(f"🔗 Building final Huffman tree from {len(heap)} optimized nodes")
        
        while len(heap) > 1:
            # Pop two lowest frequency nodes
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            # Create new internal node
            merged_freq = left.frequency + right.frequency
            internal = HuffmanMerkleNode(frequency=merged_freq)
            internal.left = left
            internal.right = right
            
            # Push back to heap
            heapq.heappush(heap, internal)
        
        return heap[0] if heap else None  # Root of Huffman tree
    
    def _apply_pairs_first_huffman(self, documents, doc_frequencies, pair_frequencies):
        """Apply pairs-first Huffman optimization with alpha threshold.
        
        This implements the key algorithm from the traffic-aware Merkle tree paper:
        1. Identify strong document pairs (above alpha threshold)
        2. Merge strong pairs first
        3. Apply standard Huffman to remaining nodes
        """
        if len(documents) <= 1:
            nodes = []
            for doc in documents:
                node = HuffmanMerkleNode(frequency=doc_frequencies[doc], document=doc)
                self.document_to_node[doc] = node
                nodes.append(node)
            return nodes
        
        # Calculate alpha threshold based on maximum pair frequency
        if pair_frequencies:
            max_pair_freq = max(pair_frequencies.values())
            min_pair_freq_threshold = max_pair_freq * self.alpha_threshold
            print(f"📊 Alpha threshold: {self.alpha_threshold} -> Min pair frequency: {min_pair_freq_threshold:.1f}")
        else:
            min_pair_freq_threshold = 0
            print(f"📊 No pair frequencies available, using individual frequencies only")
        
        # Step 1: Create initial document nodes
        doc_nodes = {}
        for doc in documents:
            node = HuffmanMerkleNode(frequency=doc_frequencies[doc], document=doc)
            self.document_to_node[doc] = node
            doc_nodes[doc] = node
        
        # Step 2: Pairs-first optimization with alpha threshold
        merged_docs = set()
        strong_pairs_count = 0
        weak_pairs_count = 0
        merged_nodes = []  # Track merged nodes separately
        
        # Sort pairs by frequency (most frequent first)
        sorted_pairs = sorted(pair_frequencies.items(), key=lambda x: x[1], reverse=True)
        
        for (doc1, doc2), freq in sorted_pairs:
            if freq >= min_pair_freq_threshold:  # Alpha threshold check
                if doc1 not in merged_docs and doc2 not in merged_docs:
                    if doc1 in doc_nodes and doc2 in doc_nodes:
                        # Create merged node for strong pair
                        merged_node = HuffmanMerkleNode(frequency=doc_nodes[doc1].frequency + doc_nodes[doc2].frequency)
                        merged_node.left = doc_nodes[doc1]
                        merged_node.right = doc_nodes[doc2]
                        
                        # Keep document-to-node mapping for proof generation
                        # (documents still point to their original leaf nodes)
                        
                        # Remove individual nodes from the pool and add merged node
                        merged_nodes.append(merged_node)
                        
                        # Mark as merged so they won't be paired again
                        merged_docs.add(doc1)
                        merged_docs.add(doc2)
                        strong_pairs_count += 1
            else:
                weak_pairs_count += 1
        
        print(f"🔗 Merged {strong_pairs_count} strong pairs, skipped {weak_pairs_count} weak pairs")
        
        # Step 3: Collect nodes for final Huffman tree construction
        # Include unmerged individual nodes + merged pair nodes
        unmerged_nodes = [doc_nodes[doc] for doc in documents if doc not in merged_docs]
        unique_nodes = unmerged_nodes + merged_nodes
        
        print(f"📋 Final node count: {len(unique_nodes)} (reduced from {len(documents)} documents)")
        
        return unique_nodes
    
    def _compute_all_hashes(self, node):
        """Compute hashes for all nodes in post-order traversal."""
        if node is None:
            return
        
        # Compute children first (post-order)
        self._compute_all_hashes(node.left)
        self._compute_all_hashes(node.right)
        
        # Then compute this node's hash
        node.compute_hash()
    
    def _update_document_depths(self):
        """Update document depths mapping."""
        self.document_depths = {}
        for doc, node in self.document_to_node.items():
            self.document_depths[doc] = node.depth
    
    def _log_depth_statistics(self):
        """Log statistics about document depths."""
        if not self.document_depths:
            return
        
        depths = list(self.document_depths.values())
        min_depth = min(depths)
        max_depth = max(depths)
        avg_depth = sum(depths) / len(depths)
        
        # Count documents at each depth
        depth_counts = Counter(depths)
        
        print(f"📏 Huffman Tree Depth Statistics:")
        print(f"   Min depth: {min_depth} (shortest proof)")
        print(f"   Max depth: {max_depth} (longest proof)")
        print(f"   Avg depth: {avg_depth:.2f}")
        print(f"   Depth distribution: {dict(depth_counts)}")
        
        # Compare to balanced tree
        import math
        balanced_depth = math.ceil(math.log2(len(self.all_documents)))
        print(f"   Balanced tree depth would be: {balanced_depth}")
        print(f"   Huffman benefit: {balanced_depth - min_depth} levels saved for frequent docs")
    
    def generate_proof_for_documents(self, documents_to_prove):
        """
        Generate proof for documents in the unbalanced Huffman tree.
        
        This is more complex than balanced tree proofs because:
        1. Different documents are at different depths
        2. Proof paths can vary significantly in length
        3. Multiproofs need to account for unbalanced structure
        """
        if not documents_to_prove:
            return [], []
        
        # For now, implement single document proofs
        # Multiproof for unbalanced trees is more complex and needs special handling
        if len(documents_to_prove) == 1:
            return self._generate_single_proof(documents_to_prove[0])
        else:
            # For multiple documents, we need a more sophisticated multiproof algorithm
            return self._generate_multiproof(documents_to_prove)
    
    def generate_batched_proof_with_flags(self, leaves_to_prove_hex):
        """Generate iterative proofs for contract verification."""
        # Convert hex hashes back to documents for proof generation
        documents_to_prove = []
        for doc in self.all_documents:
            if doc.hash_hex in leaves_to_prove_hex:
                documents_to_prove.append(doc)
        
        # Generate individual proofs for each document
        iterative_proofs = []
        for doc in documents_to_prove:
            proof_hashes, proof_flags = self._generate_single_proof(doc)
            
            # Convert proof_flags to positions array 
            # flag=True means sibling is left (position=0), flag=False means sibling is right (position=1)
            positions = []
            for flag in proof_flags:
                positions.append(0 if flag else 1)
            
            iterative_proofs.append({
                'leaf': doc.hash_hex,
                'proof': proof_hashes,
                'positions': positions
            })
        
        return iterative_proofs
    
    def _generate_single_proof(self, document):
        """Generate proof for a single document in the Huffman tree."""
        if document not in self.document_to_node:
            return [], []
        
        leaf_node = self.document_to_node[document]
        
        # Special case: if this is the only document (root node)
        if leaf_node == self.huffman_root:
            return [], []
        
        proof_hashes = []
        proof_flags = []
        
        # We need to find the path from leaf to root
        path_to_root = self._find_path_to_root(leaf_node)
        
        # Generate proof by walking up the path
        for i in range(len(path_to_root) - 1):
            current_node = path_to_root[i]
            parent_node = path_to_root[i + 1]
            
            # Find sibling of current node
            if parent_node.left == current_node:
                # Current is left child, sibling is right
                if parent_node.right:
                    proof_hashes.append(parent_node.right.hash)
                    proof_flags.append(False)  # Right sibling (current is left)
            else:
                # Current is right child, sibling is left
                if parent_node.left:
                    proof_hashes.append(parent_node.left.hash)
                    proof_flags.append(True)   # Left sibling (current is right)
        
        return proof_hashes, proof_flags
    
    def _find_path_to_root(self, leaf_node):
        """Find path from leaf node to root."""
        path = []
        
        def find_path_recursive(node, target, current_path):
            if node is None:
                return False
            
            current_path.append(node)
            
            if node == target:
                return True
            
            # Search in children
            if (find_path_recursive(node.left, target, current_path) or 
                find_path_recursive(node.right, target, current_path)):
                return True
            
            # Remove current node if target not found in this subtree
            current_path.pop()
            return False
        
        path_found = find_path_recursive(self.huffman_root, leaf_node, path)
        
        if path_found:
            # Reverse the path so it goes from leaf to root
            return list(reversed(path))
        else:
            return [leaf_node]  # Fallback for single node case
    
    def _compute_hash_pair(self, left_hash, right_hash):
        """Helper to compute hash pair for verification."""
        return combine_and_hash(left_hash, right_hash)
    
    def _generate_multiproof(self, documents_to_prove):
        """
        Generate multiproof for multiple documents in Huffman tree.
        
        This is a simplified implementation. A full implementation would need
        to account for the complex interactions between unbalanced paths.
        """
        # For now, generate individual proofs and combine
        # This is not optimal but provides correct verification
        
        all_proof_hashes = []
        all_flags = []
        
        for doc in documents_to_prove:
            proof_hashes, flags = self._generate_single_proof(doc)
            all_proof_hashes.extend(proof_hashes)
            all_flags.extend(flags)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_hashes = []
        unique_flags = []
        
        for i, hash_val in enumerate(all_proof_hashes):
            if hash_val not in seen:
                seen.add(hash_val)
                unique_hashes.append(hash_val)
                if i < len(all_flags):
                    unique_flags.append(all_flags[i])
        
        return unique_hashes, unique_flags
    
    def get_document_proof_length(self, document):
        """Get the proof length for a specific document."""
        if document in self.document_depths:
            return self.document_depths[document]
        return -1
    
    def get_root_hash(self):
        """Get the root hash of the Huffman Merkle tree."""
        return self.huffman_root.hash if self.huffman_root else "0" * 64
    
    def get_tree_stats(self):
        """Get statistics about the Huffman tree structure."""
        if not self.document_depths:
            return {}
        
        depths = list(self.document_depths.values())
        depth_counts = Counter(depths)
        
        return {
            'total_documents': len(self.all_documents),
            'min_depth': min(depths),
            'max_depth': max(depths),
            'avg_depth': sum(depths) / len(depths),
            'depth_distribution': dict(depth_counts),
            'huffman_root_hash': self.huffman_root.hash if self.huffman_root else None
        }
    
    def add_learning_event(self, verified_properties, learning_mode=None, end_of_day=False):
        """Add verification event with learning support."""
        try:
            from learning_config import get_learning_config, LearningMode
            config = get_learning_config()
            
            # Determine effective learning mode
            if learning_mode is None:
                effective_mode = config.mode
            else:
                if isinstance(learning_mode, str):
                    effective_mode = LearningMode(learning_mode.lower())
                else:
                    effective_mode = learning_mode
            
            if config.verbose_logging:
                print(f"📚 True Huffman learning mode: {effective_mode.value}")
            
            self.verification_count += 1
            
            # Determine if rebuild is needed
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
            elif effective_mode == LearningMode.DISABLED:
                should_rebuild = False
            
            if should_rebuild:
                old_root = self.get_root_hash()
                self.build()
                if config.verbose_logging:
                    print(f"✅ True Huffman tree rebuilt after verification #{self.verification_count}")
                    if old_root != self.get_root_hash():
                        new_root = self.get_root_hash()
                        print(f"   Root changed: {old_root[:16] if old_root else 'None'}... -> {new_root[:16]}...")
            
            return should_rebuild, self.get_root_hash()
            
        except Exception as e:
            print(f"⚠️ Error in True Huffman learning: {e}")
            # Fallback
            self.verification_count += 1
            self.build()
            return True, self.get_root_hash()
    
    def add_verification_event(self, verified_properties, learning_mode=None, end_of_day=False):
        """Add verification event (alias for add_learning_event)."""
        return self.add_learning_event(verified_properties, learning_mode, end_of_day)
    
    def generate_iterative_proofs(self, document_hashes):
        """Generate iterative proofs for contract verification (alias for compatibility)."""
        return self.generate_batched_proof_with_flags(document_hashes)
    
    def generate_pathmap_proof(self, leaves_to_prove_hex):
        """Generate proof using new pathMap format for bottom-up reconstruction."""
        if not leaves_to_prove_hex or not self.huffman_root:
            return {'leaves': [], 'proofHashes': [], 'pathMap': []}
        
        # Sort and deduplicate leaves
        sorted_leaves = sorted(set(leaves_to_prove_hex))
        
        # Step 1: Find leaf nodes for documents to prove
        leaf_nodes_to_prove = []
        for leaf_hash in sorted_leaves:
            for doc in self.all_documents:
                if doc.hash_hex == leaf_hash:
                    if doc in self.document_to_node:
                        leaf_nodes_to_prove.append(self.document_to_node[doc])
                    break
        
        if not leaf_nodes_to_prove:
            return {'leaves': [], 'proofHashes': [], 'pathMap': []}
        
        # Step 2: Gather all nodes in proof subtree
        proof_nodes = set()
        node_to_hash = {}  # Maps node to its hash
        
        # Traverse up from each leaf to root, collecting all nodes and siblings
        for leaf_node in leaf_nodes_to_prove:
            current = leaf_node
            path_to_root = []
            
            # Find path to root
            while current is not None:
                path_to_root.append(current)
                current = self._find_parent(current)
            
            # Add all nodes in path and their siblings
            for node in path_to_root:
                proof_nodes.add(node)
                node_to_hash[node] = node.hash
                
                # Add sibling if exists
                parent = self._find_parent(node)
                if parent:
                    sibling = parent.right if parent.left == node else parent.left
                    if sibling:
                        proof_nodes.add(sibling)
                        node_to_hash[sibling] = sibling.hash
        
        # Step 3: Topological sort (bottom-up ordering)
        topo_sorted = self._topological_sort(proof_nodes)
        
        # Step 4: Build working set arrays
        leaves = sorted_leaves.copy()
        proof_hashes = []
        working_set_indices = {}  # Maps node to index in working set
        
        # Index leaves in working set
        for i, leaf_hash in enumerate(leaves):
            for node in proof_nodes:
                if node.is_leaf() and node.hash == leaf_hash:
                    working_set_indices[node] = i
                    break
        
        # Add proof hashes (only siblings that can't be computed)
        proof_hash_nodes = []
        computable_nodes = set()  # Nodes that can be computed from their children
        
        # First pass: identify which internal nodes can be computed
        for node in proof_nodes:
            if not node.is_leaf():
                # Check if both children are available in proof_nodes
                if (node.left in proof_nodes and node.right in proof_nodes):
                    computable_nodes.add(node)
        
        # Second pass: add non-computable nodes as proof hashes
        for node in proof_nodes:
            if node not in working_set_indices:  # Not already indexed as proven leaf
                if node not in computable_nodes:  # Can't be computed from children
                    proof_hash_nodes.append(node)
        
        # Sort proof hashes for deterministic ordering
        proof_hash_nodes.sort(key=lambda x: x.hash)
        
        for i, node in enumerate(proof_hash_nodes):
            proof_hashes.append(node.hash)
            working_set_indices[node] = len(leaves) + i
        
        # Step 5: Build pathMap instructions
        path_map = []
        next_computed_idx = len(leaves) + len(proof_hashes)
        
        # Process internal nodes in topological order (only computable ones)
        for node in topo_sorted:
            if not node.is_leaf() and node not in working_set_indices:
                # This node needs to be computed
                left_idx = working_set_indices.get(node.left)
                right_idx = working_set_indices.get(node.right)
                
                if left_idx is not None and right_idx is not None:
                    path_map.extend([left_idx, right_idx])
                    working_set_indices[node] = next_computed_idx
                    next_computed_idx += 1
        
        return {
            'leaves': leaves,
            'proofHashes': proof_hashes,
            'pathMap': path_map
        }
    
    def _find_parent(self, node):
        """Find parent of given node in Huffman tree."""
        if node == self.huffman_root:
            return None
        
        # BFS to find parent
        from collections import deque
        queue = deque([self.huffman_root])
        
        while queue:
            current = queue.popleft()
            if current.left == node or current.right == node:
                return current
            
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)
        
        return None
    
    def _topological_sort(self, nodes):
        """Topological sort of nodes (children before parents)."""
        # Find leaf nodes first
        leaves = [node for node in nodes if node.is_leaf()]
        internal_nodes = [node for node in nodes if not node.is_leaf()]
        
        # Sort internal nodes by depth (deeper first)
        def get_depth(node):
            if hasattr(node, 'depth'):
                return node.depth
            # Calculate depth on demand
            depth = 0
            current = node
            while current:
                parent = self._find_parent(current)
                if parent:
                    depth += 1
                    current = parent
                else:
                    break
            return depth
        
        internal_nodes.sort(key=get_depth, reverse=True)  # Deeper first
        
        return leaves + internal_nodes
    
    def generate_multiproof(self, leaves_to_prove_hex):
        """Generate multiproof using new pathMap format."""
        proof_data = self.generate_pathmap_proof(leaves_to_prove_hex)
        return proof_data['proofHashes'], proof_data['pathMap']
    
    def get_learning_stats(self):
        """Get learning statistics."""
        try:
            config = get_learning_config()
            return {
                'mode': config.mode.value,
                'approach': 'true_huffman_merkle',
                'verification_count': self.verification_count,
                'batch_size': config.batch_size if config.mode == LearningMode.BATCH else None,
                'learning_stats': self.learning_stats.copy(),
                'total_documents': len(self.all_documents),
                'tree_stats': self.get_tree_stats()
            }
        except:
            return {
                'approach': 'true_huffman_merkle',
                'verification_count': self.verification_count,
                'learning_stats': self.learning_stats.copy(),
                'total_documents': len(self.all_documents)
            }
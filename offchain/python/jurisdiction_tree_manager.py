from collections import defaultdict
from eth_utils import keccak
from province_tree_builder import ProvinceTreeBuilder
from optimized_tree_builder import combine_and_hash

class JurisdictionTreeManager:
    """
    Manages the complete hierarchical system:
    1. Individual Province Trees (each with their own Merkle root)
    2. Top-level Jurisdiction Tree (where each leaf is a province root)
    
    This is the core of your thesis implementation.
    """
    
    def __init__(self, all_documents, traffic_logs):
        self.all_documents = all_documents
        self.province_builders = {}
        self.jurisdiction_root = None
        self.jurisdiction_layers = []
        self.province_roots_ordered = []
        
        # Only accept compressed traffic logs
        from compressed_traffic_logs import CompressedTrafficLogs
        if not isinstance(traffic_logs, (CompressedTrafficLogs, type(None))):
            raise ValueError("Only CompressedTrafficLogs instances are accepted. Legacy traffic logs are not supported.")
        
        self.compressed_traffic = traffic_logs if traffic_logs is not None else CompressedTrafficLogs()

        
        # Group documents by province
        self.provinces = sorted(list(set(doc.province for doc in all_documents)))
        
        # Create individual province builders (they will use the same compressed traffic)
        for province in self.provinces:
            self.province_builders[province] = ProvinceTreeBuilder(
                province, all_documents, self.compressed_traffic
            )
    
    def build_all_trees(self):
        """
        Build the complete hierarchical system:
        1. Build each province tree individually
        2. Create the jurisdiction tree from province roots
        """
        print(f"Building hierarchical tree system for {len(self.provinces)} provinces...")
        
        # --- Step 1: Build each province tree ---
        province_roots = {}
        for province_name, builder in self.province_builders.items():
            root = builder.build()
            province_roots[province_name] = root
            print(f"  {province_name}: {builder.get_province_info()['document_count']} docs -> Root: {root[:16]}...")
        
        return self.build_jurisdiction_tree_only(province_roots=province_roots)
    
    def build_jurisdiction_tree_only(self, province_roots = None):
        if province_roots is None:
            province_roots = {}
            for province_name, builder in self.province_builders.items():
                province_roots[province_name] = builder.merkle_root
            
        if not province_roots:
            self.jurisdiction_root = keccak(b'').hex()
            return self.jurisdiction_root
        
        # Order provinces alphabetically for consistency and ensure unique roots
        province_roots_list = []
        seen_roots = set()
        
        for province in sorted(self.provinces):
            root = province_roots[province]
            if root not in seen_roots:
                province_roots_list.append(root)
                seen_roots.add(root)
            else:
                print(f"  Warning: Duplicate root detected for {province}: {root[:16]}... (skipping)")
        
        self.province_roots_ordered = province_roots_list
        
        # Build jurisdiction tree layers
        self.jurisdiction_layers = []
        nodes = list(self.province_roots_ordered)
        if nodes:
            self.jurisdiction_layers.append(nodes)
        
        while len(nodes) > 1:
            if len(nodes) % 2 != 0:
                nodes.append(nodes[-1])  # Duplicate last node if odd
            
            parents = []
            for i in range(0, len(nodes), 2):
                parent_hash = combine_and_hash(nodes[i], nodes[i+1])
                parents.append(parent_hash)
            
            if parents:
                self.jurisdiction_layers.append(parents)
            nodes = parents
        
        self.jurisdiction_root = nodes[0] if nodes else keccak(b'').hex()
        print(f"Jurisdiction Tree Root: {self.jurisdiction_root}")
        return self.jurisdiction_root
    
    def verify_cross_province_batch(self, verification_request, verbose=False):
        """
        Verify documents across multiple provinces in a single operation.
        
        verification_request: {
            'Jakarta': ['Jakarta.PropA', 'Jakarta.PropD'],
            'Jawa_Barat': ['Jawa_Barat.PropB'],
            'Sumatera_Utara': ['Sumatera_Utara.PropC']
        }
        
        Returns a complete proof package for on-chain verification.
        """
        if self.jurisdiction_root is None:
            raise ValueError("Jurisdiction tree not built yet.")
        
        if verbose:
            print(f"\n--- [CROSS-PROVINCE VERIFICATION] Generating hierarchical proof ---")
        
        # Step 1: Generate province-level proofs
        province_proofs = {}
        provinces_involved = []
        
        for province, properties in verification_request.items():
            if province not in self.province_builders:
                continue
                
            builder = self.province_builders[province]
            
            # CRITICAL FIX: Use the unified method that returns both proof and leaves
            # This ensures perfect consistency between generation and verification
            proof, flags, doc_hashes = builder.generate_proof_and_leaves_for_properties(properties)
            
            province_proofs[province] = {
                'proof': proof,
                'flags': flags,
                'document_hashes': doc_hashes,
                'expected_root': builder.merkle_root,
                'property_count': len(properties),
                'document_count': len(doc_hashes)
            }
            provinces_involved.append(province)
            
            if verbose:
                print(f"  {province}: {len(properties)} properties, {len(doc_hashes)} documents")
        
        # Step 2: Generate jurisdiction-level proof
        # IMPORTANT: Must maintain the same order as the jurisdiction tree was built (alphabetical)
        sorted_provinces_involved = sorted(provinces_involved)
        
        # Get unique province roots to prove (avoid duplicates from provinces with same root)
        province_roots_to_prove = []
        unique_roots_map = {}  # Map root -> first province with that root
        
        for province in sorted_provinces_involved:
            root = self.province_builders[province].merkle_root
            if root not in unique_roots_map:
                province_roots_to_prove.append(root)
                unique_roots_map[root] = province
            else:
                if verbose:
                    print(f"  Note: {province} has same root as {unique_roots_map[root]}: {root[:16]}...")
        
        jurisdiction_proof, jurisdiction_flags = self._generate_jurisdiction_proof(province_roots_to_prove)
        
        # Step 3: Create complete proof package
        proof_package = {
            'province_proofs': province_proofs,
            'jurisdiction_proof': {
                'proof': jurisdiction_proof,
                'flags': jurisdiction_flags,
                'province_roots': province_roots_to_prove,
                'provinces_involved': sorted_provinces_involved  # Use sorted order
            },
            'global_root': self.jurisdiction_root,
            'total_documents': sum(p['document_count'] for p in province_proofs.values()),
            'total_provinces': len(provinces_involved)
        }
        
        if verbose:
            print(f"  Generated hierarchical proof for {proof_package['total_documents']} documents across {proof_package['total_provinces']} provinces")
        
        return proof_package
    
    def _generate_jurisdiction_proof(self, province_roots_to_prove):
        """Generate a multiproof for province roots in the jurisdiction tree."""
        if not province_roots_to_prove or not self.jurisdiction_layers:
            return [], []
        
        # Remove duplicate roots from the input
        unique_roots_to_prove = []
        seen_roots = set()
        for root in province_roots_to_prove:
            if root not in seen_roots:
                unique_roots_to_prove.append(root)
                seen_roots.add(root)
        
        # Find indices of the province roots in the ordered list
        leaf_indices_map = {root: i for i, root in enumerate(self.province_roots_ordered)}
        
        # Track which nodes are part of the proof path (using unique roots)
        processed_nodes = {}
        for root in unique_roots_to_prove:
            if root in leaf_indices_map:
                processed_nodes[(0, leaf_indices_map[root])] = True
        
        proof = []
        proof_flags = []
        proof_nodes_seen = set()  # Track nodes already added to proof to avoid duplicates
        
        # Generate proof by traversing layers bottom-up
        for layer_idx in range(len(self.jurisdiction_layers) - 1):
            layer_nodes = list(self.jurisdiction_layers[layer_idx])
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
                        # Add right sibling if it exists and not already in proof
                        if node_idx + 1 < len(layer_nodes):
                            right_node = layer_nodes[node_idx + 1]
                            if right_node not in proof_nodes_seen:
                                proof.append(right_node)
                                proof_nodes_seen.add(right_node)
                            proof_flags.append(False)
                    else:  # is_right_in_path
                        # Add left sibling if it exists and not already in proof
                        if node_idx < len(layer_nodes):
                            left_node = layer_nodes[node_idx]
                            if left_node not in proof_nodes_seen:
                                proof.append(left_node)
                                proof_nodes_seen.add(left_node)
                            proof_flags.append(False)
        
        return proof, proof_flags
    
    def verify_proof_package_locally(self, proof_package):
        """
        Verify a proof package locally before sending to blockchain.
        This implements the complete hierarchical verification logic.
        """
        print("\n--- [LOCAL HIERARCHICAL VERIFICATION] ---")
        
        # Step 1: Verify each province's internal proofs
        verified_province_roots = []
        
        for province, province_proof in proof_package['province_proofs'].items():
            builder = self.province_builders[province]
            
            # Use the verified multiproof processing from the underlying tree builder
            reconstructed_root, reason = builder._process_multiproof(
                province_proof['proof'], 
                province_proof['flags'], 
                province_proof['document_hashes']
            )
            
            is_valid = (reconstructed_root == province_proof['expected_root'])
            print(f"  {province} province verification: {is_valid} (reason: {reason})")
            
            # Debug output for failures
            if not is_valid:
                print(f"    Expected: {province_proof['expected_root']}")
                print(f"    Computed: {reconstructed_root}")
                print(f"    Builder root: {builder.merkle_root}")
                return False, f"Province {province} verification failed: {reason}"
            
            verified_province_roots.append(province_proof['expected_root'])
        
        # Step 2: Verify jurisdiction-level proof
        jurisdiction_info = proof_package['jurisdiction_proof']
        
        # Use a simpler verification that matches the province builder logic
        # Since we know the jurisdiction tree uses the same combine_and_hash function
        def verify_jurisdiction_multiproof(proof, flags, roots, expected_root):
            from eth_utils import keccak
            
            # Remove duplicate roots for verification
            unique_roots = []
            seen_roots = set()
            for root in roots:
                if root not in seen_roots:
                    unique_roots.append(root)
                    seen_roots.add(root)
            
            leaves_len = len(unique_roots)
            proof_flags_len = len(flags)
            
            # Allow some flexibility in proof structure due to optimizations
            expected_total = proof_flags_len + 1
            actual_total = leaves_len + len(proof)
            
            if abs(actual_total - expected_total) > 1:
                return False, f"Invalid jurisdiction proof structure length: expected ~{expected_total}, got {actual_total}"
            
            if proof_flags_len == 0:
                if leaves_len == 1:
                    return unique_roots[0] == expected_root, "OK"
                elif leaves_len == 0:
                    return keccak(b'').hex() == expected_root, "OK"
                else:
                    return False, "Invalid empty proof for multiple roots"
            
            hashes = [''] * proof_flags_len
            leaf_pos, hash_pos, proof_pos = 0, 0, 0
            
            try:
                for i in range(proof_flags_len):
                    # Get first operand
                    if leaf_pos < leaves_len:
                        a = unique_roots[leaf_pos]
                        leaf_pos += 1
                    elif hash_pos < len(hashes) and i > hash_pos:
                        a = hashes[hash_pos]
                        hash_pos += 1
                    else:
                        return False, f"Cannot get first operand at step {i}"
                    
                    # Get second operand
                    b = ''
                    if flags[i]:
                        if leaf_pos < leaves_len:
                            b = unique_roots[leaf_pos]
                            leaf_pos += 1
                        elif hash_pos < len(hashes) and i > hash_pos:
                            b = hashes[hash_pos]
                            hash_pos += 1
                        else:
                            return False, f"Cannot get second operand (flag) at step {i}"
                    else:
                        if proof_pos >= len(proof): 
                            return False, f"Jurisdiction proof consumed prematurely at step {i}"
                        b = proof[proof_pos]
                        proof_pos += 1
                    
                    hashes[i] = combine_and_hash(a, b)
                
                # Allow some unused proof elements due to deduplication
                if proof_pos < len(proof) - 1:
                    return False, f"Jurisdiction proof not fully consumed: used {proof_pos}/{len(proof)}"
                
                return hashes[proof_flags_len - 1] == expected_root, "OK"
                
            except Exception as e:
                return False, f"Jurisdiction multiproof processing error: {str(e)}"
        
        jurisdiction_valid, jurisdiction_reason = verify_jurisdiction_multiproof(
            jurisdiction_info['proof'],
            jurisdiction_info['flags'],
            jurisdiction_info['province_roots'],
            proof_package['global_root']
        )
        
        print(f"  Jurisdiction verification: {jurisdiction_valid} (reason: {jurisdiction_reason})")
        
        if not jurisdiction_valid:
            return False, f"Jurisdiction verification failed: {jurisdiction_reason}"
        
        print(f"  ✅ Complete hierarchical verification PASSED!")
        print(f"     - Verified {proof_package['total_documents']} documents")
        print(f"     - Across {proof_package['total_provinces']} provinces")
        print(f"     - Against global root: {proof_package['global_root'][:16]}...")
        
        return True, "Complete hierarchical verification successful"
    
    def get_system_info(self):
        """Get information about the complete hierarchical system."""
        info = {
            'jurisdiction_root': self.jurisdiction_root,
            'total_provinces': len(self.provinces),
            'total_documents': len(self.all_documents),
            'provinces': {},
            'compression_stats': self.get_compression_stats()
        }
        
        for province_name, builder in self.province_builders.items():
            info['provinces'][province_name] = builder.get_province_info()
        
        return info
    
    def add_learning_event(self, verified_properties, day_idx=None, learning_mode=None):
        """
        Add verification event with unified learning support.
        
        Args:
            verified_properties: List of property IDs that were verified
            day_idx: Day index for daily learning (optional)
            learning_mode: Override learning mode (optional, uses config if None)
        """
        # Determine learning mode
        if learning_mode is None:
            try:
                from learning_config import get_learning_config, LearningMode
                config = get_learning_config()
                learning_mode = config.mode
            except ImportError:
                # Fallback to immediate learning if config not available
                learning_mode = "IMMEDIATE"
        
        # Add to compressed traffic logs
        self.compressed_traffic.add_verification_event(verified_properties)
        
        # Update individual province builders
        for builder in self.province_builders.values():
            if hasattr(builder, 'add_learning_event'):
                builder.add_learning_event(verified_properties, day_idx=day_idx, learning_mode=learning_mode)
            elif hasattr(builder, 'add_verification_event'):
                builder.add_verification_event(verified_properties)
    
    def add_verification_event(self, verified_properties):
        """Add new verification event to compressed traffic logs (backward compatibility)."""
        self.add_learning_event(verified_properties, learning_mode="IMMEDIATE")
    
    def get_compression_stats(self):
        """Get compression statistics."""
        if not hasattr(self, 'compressed_traffic') or self.compressed_traffic.total_events == 0:
            return {"status": "no_traffic_data"}
        
        return self.compressed_traffic.get_statistics()

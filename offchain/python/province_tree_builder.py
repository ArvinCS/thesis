from optimized_tree_builder import HierarchicalTreeBuilder

class ProvinceTreeBuilder:
    """
    Builds an optimized Merkle tree for a single province using Pairs-First Huffman optimization.
    This represents one "Property Tree" in the hierarchical system.
    
    This class leverages the existing HierarchicalTreeBuilder for all the heavy lifting.
    """
    
    def __init__(self, province_name, documents, traffic_logs):
        self.province_name = province_name
        self.documents = [doc for doc in documents if doc.province == province_name]
        
        # Only accept compressed traffic logs
        from compressed_traffic_logs import CompressedTrafficLogs
        if not isinstance(traffic_logs, (CompressedTrafficLogs, type(None))):
            raise ValueError("Only CompressedTrafficLogs instances are accepted. Legacy traffic logs are not supported.")
        
        if isinstance(traffic_logs, CompressedTrafficLogs):
            # Filter compressed traffic logs to only include events relevant to this province
            self.filtered_compressed_traffic = self._filter_compressed_traffic_logs(traffic_logs)
        else:
            # Start with empty compressed traffic logs
            self.filtered_compressed_traffic = CompressedTrafficLogs()
        
        # Create the underlying HierarchicalTreeBuilder with province-specific compressed data
        self.tree_builder = HierarchicalTreeBuilder(self.documents, self.filtered_compressed_traffic)
    
    def _filter_compressed_traffic_logs(self, compressed_traffic):
        """Filter compressed traffic logs to only include events relevant to this province."""
        from compressed_traffic_logs import CompressedTrafficLogs
        
        province_compressed = CompressedTrafficLogs()
        province_prefix = f"{self.province_name}."
        
        # Filter property frequencies
        for prop_id, freq in compressed_traffic.property_frequencies.items():
            if prop_id.startswith(province_prefix):
                province_compressed.property_frequencies[prop_id] = freq
        
        # Filter pair frequencies  
        for (prop1, prop2), freq in compressed_traffic.pair_frequencies.items():
            # Include pairs where both properties are from this province
            if prop1.startswith(province_prefix) and prop2.startswith(province_prefix):
                province_compressed.pair_frequencies[(prop1, prop2)] = freq
        
        # Update other statistics
        province_compressed.total_events = compressed_traffic.total_events
        # Note: This is an approximation - in practice, you might want more precise filtering
        
        return province_compressed
    
    def build(self):
        """
        Build the province tree using the existing HierarchicalTreeBuilder.
        Returns the Merkle root for this province.
        """
        return self.tree_builder.build()
    
    def generate_proof_for_properties(self, property_full_ids):
        """
        Generate a multiproof for specific properties within this province.
        Returns (proof, proof_flags) for the given properties.
        """
        # Collect all leaf hashes for the requested properties
        leaves_to_prove_hex = []
        for prop_id in property_full_ids:
            # Extract the property part from the full ID (Province.Property)
            property_part = prop_id.split('.')[-1] if '.' in prop_id else prop_id
            
            # Try both full ID and property part for compatibility
            if prop_id in self.tree_builder.property_clusters:
                leaves_to_prove_hex.extend(self.tree_builder.property_clusters[prop_id].get_leaf_hashes_hex())
            elif property_part in self.tree_builder.property_clusters:
                leaves_to_prove_hex.extend(self.tree_builder.property_clusters[property_part].get_leaf_hashes_hex())
        
        if not leaves_to_prove_hex:
            return [], []
        
        # CRITICAL FIX: Sort by tree order and remove duplicates
        # Multiple properties may share the same documents, causing duplicate leaf hashes
        # This breaks the multiproof algorithm, so we need to deduplicate while maintaining order
        tree_leaves = self.tree_builder.ordered_leaves_hex
        leaves_with_indices = []
        seen_leaves = set()
        
        # Deduplicate and collect leaves with their tree indices
        for leaf in leaves_to_prove_hex:
            if leaf in tree_leaves and leaf not in seen_leaves:
                tree_index = tree_leaves.index(leaf)
                leaves_with_indices.append((tree_index, leaf))
                seen_leaves.add(leaf)
        
        # Sort by tree index and extract the leaves
        leaves_with_indices.sort(key=lambda x: x[0])
        leaves_to_prove_hex_sorted = [leaf for _, leaf in leaves_with_indices]
        
        # Use the existing proof generation from HierarchicalTreeBuilder
        return self.tree_builder.generate_batched_proof_with_flags(leaves_to_prove_hex_sorted)
    
    def get_province_info(self):
        """Return information about this province tree."""
        return {
            'province_name': self.province_name,
            'document_count': len(self.documents),
            'property_count': len(self.tree_builder.property_clusters),
            'merkle_root': self.tree_builder.merkle_root,
            'leaf_count': len(self.tree_builder.ordered_leaves_hex)
        }
    
    @property
    def merkle_root(self):
        """Get the Merkle root from the underlying tree builder."""
        return self.tree_builder.merkle_root
    
    @property
    def property_clusters(self):
        """Get property clusters from the underlying tree builder."""
        return self.tree_builder.property_clusters
    
    def _process_multiproof(self, proof, proof_flags, leaves):
        """Delegate multiproof processing to the underlying tree builder."""
        return self.tree_builder._process_multiproof(proof, proof_flags, leaves)
    
    def generate_proof_and_leaves_for_properties(self, property_full_ids):
        """
        Generate proof and return the exact leaves used, ensuring perfect consistency.
        Returns (proof, flags, leaves) where leaves are exactly what was used for proof generation.
        """
        # Use the same logic as generate_proof_for_properties but also return the leaves
        leaves_to_prove_hex = []
        for prop_id in property_full_ids:
            # Extract the property part from the full ID (Province.Property)
            property_part = prop_id.split('.')[-1] if '.' in prop_id else prop_id
            
            # Try both full ID and property part for compatibility
            if prop_id in self.tree_builder.property_clusters:
                leaves_to_prove_hex.extend(self.tree_builder.property_clusters[prop_id].get_leaf_hashes_hex())
            elif property_part in self.tree_builder.property_clusters:
                leaves_to_prove_hex.extend(self.tree_builder.property_clusters[property_part].get_leaf_hashes_hex())
            else:
                print(f"Warning: Property {prop_id} not found in {self.province_name} builder")
                print(f"Available properties: {list(self.tree_builder.property_clusters.keys())[:5]}...")
        
        # CRITICAL FIX: Sort by tree order and remove duplicates (same as fixed version)
        # Multiple properties may share the same documents, causing duplicate hashes
        tree_leaves = self.tree_builder.ordered_leaves_hex
        leaves_with_indices = []
        for leaf in leaves_to_prove_hex:
            if leaf in tree_leaves:
                tree_index = tree_leaves.index(leaf)
                leaves_with_indices.append((tree_index, leaf))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_leaves_with_indices = []
        for item in leaves_with_indices:
            if item not in seen:
                seen.add(item)
                unique_leaves_with_indices.append(item)
        
        # Sort by tree index and extract the leaves
        unique_leaves_with_indices.sort(key=lambda x: x[0])
        leaves_to_prove_hex_sorted = [leaf for _, leaf in unique_leaves_with_indices]
        
        if not leaves_to_prove_hex_sorted:
            return [], [], []
        
        # Generate the proof using the deduplicated and sorted leaves
        proof, flags = self.tree_builder.generate_batched_proof_with_flags(leaves_to_prove_hex_sorted)
        
        return proof, flags, leaves_to_prove_hex_sorted
    
    def add_learning_event(self, verified_properties, day_idx=None, learning_mode=None):
        """
        Add verification event with unified learning support (delegates to underlying tree builder).
        
        Args:
            verified_properties: List of property IDs that were verified
            day_idx: Day index for daily learning (optional)
            learning_mode: Override learning mode (optional, uses config if None)
        """
        # Filter properties to only include ones from this province
        province_prefix = f"{self.province_name}."
        province_properties = [prop for prop in verified_properties if prop.startswith(province_prefix)]
        
        if province_properties:
            return self.tree_builder.add_learning_event(province_properties, learning_mode=learning_mode)
        return False, None

    def add_verification_event(self, verified_properties, learning_mode=None):
        """Add new verification event (backward compatibility)."""
        return self.add_learning_event(verified_properties, learning_mode)
    
    def get_compression_stats(self):
        """Get compression statistics from underlying tree builder."""
        if hasattr(self.tree_builder, 'get_compression_stats'):
            return self.tree_builder.get_compression_stats()
        return {"status": "no_compression_support"}

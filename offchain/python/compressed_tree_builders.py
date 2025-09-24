#!/usr/bin/env python3
"""
Updated Tree Builders with Compressed Traffic Logs

Integrate compressed traffic logs into existing tree builders for better memory efficiency.
"""

from compressed_traffic_logs import CompressedTrafficLogs, convert_traditional_to_compressed
from traditional_multiproof_with_huffman_builder import TraditionalMultiproofWithHuffmanBuilder
from clustered_flat_tree_builder import ClusteredFlatTreeBuilder
from optimized_tree_builder import Document
import time

class CompressedTraditionalHuffmanBuilder(TraditionalMultiproofWithHuffmanBuilder):
    """
    Traditional Huffman builder using compressed traffic logs.
    """
    
    def __init__(self, all_documents, traffic_logs=None):
        # Initialize parent class first
        super().__init__(all_documents, [])  # Empty traffic logs initially
        
        # Only accept compressed traffic logs
        if not isinstance(traffic_logs, (CompressedTrafficLogs, type(None))):
            raise ValueError("Only CompressedTrafficLogs instances are accepted. Legacy traffic logs are not supported.")
        
        self.compressed_traffic = traffic_logs if traffic_logs is not None else CompressedTrafficLogs()
    
    def _calculate_frequencies(self):
        """Override to use compressed traffic logs."""
        # Use pre-calculated frequencies from compressed logs
        return (
            dict(self.compressed_traffic.property_frequencies),
            dict(self.compressed_traffic.pair_frequencies)
        )
    
    def add_daily_verification_compressed(self, properties, day_idx=None):
        """Add verification using compressed format."""
        self.compressed_traffic.add_verification_event(properties)
        
        # Also update the parent's tracking for compatibility
        self.verification_count += 1
        if hasattr(self, 'daily_traffic_logs'):
            self.daily_traffic_logs.append(properties)
    
    def get_compression_stats(self):
        """Get compression statistics."""
        # Convert to traditional format for comparison
        traditional_format = self.compressed_traffic.expand_to_traditional_format()
        memory_stats = self.compressed_traffic.memory_comparison(traditional_format)
        traffic_stats = self.compressed_traffic.get_statistics()
        
        return {
            'memory': memory_stats,
            'traffic': traffic_stats
        }


class CompressedClusteredFlatBuilder(ClusteredFlatTreeBuilder):
    """
    Clustered flat builder using compressed traffic logs.
    """
    
    def __init__(self, all_documents, traffic_logs=None):
        # Store documents first
        self.all_documents = all_documents
        
        # Convert and store compressed traffic logs
        if traffic_logs:
            if isinstance(traffic_logs, CompressedTrafficLogs):
                self.compressed_traffic = traffic_logs
            else:
                self.compressed_traffic = convert_traditional_to_compressed(traffic_logs)
        else:
            self.compressed_traffic = CompressedTrafficLogs()
        
        # Initialize other attributes
        self.province_clusters = self._create_province_clusters()
        self.ordered_leaves_hex = []
        self.merkle_root = None
        self.tree_layers = []
    
    def _calculate_frequencies(self):
        """Override to use compressed traffic logs."""
        # Use pre-calculated frequencies from compressed logs
        return (
            dict(self.compressed_traffic.property_frequencies),
            dict(self.compressed_traffic.pair_frequencies)
        )
    
    def get_compression_stats(self):
        """Get compression statistics."""
        # Convert to traditional format for comparison
        traditional_format = self.compressed_traffic.expand_to_traditional_format()
        memory_stats = self.compressed_traffic.memory_comparison(traditional_format)
        traffic_stats = self.compressed_traffic.get_statistics()
        
        return {
            'memory': memory_stats,
            'traffic': traffic_stats
        }


def benchmark_compression_performance():
    """Benchmark the performance improvement from compression."""
    print("=== Compression Performance Benchmark ===\n")
    
    # Create test documents
    documents = []
    provinces = ["PROV_A", "PROV_B", "PROV_C", "PROV_D"]
    properties = [f"PROP_{i}" for i in range(1, 21)]  # 20 properties per province
    
    doc_id = 1
    for prov in provinces:
        for prop in properties:
            doc = Document(
                doc_id=f"DOC_{doc_id:04d}",
                content=f"Content_{doc_id}",
                property_id=prop,
                province=prov
            )
            documents.append(doc)
            doc_id += 1
    
    print(f"Created {len(documents)} test documents")
    
    # Create traffic logs with many duplicates (realistic scenario)
    import random
    random.seed(42)
    
    traffic_logs = []
    
    # Popular pairs (appear frequently)
    popular_pairs = [
        ["PROV_A.PROP_1", "PROV_A.PROP_2"],
        ["PROV_B.PROP_3", "PROV_B.PROP_4"],
        ["PROV_A.PROP_1", "PROV_B.PROP_1"],
    ]
    
    # Generate 1000 events with lots of duplicates
    for _ in range(1000):
        if random.random() < 0.6:  # 60% chance of popular pattern
            traffic_logs.append(random.choice(popular_pairs).copy())
        else:  # 40% chance of random pattern
            num_props = random.choice([1, 2, 3])
            selected_docs = random.sample(documents, num_props)
            traffic_logs.append([doc.full_id for doc in selected_docs])
    
    print(f"Generated {len(traffic_logs)} traffic events")
    
    # Benchmark traditional approach
    print("\n1. Traditional Approach:")
    start_time = time.time()
    
    traditional_builder = TraditionalMultiproofWithHuffmanBuilder(documents, traffic_logs)
    traditional_builder.build()
    
    traditional_time = time.time() - start_time
    print(f"   Build time: {traditional_time:.3f} seconds")
    
    # Benchmark compressed approach
    print("\n2. Compressed Approach:")
    start_time = time.time()
    
    compressed_builder = CompressedTraditionalHuffmanBuilder(documents, traffic_logs)
    compressed_builder.build()
    
    compressed_time = time.time() - start_time
    print(f"   Build time: {compressed_time:.3f} seconds")
    
    # Get compression statistics
    compression_stats = compressed_builder.get_compression_stats()
    
    print(f"\n3. Compression Results:")
    print(f"   Memory saved: {compression_stats['memory']['space_saved_percent']:.1f}%")
    print(f"   Compression ratio: {compression_stats['memory']['compression_ratio']:.2f}x")
    print(f"   Unique properties: {compression_stats['traffic']['unique_properties']}")
    print(f"   Unique pairs: {compression_stats['traffic']['unique_pairs']}")
    print(f"   Total events: {compression_stats['traffic']['total_events']}")
    
    # Performance comparison
    speedup = traditional_time / max(compressed_time, 0.001)  # Avoid division by zero
    print(f"\n4. Performance:")
    print(f"   Traditional: {traditional_time:.3f}s")
    print(f"   Compressed:  {compressed_time:.3f}s")
    print(f"   Speedup: {speedup:.2f}x")
    
    # Verify both produce same results
    print(f"\n5. Verification:")
    print(f"   Same root hash: {traditional_builder.merkle_root == compressed_builder.merkle_root}")
    print(f"   Same leaf count: {len(traditional_builder.ordered_leaves_hex) == len(compressed_builder.ordered_leaves_hex)}")
    
    # Test multiproof generation
    test_leaves = traditional_builder.ordered_leaves_hex[:5]
    
    trad_proof, trad_flags = traditional_builder.generate_batched_proof_with_flags(test_leaves)
    comp_proof, comp_flags = compressed_builder.generate_batched_proof_with_flags(test_leaves)
    
    print(f"   Same proof size: {len(trad_proof) == len(comp_proof) and len(trad_flags) == len(comp_flags)}")


if __name__ == "__main__":
    benchmark_compression_performance()

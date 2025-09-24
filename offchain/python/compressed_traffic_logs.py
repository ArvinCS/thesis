#!/usr/bin/env python3
"""
Compressed Traffic Logs Implementation

Instead of storing duplicate traffic patterns, store frequency counters.
This reduces memory usage and improves performance for frequency-based optimizations.
"""

from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import itertools

class CompressedTrafficLogs:
    """
    Stores traffic logs as frequency counters instead of duplicate entries.
    """
    
    def __init__(self):
        # Store pair frequencies: {(prop1, prop2): count}
        self.pair_frequencies = defaultdict(int)
        # Store single property frequencies: {prop: count}
        self.property_frequencies = defaultdict(int)
        # Store group sizes for analysis: {size: count}
        self.group_size_frequencies = defaultdict(int)
        # Total number of verification events
        self.total_events = 0
        
    def add_verification_event(self, properties: List[str]):
        """Add a verification event (list of properties verified together)."""
        if not properties:
            return
            
        # Normalize properties (remove duplicates, sort for consistency)
        unique_props = sorted(set(properties))
        
        self.total_events += 1
        self.group_size_frequencies[len(unique_props)] += 1
        
        # Update property frequencies
        for prop in unique_props:
            self.property_frequencies[prop] += 1
            
        # Update pair frequencies (for groups of 2 or more)
        if len(unique_props) >= 2:
            for prop1, prop2 in itertools.combinations(unique_props, 2):
                # Ensure consistent ordering (smaller first)
                pair = (min(prop1, prop2), max(prop1, prop2))
                self.pair_frequencies[pair] += 1
    
    def add_multiple_events(self, traffic_logs: List[List[str]]):
        """Add multiple verification events from traditional format."""
        for event in traffic_logs:
            self.add_verification_event(event)
    
    def get_pair_frequency(self, prop1: str, prop2: str) -> int:
        """Get frequency of a specific pair."""
        pair = (min(prop1, prop2), max(prop1, prop2))
        return self.pair_frequencies.get(pair, 0)
    
    def get_property_frequency(self, prop: str) -> int:
        """Get frequency of a specific property."""
        return self.property_frequencies.get(prop, 0)
    
    def get_most_frequent_pairs(self, top_k: int = 10) -> List[Tuple[Tuple[str, str], int]]:
        """Get top-k most frequent pairs."""
        return sorted(self.pair_frequencies.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    def get_most_frequent_properties(self, top_k: int = 10) -> List[Tuple[str, int]]:
        """Get top-k most frequent properties."""
        return sorted(self.property_frequencies.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    def expand_to_traditional_format(self) -> List[List[str]]:
        """
        Expand back to traditional format for compatibility.
        Note: This recreates the original structure but may not preserve exact order.
        """
        traditional_logs = []
        
        # Add single property events
        for prop, freq in self.property_frequencies.items():
            # Check if this property appears in any pairs
            appears_in_pairs = any(prop in pair for pair in self.pair_frequencies.keys())
            if not appears_in_pairs:
                # Add as single events
                for _ in range(freq):
                    traditional_logs.append([prop])
        
        # Add pair events
        for (prop1, prop2), freq in self.pair_frequencies.items():
            for _ in range(freq):
                traditional_logs.append([prop1, prop2])
                
        return traditional_logs
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the traffic logs."""
        return {
            'total_events': self.total_events,
            'unique_properties': len(self.property_frequencies),
            'unique_pairs': len(self.pair_frequencies),
            'average_group_size': sum(size * count for size, count in self.group_size_frequencies.items()) / max(1, self.total_events),
            'group_size_distribution': dict(self.group_size_frequencies),
            'most_active_property': max(self.property_frequencies.items(), key=lambda x: x[1]) if self.property_frequencies else None,
            'most_frequent_pair': max(self.pair_frequencies.items(), key=lambda x: x[1]) if self.pair_frequencies else None
        }
    
    def merge(self, other: 'CompressedTrafficLogs'):
        """Merge with another compressed traffic log."""
        for pair, freq in other.pair_frequencies.items():
            self.pair_frequencies[pair] += freq
            
        for prop, freq in other.property_frequencies.items():
            self.property_frequencies[prop] += freq
            
        for size, count in other.group_size_frequencies.items():
            self.group_size_frequencies[size] += count
            
        self.total_events += other.total_events
    
    def memory_comparison(self, traditional_logs: List[List[str]]) -> Dict:
        """Compare memory usage with traditional format."""
        import sys
        
        # Estimate traditional format size
        traditional_size = sys.getsizeof(traditional_logs)
        for event in traditional_logs:
            traditional_size += sys.getsizeof(event)
            for prop in event:
                traditional_size += sys.getsizeof(prop)
        
        # Estimate compressed format size
        compressed_size = (
            sys.getsizeof(self.pair_frequencies) +
            sys.getsizeof(self.property_frequencies) +
            sys.getsizeof(self.group_size_frequencies) +
            sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in self.pair_frequencies.items()) +
            sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in self.property_frequencies.items())
        )
        
        return {
            'traditional_bytes': traditional_size,
            'compressed_bytes': compressed_size,
            'compression_ratio': traditional_size / max(1, compressed_size),
            'space_saved_bytes': traditional_size - compressed_size,
            'space_saved_percent': ((traditional_size - compressed_size) / max(1, traditional_size)) * 100
        }


def convert_traditional_to_compressed(traffic_logs: List[List[str]]) -> CompressedTrafficLogs:
    """Convert traditional traffic logs to compressed format."""
    compressed = CompressedTrafficLogs()
    compressed.add_multiple_events(traffic_logs)
    return compressed


def demo_compression():
    """Demonstrate the compression effectiveness."""
    print("=== Traffic Log Compression Demo ===\n")
    
    # Create sample traditional traffic logs with many duplicates
    traditional_logs = [
        ["PROV_A.PROP_1", "PROV_A.PROP_2"],  # Repeated pattern
        ["PROV_A.PROP_1", "PROV_A.PROP_2"],
        ["PROV_A.PROP_1", "PROV_A.PROP_2"],
        ["PROV_B.PROP_1", "PROV_B.PROP_2"],  # Another repeated pattern
        ["PROV_B.PROP_1", "PROV_B.PROP_2"],
        ["PROV_A.PROP_1"],                    # Single property
        ["PROV_A.PROP_1"],
        ["PROV_A.PROP_1", "PROV_B.PROP_1", "PROV_C.PROP_1"],  # Triple
        ["PROV_A.PROP_1", "PROV_B.PROP_1", "PROV_C.PROP_1"],
    ]
    
    print(f"Traditional format: {len(traditional_logs)} events")
    for i, event in enumerate(traditional_logs):
        print(f"  {i+1}: {event}")
    
    # Convert to compressed format
    compressed = convert_traditional_to_compressed(traditional_logs)
    
    print(f"\nCompressed format:")
    print(f"  Total events: {compressed.total_events}")
    print(f"  Unique properties: {len(compressed.property_frequencies)}")
    print(f"  Unique pairs: {len(compressed.pair_frequencies)}")
    
    print(f"\nProperty frequencies:")
    for prop, freq in sorted(compressed.property_frequencies.items()):
        print(f"  {prop}: {freq}")
    
    print(f"\nPair frequencies:")
    for pair, freq in sorted(compressed.pair_frequencies.items()):
        print(f"  {pair[0]} + {pair[1]}: {freq}")
    
    # Memory comparison
    memory_stats = compressed.memory_comparison(traditional_logs)
    print(f"\nMemory comparison:")
    print(f"  Traditional: {memory_stats['traditional_bytes']} bytes")
    print(f"  Compressed: {memory_stats['compressed_bytes']} bytes")
    print(f"  Compression ratio: {memory_stats['compression_ratio']:.2f}x")
    print(f"  Space saved: {memory_stats['space_saved_percent']:.1f}%")
    
    # Statistics
    stats = compressed.get_statistics()
    print(f"\nStatistics:")
    print(f"  Average group size: {stats['average_group_size']:.2f}")
    print(f"  Group size distribution: {stats['group_size_distribution']}")
    print(f"  Most active property: {stats['most_active_property']}")
    print(f"  Most frequent pair: {stats['most_frequent_pair']}")


if __name__ == "__main__":
    demo_compression()

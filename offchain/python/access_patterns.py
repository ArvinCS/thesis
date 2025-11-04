# In access_patterns.py

from itertools import combinations
import random
from collections import Counter
import numpy as np

class TransactionalPattern:
    """
    Defines and generates an assumed access pattern for transactional users
    based on a weighted map of document importance.
    """
    def __init__(self, document_importance_map: dict[str, int], alpha_threshold: float = 0.3):
        """
        Args:
            document_importance_map: A dict mapping doc_types to an integer
                                     representing their access frequency/importance.
        """
        self.document_importance_map = document_importance_map
        self.alpha_threshold = alpha_threshold
    
    def get_document_frequencies(self, documents: list) -> dict:
        """
        Generates a frequency map for a given list of documents based on
        the predefined importance weights. This is the "traffic log".
        """
        frequencies = {}
        for doc in documents:
            # Look up the base importance/frequency for the document's type.
            # Default to a low value (1) if the type is not in our map.
            base_frequency = self.document_importance_map.get(doc.doc_type, 1)
            frequencies[doc] = base_frequency  # Use document object as key
        return frequencies
    
    def get_document_pair_frequencies(self, documents: list, num_simulated_queries: int = 100) -> Counter:
        """
        Generates a PAIR co-occurrence frequency map (the "traffic log").
        This is the direct input for the Pairs-first Huffman algorithm.

        It works by simulating a number of queries based on document importance
        and counting which pairs appear together most often.
        """
        if len(documents) < 2:
            return Counter()

        pair_counter = Counter()
        
        # Get the individual doc weights to drive the simulation
        doc_frequencies = self.get_document_frequencies(documents)
        doc_ids = list(doc_frequencies.keys())
        weights = list(doc_frequencies.values())

        for _ in range(num_simulated_queries):
            # For each simulated query, pick a random number of documents to access
            num_docs_to_query = random.randint(2, len(doc_ids))
            
            # Select a subset of documents based on their importance weights
            queried_docs = random.choices(doc_ids, weights=weights, k=num_docs_to_query)
            # Ensure we only count unique pairs (no need to sort, just get unique)
            unique_queried_docs = list(set(queried_docs))

            # If the query has at least 2 unique docs, find all pairs and count them
            if len(unique_queried_docs) >= 2:
                for pair in combinations(unique_queried_docs, 2):
                    # Sort the pair by a consistent attribute (like doc_id) for consistent ordering
                    sorted_pair = tuple(sorted(pair, key=lambda x: f"{x.province}.{x.property_id}.{x.doc_id}"))
                    pair_counter[sorted_pair] += 1
        
        pruned_pair_counter = Counter()
        for pair, count in pair_counter.items():
            if count >= self.alpha_threshold * pair_counter.total():
                pruned_pair_counter[pair] = count

        return pruned_pair_counter

# --- ADD THE NEW AUDIT PATTERN CLASS ---
class AuditPattern:
    """
    Defines an assumed access pattern for audit users with flexible province and document selection.

    This class models audit scenarios that can span:
    - Single or multiple provinces
    - Single or multiple document types
    - Selective document sampling from properties (not necessarily all documents)
    """
    def __init__(self, province_weights: dict[str, float], 
                 avg_sample_size: int = 30, min_sample_size: int = 5,
                 min_provinces_per_audit: int = 1, max_provinces_per_audit: int = 3,
                 avg_docs_per_property: int = 2, min_docs_per_property: int = 1):
        """
        Args:
            province_weights: Dict mapping province names to their selection weights
            avg_sample_size: Average number of properties to sample per province (λ for Poisson)
            min_sample_size: Minimum number of properties to ensure at least some sampling
            min_provinces_per_audit: Minimum number of provinces to include in an audit
            max_provinces_per_audit: Maximum number of provinces to include in an audit
            avg_docs_per_property: Average documents to select per property (λ for Poisson)
            min_docs_per_property: Minimum documents to ensure at least some selection
        """
        if not province_weights:
            raise ValueError("Province weights dictionary cannot be empty.")

        self.provinces = list(province_weights.keys())
        self.weights = list(province_weights.values())
        self.avg_sample_size = avg_sample_size
        self.min_sample_size = min_sample_size
        self.min_provinces_per_audit = min_provinces_per_audit
        self.max_provinces_per_audit = max_provinces_per_audit
        self.avg_docs_per_property = avg_docs_per_property
        self.min_docs_per_property = min_docs_per_property

    def get_random_audit_region(self) -> str:
        """Selects a single, random province using a weighted distribution."""
        return random.choices(self.provinces, weights=self.weights, k=1)[0]
    
    def get_cross_province_sample(self, properties_by_province: dict, num_provinces_to_sample: int = 5) -> list:
        """
        Selects a few provinces and takes a smaller sample of properties from each,
        simulating a national, cross-region audit stress test.
        """
        if len(self.provinces) < num_provinces_to_sample:
            selected_provinces = self.provinces
        else:
            # Select distinct provinces randomly (unweighted for a chaotic test)
            selected_provinces = random.choices(self.provinces, weights=self.weights, k=num_provinces_to_sample)

        final_sampled_properties = []
        # Take a smaller sample from each selected province to keep the query size reasonable
        small_sample_size = max(1, self.get_audit_sample_size() // num_provinces_to_sample)
        
        for province in selected_provinces:
            props_in_province = properties_by_province.get(province, [])
            if not props_in_province:
                continue
            
            if len(props_in_province) > small_sample_size:
                final_sampled_properties.extend(random.sample(props_in_province, small_sample_size))
            else:
                final_sampled_properties.extend(props_in_province)
        
        return final_sampled_properties
    
    def get_random_audit_regions(self) -> list[str]:
        """Selects one or more provinces to be the targets of an audit."""
        num_provinces = random.randint(self.min_provinces_per_audit, 
                                     min(self.max_provinces_per_audit, len(self.provinces)))
        
        # Use weighted selection without replacement
        selected_provinces = []
        available_provinces = self.provinces.copy()
        available_weights = self.weights.copy()
        
        for _ in range(num_provinces):
            if not available_provinces:
                break
            
            selected = random.choices(available_provinces, weights=available_weights, k=1)[0]
            selected_provinces.append(selected)
            
            # Remove selected province to avoid duplicates
            idx = available_provinces.index(selected)
            available_provinces.pop(idx)
            available_weights.pop(idx)
        
        return selected_provinces[:1]
    
    def get_audit_sample_size(self) -> int:
        """Determines a realistic sample size using Poisson distribution centered around average."""
        # Generate from Poisson distribution with lambda = avg_sample_size
        sample_size = np.random.poisson(self.avg_sample_size)
        
        # Ensure we don't go below minimum sample size for practical auditing
        return max(sample_size, self.min_sample_size)
    
    def get_docs_per_property_count(self) -> int:
        """Determines document count per property using Poisson distribution for realism."""
        # Generate from Poisson distribution with lambda = avg_docs_per_property
        docs_count = np.random.poisson(self.avg_docs_per_property)
        
        # Ensure we select at least the minimum number of documents
        return max(docs_count, self.min_docs_per_property)
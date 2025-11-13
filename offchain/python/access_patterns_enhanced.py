# Enhanced access_patterns.py with Zipfian distribution support

from itertools import combinations
import random
from collections import Counter
import numpy as np
import scipy.stats as stats

# Fixed seeds for reproducible access patterns
PATTERN_SEED = 42

class TransactionalPattern:
    """
    Defines and generates an assumed access pattern for transactional users
    based on a weighted map of document importance.
    """
    def __init__(self, document_importance_map: dict[str, int], alpha_threshold: float = 0.3, 
                 use_zipfian: bool = True, zipf_parameter: float = 1.2,
                 use_property_zipfian: bool = True, property_zipf_parameter: float = 1.1,
                 random_seed: int = PATTERN_SEED):
        """
        Args:
            document_importance_map: A dict mapping doc_types to an integer
                                     representing their access frequency/importance.
            alpha_threshold: Minimum frequency threshold for pair inclusion
            use_zipfian: Whether to use Zipfian distribution for realistic access patterns
            zipf_parameter: Zipfian distribution parameter (1.0-2.0, higher = more skewed)
            use_property_zipfian: Whether to use Zipfian distribution for property selection
            property_zipf_parameter: Zipfian parameter for property-level distribution (typically lower than document-level)
        """
        self.document_importance_map = document_importance_map
        self.alpha_threshold = alpha_threshold
        self.use_zipfian = use_zipfian
        self.zipf_parameter = zipf_parameter
        self.use_property_zipfian = use_property_zipfian
        self.property_zipf_parameter = property_zipf_parameter
        self.random_seed = random_seed
    
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
    
    def get_province_zipfian_weights(self, properties: list) -> dict:
        """
        Stage 1: Generate Zipfian-distributed weights for province selection.
        Creates outer Zipfian distribution across provinces based on economic importance.
        
        Returns dict mapping province -> (zipfian_weight, properties_in_province)
        """
        if not properties:
            return {}
        
        # Group properties by province
        province_groups = {}
        for prop in properties:
            if prop.province not in province_groups:
                province_groups[prop.province] = []
            province_groups[prop.province].append(prop)
        
        if not self.use_property_zipfian:
            # Uniform distribution (legacy behavior)
            uniform_weight = 1.0 / len(province_groups)
            return {province: (uniform_weight, props) for province, props in province_groups.items()}
        
        # Calculate provincial importance scores
        province_scores = []
        for province, props in province_groups.items():
            # Provincial score = location bonus + total document portfolio value
            location_bonus = self._get_location_importance_bonus(province)
            portfolio_score = sum(
                sum(self.document_importance_map.get(doc.doc_type, 1) for doc in prop.documents)
                for prop in props
            )
            total_score = location_bonus * len(props) + portfolio_score  # Scale location by property count
            province_scores.append((province, total_score, props))
        
        # Sort provinces by importance (descending) for Zipfian ranking
        sorted_provinces = sorted(province_scores, key=lambda x: x[1], reverse=True)
        
        # Generate Zipfian probabilities for provinces (Stage 1 - Outer Zipfian)
        province_weights = {}
        for rank, (province, score, props) in enumerate(sorted_provinces, 1):
            zipf_prob = 1.0 / (rank ** self.property_zipf_parameter)  # Use same parameter for consistency
            province_weights[province] = (zipf_prob, props)
        
        # Normalize provincial probabilities
        total_weight = sum(weight for weight, _ in province_weights.values())
        normalized_weights = {
            province: (weight / total_weight, props) 
            for province, (weight, props) in province_weights.items()
        }
        
        return normalized_weights

    def get_within_province_zipfian_weights(self, properties_in_province: list) -> list:
        """
        Stage 2: Generate Zipfian-distributed weights for property selection within a province.
        Creates inner Zipfian distribution for properties within the selected province.
        
        Returns list of weights corresponding to properties_in_province list.
        """
        if not properties_in_province:
            return []
        
        if len(properties_in_province) == 1:
            return [1.0]  # Only one property, no need for distribution
        
        # Calculate property importance scores within province
        property_scores = []
        for i, prop in enumerate(properties_in_province):
            # Score based on document portfolio value
            doc_score = sum(self.document_importance_map.get(doc.doc_type, 1) 
                           for doc in prop.documents)
            property_scores.append((i, doc_score))
        
        # Sort properties by importance (descending) for Zipfian ranking
        sorted_property_scores = sorted(property_scores, key=lambda x: x[1], reverse=True)
        
        # Create rank mapping: property_index -> rank
        rank_mapping = {}
        for rank, (prop_index, score) in enumerate(sorted_property_scores, 1):
            rank_mapping[prop_index] = rank
        
        # Generate Zipfian probabilities for properties within province (Stage 2 - Inner Zipfian)
        weights = []
        for i in range(len(properties_in_province)):
            rank = rank_mapping[i]
            zipf_prob = 1.0 / (rank ** (self.property_zipf_parameter * 0.8))  # Slightly less skewed within province
            weights.append(zipf_prob)
        
        # Normalize to probabilities
        total_weight = sum(weights)
        return [w / total_weight for w in weights]
    
    def _get_location_importance_bonus(self, province: str) -> int:
        """
        Assigns location-based importance bonuses for property access frequency.
        Major economic centers tend to have higher property transaction volumes.
        """
        # Major economic centers get higher base access frequency
        major_provinces = {
            "DKI JAKARTA": 50,      # Capital, highest activity
            "JAWA BARAT": 40,       # Greater Jakarta area  
            "JAWA TENGAH": 35,      # Major industrial center
            "JAWA TIMUR": 35,       # Surabaya economic zone
            "BALI": 30,             # Tourism hub
            "RIAU": 25,             # Oil & gas center
            "SUMATERA UTARA": 20,   # Medan economic zone
        }
        return major_provinces.get(province, 10)  # Default for smaller provinces
    
    def get_document_pair_frequencies(self, documents: list, num_simulated_queries: int = 100) -> Counter:
        """
        Generates a PAIR co-occurrence frequency map (the "traffic log").
        This is the direct input for the Pairs-first Huffman algorithm.

        Uses either Zipfian distribution (realistic) or weighted sampling (legacy)
        to model document access patterns and pair co-occurrences.
        """
        # Set seed for reproducible document pair generation
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        if len(documents) < 2:
            return Counter()

        pair_counter = Counter()
        
        if self.use_zipfian:
            # Zipfian approach: Create realistic heavy-tail access distribution
            doc_ids = list(documents)
            
            # Sort documents by importance to create natural ranking
            base_frequencies = self.get_document_frequencies(documents)
            sorted_docs = sorted(doc_ids, key=lambda d: base_frequencies[d], reverse=True)
            
            # Generate Zipfian probabilities based on rank
            ranks = np.arange(1, len(sorted_docs) + 1)
            # Use power law: P(rank) = 1 / (rank^zipf_parameter)
            zipf_probs = 1.0 / (ranks ** self.zipf_parameter)
            
            # Normalize to probabilities
            zipf_probs = zipf_probs / zipf_probs.sum()
            
            # Create mapping from documents to Zipfian probabilities
            doc_zipf_weights = {doc: prob for doc, prob in zip(sorted_docs, zipf_probs)}
            
        else:
            # Legacy approach: weighted sampling based on importance map
            doc_frequencies = self.get_document_frequencies(documents)
            doc_ids = list(doc_frequencies.keys())
            weights = list(doc_frequencies.values())
            doc_zipf_weights = {doc: weight for doc, weight in zip(doc_ids, weights)}

        for _ in range(num_simulated_queries):
            # For each simulated query, pick a random number of documents to access
            num_docs_to_query = random.randint(2, min(len(doc_ids if not self.use_zipfian else doc_zipf_weights), 7))
            
            if self.use_zipfian:
                # Select documents using Zipfian distribution (more realistic)
                docs_list = list(doc_zipf_weights.keys())
                probs_list = list(doc_zipf_weights.values())
                queried_docs = random.choices(docs_list, weights=probs_list, k=num_docs_to_query)
            else:
                # Legacy weighted selection
                queried_docs = random.choices(doc_ids, weights=weights, k=num_docs_to_query)
                
            # Ensure we only count unique pairs (no need to sort, just get unique)
            unique_queried_docs = list(set(queried_docs))

            # If the query has at least 2 unique docs, find all pairs and count them
            if len(unique_queried_docs) >= 2:
                for pair in combinations(unique_queried_docs, 2):
                    # Sort the pair by a consistent attribute (like doc_id) for consistent ordering
                    sorted_pair = tuple(sorted(pair, key=lambda x: f"{x.province}.{x.property_id}.{x.doc_id}"))
                    pair_counter[sorted_pair] += 1
        
        # Apply alpha threshold filtering
        pruned_pair_counter = Counter()
        if pair_counter:
            threshold = max(1, int(self.alpha_threshold * pair_counter.total()))
            for pair, count in pair_counter.items():
                if count >= threshold:
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
                 avg_docs_per_property: int = 2, min_docs_per_property: int = 1,
                 random_seed: int = PATTERN_SEED):
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
        self.random_seed = random_seed

    def get_random_audit_region(self) -> str:
        """Selects a single, random province using a weighted distribution."""
        random.seed(self.random_seed)
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
    
    def get_regional_audit_zipfian_sample(self, properties_in_province: list, 
                                        document_importance_map: dict) -> list:
        """
        Selects properties within a province using Zipfian distribution with α = 0.5
        for more fair/balanced sampling in regional audits.
        
        Args:
            properties_in_province: List of Property objects in the selected province
            document_importance_map: Map of document types to importance scores
            
        Returns:
            List of selected Property objects using fair Zipfian distribution
        """
        if not properties_in_province:
            return []
        
        sample_size = self.get_audit_sample_size()
        
        # If we need all or more properties than available, return all
        if sample_size >= len(properties_in_province):
            return properties_in_province
        
        # Calculate property importance scores for ranking
        property_scores = []
        for i, prop in enumerate(properties_in_province):
            # Score based on document portfolio value
            doc_score = sum(document_importance_map.get(doc.doc_type, 1) 
                           for doc in prop.documents)
            property_scores.append((i, doc_score))
        
        # Sort properties by importance (descending) for Zipfian ranking
        sorted_property_scores = sorted(property_scores, key=lambda x: x[1], reverse=True)
        
        # Create rank mapping: property_index -> rank
        rank_mapping = {}
        for rank, (prop_index, score) in enumerate(sorted_property_scores, 1):
            rank_mapping[prop_index] = rank
        
        # Generate Zipfian probabilities with α = 0.5 (more fair/balanced)
        zipf_alpha = 0.5  # More balanced than transactional queries
        weights = []
        for i in range(len(properties_in_province)):
            rank = rank_mapping[i]
            zipf_prob = 1.0 / (rank ** zipf_alpha)
            weights.append(zipf_prob)
        
        # Normalize to probabilities
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Sample properties using Zipfian distribution (without replacement)
        selected_indices = []
        remaining_properties = list(range(len(properties_in_province)))
        remaining_weights = normalized_weights.copy()
        
        for _ in range(sample_size):
            if not remaining_properties:
                break
                
            # Select one property using current weights
            selected_idx = random.choices(remaining_properties, weights=remaining_weights, k=1)[0]
            selected_indices.append(selected_idx)
            
            # Remove selected property from remaining options
            removal_pos = remaining_properties.index(selected_idx)
            remaining_properties.pop(removal_pos)
            remaining_weights.pop(removal_pos)
            
            # Renormalize weights
            if remaining_weights:
                total_remaining_weight = sum(remaining_weights)
                remaining_weights = [w / total_remaining_weight for w in remaining_weights]
        
        # Return selected properties
        return [properties_in_province[i] for i in selected_indices]
    
    def get_national_audit_zipfian_sample(self, properties_by_province: dict,
                                        document_importance_map: dict,
                                        num_provinces_to_sample: int = 5) -> list:
        """
        Selects properties from multiple provinces for national audit using Zipfian distribution.
        Uses α = 0.5 for fair sampling within each selected province.
        
        Args:
            properties_by_province: Dict mapping province names to lists of Property objects
            document_importance_map: Map of document types to importance scores  
            num_provinces_to_sample: Number of provinces to include in national audit
            
        Returns:
            List of selected Property objects from multiple provinces
        """
        if len(self.provinces) < num_provinces_to_sample:
            selected_provinces = self.provinces
        else:
            # Select distinct provinces using weighted selection (economic importance)
            selected_provinces = random.sample(self.provinces, num_provinces_to_sample)

        final_sampled_properties = []
        
        # Determine how many properties to sample per province
        total_sample_size = self.get_audit_sample_size() * 2  # Larger for national audit
        properties_per_province = max(1, total_sample_size // len(selected_provinces))
        
        for province in selected_provinces:
            props_in_province = properties_by_province.get(province, [])
            if not props_in_province:
                continue
            
            # Use Zipfian sampling within each province (α = 0.5 for fairness)
            if len(props_in_province) <= properties_per_province:
                # Take all properties if we need more than available
                final_sampled_properties.extend(props_in_province)
            else:
                # Create a temporary AuditPattern for this province sampling
                temp_sample_size = properties_per_province
                
                # Calculate property importance scores for ranking
                property_scores = []
                for i, prop in enumerate(props_in_province):
                    doc_score = sum(document_importance_map.get(doc.doc_type, 1) 
                                   for doc in prop.documents)
                    property_scores.append((i, doc_score))
                
                # Sort properties by importance (descending) for Zipfian ranking
                sorted_property_scores = sorted(property_scores, key=lambda x: x[1], reverse=True)
                
                # Create rank mapping and apply Zipfian (α = 0.5)
                rank_mapping = {}
                for rank, (prop_index, score) in enumerate(sorted_property_scores, 1):
                    rank_mapping[prop_index] = rank
                
                zipf_alpha = 0.5  # Fair sampling for national audits
                weights = []
                for i in range(len(props_in_province)):
                    rank = rank_mapping[i]
                    zipf_prob = 1.0 / (rank ** zipf_alpha)
                    weights.append(zipf_prob)
                
                # Normalize and sample
                total_weight = sum(weights)
                normalized_weights = [w / total_weight for w in weights]
                
                # Sample without replacement
                selected_indices = []
                remaining_properties = list(range(len(props_in_province)))
                remaining_weights = normalized_weights.copy()
                
                for _ in range(temp_sample_size):
                    if not remaining_properties:
                        break
                        
                    selected_idx = random.choices(remaining_properties, weights=remaining_weights, k=1)[0]
                    selected_indices.append(selected_idx)
                    
                    # Remove and renormalize
                    removal_pos = remaining_properties.index(selected_idx)
                    remaining_properties.pop(removal_pos)
                    remaining_weights.pop(removal_pos)
                    
                    if remaining_weights:
                        total_remaining_weight = sum(remaining_weights)
                        remaining_weights = [w / total_remaining_weight for w in remaining_weights]
                
                # Add selected properties from this province
                final_sampled_properties.extend([props_in_province[i] for i in selected_indices])
        
        return final_sampled_properties
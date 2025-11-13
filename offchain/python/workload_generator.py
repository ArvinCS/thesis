import argparse
import random
from typing import TypeAlias, Final, Union
from access_patterns_enhanced import AuditPattern, TransactionalPattern
from seed_generator import SeedGenerator, Property
from basic_data_structure import Document

# Fixed seed for reproducible workload generation
WORKLOAD_SEED = 42

# --- Step 1: Import the refined data structures and generator ---
# This links all your framework components together.

# --- Step 2: Define the Query TypeAlias for clarity ---
# Format: (QUERY_TYPE, TARGET_PROPERTY_IDS, TARGET_DOCUMENT_IDS)
# A transactional query targets ONE property and a list of its documents
TransactionalQuery: TypeAlias = tuple[str, str, list[str]]

# An audit query targets a list of explicit (property, document) pairs
AuditQuery: TypeAlias = tuple[str, list[tuple[str, str]]]

# The main Query type is a union of the two specific types
Query: TypeAlias = Union[TransactionalQuery, AuditQuery]

# --- Step 3: The Complete WorkloadGenerator Class ---

class WorkloadGenerator:
    """
    Generates query workloads for benchmarking Merkle tree performance.

    This class takes a dataset of properties and can generate lists of
    queries that simulate different user access patterns (Transactional, Audit, Mixed).
    """
    def __init__(self, properties: list[Property], transactional_pattern: TransactionalPattern, audit_pattern: AuditPattern, random_seed: int = WORKLOAD_SEED):
        """
        Initializes the generator with the full property dataset.

        Args:
            properties: A list of Property objects from the SeedGenerator.
            transactional_pattern: Pattern for generating transactional queries.
            audit_pattern: Pattern for generating audit queries.
            random_seed: Fixed seed for reproducible workload generation.
        """
        if not properties:
            raise ValueError("Properties list cannot be empty.")
        
        self.properties = properties
        self.random_seed = random_seed
        
        # Validate document ID uniqueness across all properties
        all_doc_ids = []
        for prop in properties:
            for doc in prop.documents:
                all_doc_ids.append(doc.doc_id)
        
        unique_doc_ids = set(all_doc_ids)
        if len(all_doc_ids) != len(unique_doc_ids):
            # duplicates = [doc_id for doc_id in all_doc_ids if all_doc_ids.count(doc_id) > 1]
            # print(f"⚠️ WARNING: WorkloadGenerator found {len(all_doc_ids) - len(unique_doc_ids)} duplicate document IDs: {set(duplicates)}")
            print(f"⚠️ WARNING: WorkloadGenerator found {len(all_doc_ids) - len(unique_doc_ids)} duplicate documents")
        else:
            print(f"✅ WorkloadGenerator: All {len(all_doc_ids)} document IDs are unique across {len(properties)} properties")
        self.transactional_pattern = transactional_pattern
        self.audit_pattern = audit_pattern
        
        # Pre-process data for efficient audit query generation
        self._doc_type_map = self._build_doc_type_map()
        self._available_doc_types = list(self._doc_type_map.keys())
        self._property_map = {p.property_id: p for p in self.properties}
        self._province_map = self._build_province_map()

    def _build_province_map(self) -> dict[str, list]:
        """Helper to group properties by province for fast lookups."""
        province_map = {p: [] for p in self.audit_pattern.provinces}
        for prop in self.properties:
            if prop.province in province_map:
                province_map[prop.province].append(prop)
        return province_map
    
    def _build_doc_type_map(self) -> dict[str, list[tuple[str, str]]]:
        """Helper to map doc_type to a list of (property_id, doc_id)."""
        doc_map = {}
        for prop in self.properties:
            for doc in prop.documents:
                # The doc_type is now an attribute of your Document class
                if doc.doc_type not in doc_map:
                    doc_map[doc.doc_type] = []
                doc_map[doc.doc_type].append((prop.property_id, doc.doc_id))
        return doc_map

    def generate_transactional_workload(self, num_queries: int) -> list[Query]:
        """
        Generates a transactional workload using hierarchical Zipfian distribution.
        
        TRANSACTIONAL PATTERN: Choose ONE property, query MULTIPLE documents within that property
        
        Two-stage sampling process:
        Stage 1: Provincial Access Frequency (Outer Zipfian) - Select province
        Stage 2: Within-Province Property Selection (Inner Zipfian) - Select property in province
        """
        # Set seed for reproducible transactional workload
        random.seed(self.random_seed)
        queries = []
        
        # Stage 1: Get provincial-level Zipfian weights (Outer Zipfian)
        province_weights_map = self.transactional_pattern.get_province_zipfian_weights(self.properties)
        
        # Prepare for Stage 1 sampling
        provinces_list = list(province_weights_map.keys())
        province_weights_list = [weight for weight, _ in province_weights_map.values()]
        
        for _ in range(num_queries):
            # Stage 1: Select province using Outer Zipfian distribution
            selected_province = random.choices(provinces_list, weights=province_weights_list, k=1)[0]
            
            # Get properties in the selected province
            _, properties_in_province = province_weights_map[selected_province]
            
            # Stage 2: Select ONE property within province using Inner Zipfian distribution
            within_province_weights = self.transactional_pattern.get_within_province_zipfian_weights(properties_in_province)
            target_property = random.choices(properties_in_province, weights=within_province_weights, k=1)[0]
            
            # Get the frequency distribution for this property's documents
            doc_frequencies = self.transactional_pattern.get_document_frequencies(target_property.documents)
            
            docs = list(doc_frequencies.keys())  # Document objects
            weights = list(doc_frequencies.values())

            # Select MULTIPLE documents from this SINGLE property (typical transactional pattern)
            num_docs_to_query = random.randint(1, min(len(docs), 7))
            
            if not docs: 
                continue # Skip if property has no docs
            
            queried_docs = random.choices(
                docs,
                weights=weights,
                k=num_docs_to_query
            )
            # Ensure unique documents in the query and extract doc_ids
            unique_docs = list(set(queried_docs))
            queried_doc_ids = [doc.doc_id for doc in unique_docs]
            
            # Debug: Verify that all queried_doc_ids actually belong to this property
            property_doc_ids = [doc.doc_id for doc in target_property.documents]
            invalid_docs = [doc_id for doc_id in queried_doc_ids if doc_id not in property_doc_ids]
            if invalid_docs:
                print(f"⚠️ WorkloadGenerator BUG: Property {target_property.property_id} doesn't have docs: {invalid_docs}")
                print(f"   Property has: {property_doc_ids}")
                print(f"   Query requested: {queried_doc_ids}")
                # Fix by filtering to only valid docs
                queried_doc_ids = [doc_id for doc_id in queried_doc_ids if doc_id in property_doc_ids]
                if not queried_doc_ids:
                    continue  # Skip this query if no valid docs
            
            # TRANSACTIONAL: Single property, multiple documents
            query = ('TRANSACTIONAL', target_property.property_id, queried_doc_ids)
            queries.append(query)
            
        return queries

    # deprecated
    def generate_audit_workload(self, num_queries: int) -> list[Query]:
        """
        Generates a flexible audit workload that can span multiple provinces with selective document sampling.

        Each query simulates audit scenarios that can involve:
        - One or more provinces (based on audit pattern configuration)
        - Selective document sampling from properties (not necessarily all documents)
        - Multiple document types (can be enhanced further if needed)
        """
        if not self._available_doc_types:
            return []

        queries: list[Query] = []
        for _ in range(num_queries):
            # 1. Select one or more document types to audit (start with 1, can be enhanced)
            target_doc_types = [random.choice(self._available_doc_types)]
            
            # 2. Select one or more provinces for this audit
            target_provinces = self.audit_pattern.get_random_audit_regions()
            
            audit_targets = []
            
            # 3. Process each selected province
            for target_province in target_provinces:
                properties_in_province = self._province_map.get(target_province, [])
                if not properties_in_province:
                    continue  # No properties in this province, skip
                
                # 4. Sample properties within this province
                sample_size = self.audit_pattern.get_audit_sample_size()
                
                if len(properties_in_province) > sample_size:
                    sampled_properties = random.sample(properties_in_province, sample_size)
                else:
                    sampled_properties = properties_in_province
                
                # 5. For each sampled property, select documents more flexibly
                for prop in sampled_properties:
                    # Get documents of target types from this property
                    eligible_docs = []
                    for doc in prop.documents:
                        if doc.doc_type in target_doc_types:
                            eligible_docs.append(doc)
                    
                    # 6. Select a subset of eligible documents (not necessarily all)
                    if eligible_docs:
                        docs_to_select = self.audit_pattern.get_docs_per_property_count()
                        docs_to_select = min(docs_to_select, len(eligible_docs))
                        
                        if docs_to_select == len(eligible_docs):
                            selected_docs = eligible_docs
                        else:
                            selected_docs = random.sample(eligible_docs, docs_to_select)
                        
                        # Add selected documents to audit targets
                        for doc in selected_docs:
                            audit_targets.append((prop.property_id, doc.doc_id))

            if audit_targets:
                query = ('AUDIT', audit_targets)
                queries.append(query)
            
        return queries

    def generate_regional_audit_workload(self, num_queries: int) -> list[Query]:
        """
        Generates a geographically-focused REGIONAL audit workload.
        
        REGIONAL AUDIT PATTERN: Choose ONE OR MORE properties within SAME province
        """
        # Set seed for reproducible regional audit workload  
        random.seed(self.random_seed + 1)  # Use offset to ensure different from transactional
        if not self._available_doc_types:
            return []

        queries = []
        for _ in range(num_queries):
            # Select a specific document type to audit
            target_doc_type = random.choice(self._available_doc_types)
            
            # Select a single province for this regional audit
            target_province = self.audit_pattern.get_random_audit_region()
            
            properties_in_province = self._province_map.get(target_province, [])
            if not properties_in_province:
                continue

            # Sample multiple properties within this single province using Zipfian (α = 0.5)
            # This creates more fair/balanced selection compared to transactional queries
            sampled_properties = self.audit_pattern.get_regional_audit_zipfian_sample(
                properties_in_province, 
                self.transactional_pattern.document_importance_map
            )
            
            # Collect documents of target type from ALL sampled properties in the province
            audit_targets = []
            for prop in sampled_properties:
                for doc in prop.documents:
                    if doc.doc_type == target_doc_type:
                        audit_targets.append((prop.property_id, doc.doc_id))

            if audit_targets:
                # REGIONAL_AUDIT: Multiple properties, same province, specific document type
                query: Query = ('REGIONAL_AUDIT', audit_targets)
                queries.append(query)
            
        return queries
    
    def generate_national_audit_workload(self, num_queries: int) -> list[Query]:
        """
        Generates a cross-province NATIONAL audit workload (stress test).
        
        NATIONAL AUDIT PATTERN: Choose ONE OR MORE properties across MULTIPLE provinces
        """
        # Set seed for reproducible national audit workload
        random.seed(self.random_seed + 2)  # Use offset to ensure different sequence
        if not self._available_doc_types:
            return []

        queries = []
        for _ in range(num_queries):
            # Select a specific document type to audit nationally
            target_doc_type = random.choice(self._available_doc_types)
            
            # Sample properties from MULTIPLE provinces using Zipfian (α = 0.5) within each province
            sampled_properties = self.audit_pattern.get_national_audit_zipfian_sample(
                self._province_map,
                self.transactional_pattern.document_importance_map
            )

            # Collect documents of target type from properties across ALL provinces
            audit_targets = []
            for prop in sampled_properties:
                for doc in prop.documents:
                    if doc.doc_type == target_doc_type:
                        audit_targets.append((prop.property_id, doc.doc_id))
            
            if audit_targets:
                # NATIONAL_AUDIT: Multiple properties, multiple provinces, specific document type
                query: Query = ('NATIONAL_AUDIT', audit_targets)
                queries.append(query)
        
        return queries
    
    def generate_mixed_workload(
        self, total_queries: int, transactional_ratio: float = 0.99
    ) -> list[Query]:
        """
        Generates a mixed workload for normal operations: Transactional + Regional Audit.

        This represents typical day-to-day usage patterns where users perform
        property transactions and auditors perform regional (same-province) audits.

        Args:
            total_queries: The total number of queries to generate.
            transactional_ratio: The percentage of queries that should be transactional.

        Returns:
            A shuffled list of transactional and regional audit query tuples.
        """
        num_transactional = int(total_queries * transactional_ratio)
        num_regional_audit = total_queries - num_transactional

        # 1. Generate the component workloads for normal operations
        transactional_queries = self.generate_transactional_workload(num_transactional)
        regional_audit_queries = self.generate_regional_audit_workload(num_regional_audit)

        # 2. Combine them into a single mixed workload
        mixed_queries = transactional_queries + regional_audit_queries
        
        # 3. Shuffle the list to simulate realistic interleaved usage (use separate seed for shuffle)
        random.seed(self.random_seed + 10)  # Use distinct seed for shuffling
        random.shuffle(mixed_queries)
        
        return mixed_queries

    def generate_stress_test_workload(self, total_queries: int) -> list[Query]:
        """
        Generates a stress test workload consisting purely of National Audit queries.

        This represents intensive cross-province audit scenarios that stress-test
        the system's ability to handle geographically distributed queries that
        span multiple provinces simultaneously.

        Args:
            total_queries: The total number of national audit queries to generate.

        Returns:
            A list of national audit query tuples for stress testing.
        """
        return self.generate_national_audit_workload(total_queries)

# --- Step 4: Example Usage Demonstrating the Full Pipeline ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Workload Generator')
    parser.add_argument('--properties', type=int, default=5000, help='Number of properties to generate')
    
    args = parser.parse_args()
    
    # --- Part 1: Generate a realistic dataset ---
    print("--- 1. Generating Seed Data ---")
    seed_gen = SeedGenerator(total_properties=args.properties)
    property_dataset = seed_gen.generate_dataset()
    print("-" * 30)

    # --- Part 2: Use the generated dataset to create workloads ---
    print("\n--- 2. Initializing Workload Generator ---")
    workload_gen = WorkloadGenerator(property_dataset)
    print("Workload generator is ready.")
    print("-" * 30)

    # --- Part 3: Generate and display each type of workload ---
    
    # Example 1: Pure Transactional
    print("\n--- Generating 100% Transactional Workload (3 examples) ---")
    transactional_workload = workload_gen.generate_transactional_workload(3)
    for q in transactional_workload:
        print(f"  - Query: {q[0]}, Property: {q[1][0]}, Num_Docs: {len(q[2])}")
        
    # Example 2: Pure Audit
    print("\n--- Generating 100% Audit Workload (2 examples) ---")
    audit_workload = workload_gen.generate_audit_workload(2)
    for q in audit_workload:
        print(f"  - Query: {q[0]}, Num_Properties: {len(q[1])}, Num_Docs: {len(q[2])}")
        
    # Example 3: The realistic Mixed Workload
    print("\n--- Generating 99/1 Mixed Workload (100 total queries) ---")
    mixed_workload = workload_gen.generate_mixed_workload(100)
    print(f"  Generated {len(mixed_workload)} mixed queries.")
    print("  First 5 queries in the shuffled list:")
    for q in mixed_workload[:5]:
        print(f"    - Query Type: {q[0]}")
    print("-" * 30)
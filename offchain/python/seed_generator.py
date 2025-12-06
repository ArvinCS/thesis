import random
import uuid
from typing import Final
from basic_data_structure import Document, Property

# Set fixed seed for reproducible dataset generation
FIXED_SEED = 42

class SeedGenerator:
    """
    Generates a realistic dataset of Indonesian properties for benchmarking.

    This version is refined to use the specific Document class from
    basic_data_structure.py, which includes content hashing.
    """

    # SOURCE: Badan Pusat Statistik (BPS) Indonesia, 2023.
    # Data represents the number of households (in thousands) per province.
    _PROVINCE_DISTRIBUTION_RAW: Final[dict[str, int]] = {
        "JAWA BARAT": 13533,
        "JAWA TIMUR": 11538,
        "JAWA TENGAH": 10217,
        "SUMATERA UTARA": 4149,
        "BANTEN": 3614,
        "DKI JAKARTA": 2999,
        "SULAWESI SELATAN": 2577,
        "LAMPUNG": 2542,
        "SUMATERA SELATAN": 2484,
        "RIAU": 1969,
        "SUMATERA BARAT": 1600,
        "KALIMANTAN BARAT": 1599,
        "NUSA TENGGARA TIMUR": 1478,
        "NUSA TENGGARA BARAT": 1475,
        "KALIMANTAN SELATAN": 1282,
        "ACEH": 1261,
        "JAMBI": 1076,
        "KALIMANTAN TIMUR": 1069,
        # The rest of the provinces...
        "SULAWESI TENGAH": 844,
        "KEPULAUAN RIAU": 688,
        "SULAWESI TENGGARA": 751,
        "KALIMANTAN TENGAH": 790,
        "PAPUA": 732,
        "BENGKULU": 591,
        "DI YOGYAKARTA": 1133,
        "MALUKU": 487,
        "SULAWESI UTARA": 757,
        "BALI": 1242,
        "BANGKA BELITUNG": 427,
        "GORONTALO": 349,
        "MALUKU UTARA": 346,
        "SULAWESI BARAT": 401,
        "PAPUA BARAT": 283,
        "KALIMANTAN UTARA": 204,
        "PAPUA SELATAN": 140,
        "PAPUA TENGAH": 312,
        "PAPUA PEGUNUNGAN": 321,
        "PAPUA BARAT DAYA": 160,
    }

    # Define the types of documents a property might have.
    _DOCUMENT_TYPES: Final[dict[str, float]] = {
        "TitleDeed": 1.0,  # Sertifikat Hak Milik (Always exists)
        "BuildingPermit": 0.8, # Izin Mendirikan Bangunan (IMB)
        "LandAndBuildingTax2025": 0.95, # PBB-P2
        "LandAndBuildingTax2024": 0.98, # PBB-P2
        "ElectricityBill_Latest": 0.9,
        "WaterBill_Latest": 0.75,
    }

    def __init__(self, total_properties: int, random_seed: int = FIXED_SEED):
        self.total_properties = total_properties
        self.random_seed = random_seed
        total_households = sum(self._PROVINCE_DISTRIBUTION_RAW.values())
        self.province_weights = {
            p: c / total_households for p, c in self._PROVINCE_DISTRIBUTION_RAW.items()
        }

    def _create_random_documents(self, province: str, property_id: str) -> list[Document]:
        """
        Generates a list of Document objects for a single property.
        
        Args:
            province: The province of the parent property.
            property_id: The ID of the parent property.

        Returns:
            A list of fully instantiated Document objects.
        """
        documents = []
        for doc_type, probability in self._DOCUMENT_TYPES.items():
            if random.random() < probability:
                doc_id = f"doc_{doc_type}_{uuid.uuid4().hex[:10]}"
                
                # Generate unique, placeholder content for hashing
                content = (
                    f"This is the official content for document '{doc_id}' of type '{doc_type}' "
                    f"for property '{property_id}' in {province}. "
                    f"Generated on {__import__('datetime').datetime.now().isoformat()}."
                )
                
                # Instantiate YOUR Document class
                new_doc = Document(
                    doc_id=doc_id,
                    content=content,
                    doc_type=doc_type,
                    province=province,
                    property_id=property_id
                )
                documents.append(new_doc)
        return documents

    def _create_random_property(self, province: str) -> Property:
        """Creates a single, randomized Property object for a given province."""
        prop_id = f"prop_{province.replace(' ', '_')}_{uuid.uuid4().hex[:10]}"
        
        # Pass the necessary IDs down to the document creation method
        documents = self._create_random_documents(province=province, property_id=prop_id)
        
        return Property(property_id=prop_id, province=province, documents=documents)

    def generate_dataset(self) -> list[Property]:
        """
        Generates the full dataset of properties based on provincial distribution.
        Uses fixed ratio split for deterministic, reproducible results.
        
        Minimum requirement: 500 properties to ensure all provinces get representation.
        Recommended scale: 100,000+ properties for production benchmarks.
        """
        # Validate minimum property count
        MIN_PROPERTIES = 500
        if self.total_properties < MIN_PROPERTIES:
            raise ValueError(
                f"Property count must be at least {MIN_PROPERTIES} to ensure all 38 provinces "
                f"get adequate representation. You requested {self.total_properties}.\n"
                f"Recommended scales:\n"
                f"  - Small test: 1,000 properties\n"
                f"  - Medium benchmark: 10,000 properties\n"
                f"  - Large benchmark: 100,000 properties\n"
                f"  - Production scale: 1,000,000+ properties"
            )
        
        # Set fixed seed for reproducible results
        random.seed(self.random_seed)
        print(f"Generating a dataset of {self.total_properties} properties (seed={self.random_seed})...")
        print("Using FIXED RATIO SPLIT for deterministic distribution...")
        dataset = []
        
        provinces = list(self.province_weights.keys())
        weights = list(self.province_weights.values())
        
        # Calculate exact property counts per province using fixed ratios
        property_counts = {}
        allocated_total = 0
        
        # First pass: allocate floor values
        for province, weight in zip(provinces, weights):
            exact_count = weight * self.total_properties
            floor_count = int(exact_count)
            property_counts[province] = {
                'count': floor_count,
                'remainder': exact_count - floor_count
            }
            allocated_total += floor_count
        
        # Second pass: distribute remaining properties to provinces with largest remainders
        remaining = self.total_properties - allocated_total
        if remaining > 0:
            # Sort by remainder (descending) and allocate one extra property each
            sorted_by_remainder = sorted(
                property_counts.items(), 
                key=lambda x: x[1]['remainder'], 
                reverse=True
            )
            for i in range(remaining):
                province_name = sorted_by_remainder[i][0]
                property_counts[province_name]['count'] += 1
        
        # Generate properties for each province
        for province_name in provinces:
            count = property_counts[province_name]['count']
            for _ in range(count):
                dataset.append(self._create_random_property(province=province_name))
        
        print("Dataset generation complete.")
        print("\n--- Top 5 Provinces in Generated Dataset ---")
        final_counts = {p: property_counts[p]['count'] for p in provinces}
        sorted_counts = sorted(final_counts.items(), key=lambda item: item[1], reverse=True)
        for province, count in sorted_counts[:5]:
            percentage = (count / self.total_properties) * 100
            expected_percentage = self.province_weights[province] * 100
            print(f"  - {province:<20}: {count:>5} properties ({percentage:.2f}% - expected {expected_percentage:.2f}%)")
            
        return dataset

# --- Step 4: Example Usage ---

if __name__ == "__main__":
    TOTAL_PROPERTIES_TO_GENERATE = 1000  # Smaller number for a quick demo

    seed_generator = SeedGenerator(total_properties=TOTAL_PROPERTIES_TO_GENERATE)
    property_dataset = seed_generator.generate_dataset()

    print(f"\nTotal properties in final dataset: {len(property_dataset)}")
    if property_dataset:
        print("\n--- Example Property ---")
        example_prop = random.choice(property_dataset)
        print(f"  Property ID: {example_prop.property_id}")
        print(f"  Province:    {example_prop.province}")
        print("  Documents:")
        for doc in example_prop.documents:
            # The __repr__ comes from your Document class
            print(f"    - {doc}")
            print(f"      Hash: {doc.hash_hex}")
"""
Large Scale Document and Traffic Generator for Hierarchical Merkle Tree Benchmarking

This module generates thousands of documents across Indonesian provinces with realistic
distribution patterns and traffic logs for comprehensive benchmarking.
"""

import json
import random
import os
from datetime import datetime, timedelta
from collections import defaultdict
from optimized_tree_builder import Document

# Import report organizer for structured file saving
try:
    from report_organizer import save_organized_file
except ImportError:
    # Fallback if report_organizer is not available
    def save_organized_file(data, filename, file_type="generated_data"):
        import json
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        return filename

# Indonesian provinces with realistic population-based weights
INDONESIAN_PROVINCES = {
    'Jawa_Barat': {'weight': 15, 'base_docs': 200},      # Most populous
    'Jawa_Tengah': {'weight': 12, 'base_docs': 150},
    'Jawa_Timur': {'weight': 13, 'base_docs': 170},
    'Jakarta': {'weight': 8, 'base_docs': 120},          # High value, commercial
    'Sumatera_Utara': {'weight': 5, 'base_docs': 80},
    'Sumatera_Selatan': {'weight': 4, 'base_docs': 60},
    'Sumatera_Barat': {'weight': 3, 'base_docs': 50},
    'Bali': {'weight': 2, 'base_docs': 40},              # Tourism
    'Sulawesi_Selatan': {'weight': 4, 'base_docs': 70},
    'Kalimantan_Timur': {'weight': 2, 'base_docs': 35},  # Mining/oil
    'Kalimantan_Selatan': {'weight': 2, 'base_docs': 30},
    'Papua': {'weight': 1, 'base_docs': 20},
    'Maluku': {'weight': 1, 'base_docs': 15},
    'Nusa_Tenggara_Barat': {'weight': 2, 'base_docs': 25},
    'Lampung': {'weight': 3, 'base_docs': 45},
}

# Document types with realistic frequencies
DOCUMENT_TYPES = {
    'deed': {'weight': 30, 'template': 'Property deed for {} in {}'},
    'title': {'weight': 25, 'template': 'Property title certificate for {} in {}'},
    'permit': {'weight': 20, 'template': 'Building permit for {} development in {}'},
    'survey': {'weight': 15, 'template': 'Land survey report for {} in {}'},
    'license': {'weight': 5, 'template': 'Business license for {} operations in {}'},
    'tax': {'weight': 3, 'template': 'Property tax assessment for {} in {}'},
    'mortgage': {'weight': 2, 'template': 'Mortgage agreement for {} in {}'},
}

# Property types with different verification patterns
PROPERTY_TYPES = {
    'residential': {'weight': 50, 'docs_per_property': [2, 3, 4]},
    'commercial': {'weight': 25, 'docs_per_property': [3, 4, 5, 6]},
    'industrial': {'weight': 15, 'docs_per_property': [4, 5, 6, 7]},
    'agricultural': {'weight': 8, 'docs_per_property': [2, 3]},
    'tourism': {'weight': 2, 'docs_per_property': [3, 4, 5]},
}

class LargeScaleDocumentGenerator:
    """Generates thousands of documents with realistic distribution patterns."""
    
    def __init__(self, target_document_count=5000, seed=42):
        self.target_document_count = target_document_count
        self.seed = seed
        random.seed(seed)
        
        self.documents = []
        self.properties_by_province = defaultdict(list)
        self.generation_stats = {}
    
    def generate_documents(self):
        """Generate documents distributed across Indonesian provinces."""
        print(f"Generating {self.target_document_count} documents across {len(INDONESIAN_PROVINCES)} provinces...")
        
        # Calculate document distribution by province
        total_weight = sum(p['weight'] for p in INDONESIAN_PROVINCES.values())
        province_doc_counts = {}
        
        for province, info in INDONESIAN_PROVINCES.items():
            proportion = info['weight'] / total_weight
            doc_count = max(info['base_docs'], int(self.target_document_count * proportion))
            province_doc_counts[province] = doc_count
        
        # Generate documents for each province
        doc_id_counter = 1
        
        for province, target_docs in province_doc_counts.items():
            print(f"  Generating {target_docs} documents for {province}...")
            
            # Generate properties for this province
            properties_count = max(10, target_docs // 4)  # Average 4 docs per property
            province_properties = []
            
            for prop_idx in range(properties_count):
                # Select property type
                prop_type = self._weighted_choice(PROPERTY_TYPES)
                property_id = f"Prop_{prop_type[:3].upper()}_{prop_idx+1:04d}"
                full_property_id = f"{province}.{property_id}"
                
                # Determine documents per property
                docs_per_prop = random.choice(PROPERTY_TYPES[prop_type]['docs_per_property'])
                
                property_docs = []
                for doc_idx in range(docs_per_prop):
                    if doc_id_counter > self.target_document_count:
                        break
                        
                    # Select document type
                    doc_type = self._weighted_choice(DOCUMENT_TYPES)
                    
                    # Create document
                    doc_id = f"doc_{doc_id_counter:06d}"
                    location = self._get_realistic_location(province)
                    # CRITICAL FIX: Add document index to ensure uniqueness within property
                    # This prevents duplicate hashes when multiple documents have same type/location
                    base_content = DOCUMENT_TYPES[doc_type]['template'].format(property_id, location)
                    content = f"{base_content} [Document #{doc_idx+1}]"
                    
                    document = Document(doc_id, content, province, property_id)
                    self.documents.append(document)
                    property_docs.append(document)
                    doc_id_counter += 1
                
                if property_docs:
                    province_properties.append({
                        'property_id': property_id,
                        'full_id': full_property_id,
                        'type': prop_type,
                        'documents': property_docs
                    })
            
            self.properties_by_province[province] = province_properties
        
        # Update generation stats
        self.generation_stats = {
            'total_documents': len(self.documents),
            'total_properties': sum(len(props) for props in self.properties_by_province.values()),
            'provinces': len(self.properties_by_province),
            'documents_per_province': {p: len([d for d in self.documents if d.province == p]) 
                                    for p in INDONESIAN_PROVINCES.keys()}
        }
        
        print(f"Generated {len(self.documents)} documents across {len(self.properties_by_province)} provinces")
        return self.documents
    
    def _weighted_choice(self, choices_dict):
        """Select item based on weights."""
        choices = list(choices_dict.keys())
        weights = [choices_dict[choice]['weight'] for choice in choices]
        return random.choices(choices, weights=weights)[0]
    
    def _get_realistic_location(self, province):
        """Get realistic location names for Indonesian provinces."""
        locations = {
            'Jakarta': ['Menteng', 'Kemang', 'Senayan', 'Kelapa Gading', 'PIK'],
            'Jawa_Barat': ['Bandung', 'Bekasi', 'Bogor', 'Depok', 'Karawang'],
            'Jawa_Tengah': ['Semarang', 'Solo', 'Yogyakarta', 'Magelang', 'Purwokerto'],
            'Jawa_Timur': ['Surabaya', 'Malang', 'Kediri', 'Blitar', 'Jember'],
            'Bali': ['Denpasar', 'Ubud', 'Sanur', 'Kuta', 'Nusa Dua'],
            'Sumatera_Utara': ['Medan', 'Binjai', 'Pematangsiantar', 'Tanjungbalai'],
        }
        
        # Default locations for provinces not in the detailed list
        default_locations = ['Central District', 'North District', 'South District', 'East District', 'West District']
        
        province_locations = locations.get(province, default_locations)
        return random.choice(province_locations)
    
    def get_generation_stats(self):
        """Get statistics about the generated documents."""
        return self.generation_stats
    
    def save_documents(self, filename):
        """Save generated documents to JSON file using organized structure."""
        docs_data = []
        for doc in self.documents:
            docs_data.append({
                'doc_id': doc.doc_id,
                'content': doc.content,
                'province': doc.province,
                'property_id': doc.property_id
            })
        
        # Save using organized structure
        saved_file = save_organized_file(docs_data, filename, "generated_data")
        print(f"Saved {len(docs_data)} documents to {saved_file}")
        return saved_file

class RealisticTrafficGenerator:
    """Generates realistic traffic patterns for document verification."""
    
    def __init__(self, documents, properties_by_province):
        self.documents = documents
        self.properties_by_province = properties_by_province
        self.traffic_patterns = {
            'intra_province': 0.59,      # 59% verifications within same province
            'cross_province_2': 0.22,    # 22% across 2 provinces
            'cross_province_3': 0.10,    # 10% across 3 provinces
            'cross_province_4plus': 0.09  # 9% across 4+ provinces
        }
    
    def generate_traffic_logs(self, num_events=2000):
        """Generate realistic traffic logs with various verification patterns."""
        print(f"Generating {num_events} traffic verification events...")
        
        traffic_logs = []
        province_names = list(self.properties_by_province.keys())
        
        for event_idx in range(num_events):
            # Determine verification pattern
            pattern = self._choose_verification_pattern()
            
            if pattern == 'intra_province':
                # Single province verification
                province = random.choice(province_names)
                properties = self.properties_by_province[province]
                
                if properties:
                    # Select 1-5 properties from same province
                    num_props = min(random.randint(1, 5), len(properties))
                    selected_props = random.sample(properties, num_props)
                    event = [prop['full_id'] for prop in selected_props]
                    traffic_logs.append(event)
            
            elif pattern == 'cross_province_2':
                # Cross-province verification (2 provinces)
                selected_provinces = random.sample(province_names, 2)
                event = []
                
                for province in selected_provinces:
                    properties = self.properties_by_province[province]
                    if properties:
                        num_props = random.randint(1, 3)
                        selected = random.sample(properties, min(num_props, len(properties)))
                        event.extend([prop['full_id'] for prop in selected])
                
                if event:
                    traffic_logs.append(event)
            
            elif pattern == 'cross_province_3':
                # Cross-province verification (3 provinces)
                selected_provinces = random.sample(province_names, min(3, len(province_names)))
                event = []
                
                for province in selected_provinces:
                    properties = self.properties_by_province[province]
                    if properties:
                        num_props = random.randint(1, 2)
                        selected = random.sample(properties, min(num_props, len(properties)))
                        event.extend([prop['full_id'] for prop in selected])
                
                if event:
                    traffic_logs.append(event)
            
            else:  # cross_province_4plus
                # Large cross-province verification (4+ provinces)
                num_provinces = min(random.randint(4, 6), len(province_names))
                selected_provinces = random.sample(province_names, num_provinces)
                event = []
                
                for province in selected_provinces:
                    properties = self.properties_by_province[province]
                    if properties:
                        num_props = random.randint(1, 2)
                        selected = random.sample(properties, min(num_props, len(properties)))
                        event.extend([prop['full_id'] for prop in selected])
                
                if event:
                    traffic_logs.append(event)
        
        print(f"Generated {len(traffic_logs)} traffic events")
        return traffic_logs
    
    def _choose_verification_pattern(self):
        """Choose verification pattern based on realistic probabilities."""
        rand = random.random()
        cumulative = 0
        
        for pattern, probability in self.traffic_patterns.items():
            cumulative += probability
            if rand <= cumulative:
                return pattern
        
        return 'intra_province'  # fallback
    
    def save_traffic_logs(self, traffic_logs, filename):
        """Save traffic logs to JSON file using organized structure."""
        # Save using organized structure
        saved_file = save_organized_file(traffic_logs, filename, "generated_data")
        print(f"Saved {len(traffic_logs)} traffic events to {saved_file}")
        return saved_file
    
    def analyze_traffic_patterns(self, traffic_logs):
        """Analyze the generated traffic patterns for insights."""
        stats = {
            'total_events': len(traffic_logs),
            'intra_province_events': 0,
            'cross_province_events': 0,
            'avg_properties_per_event': 0,
            'max_properties_per_event': 0,
            'province_frequency': defaultdict(int),
            'cross_province_distribution': defaultdict(int)
        }
        
        total_properties = 0
        
        for event in traffic_logs:
            total_properties += len(event)
            stats['max_properties_per_event'] = max(stats['max_properties_per_event'], len(event))
            
            # Analyze provinces involved
            provinces_in_event = set()
            for prop_id in event:
                if '.' in prop_id:
                    province = prop_id.split('.')[0]
                    provinces_in_event.add(province)
                    stats['province_frequency'][province] += 1
            
            num_provinces = len(provinces_in_event)
            if num_provinces == 1:
                stats['intra_province_events'] += 1
            else:
                stats['cross_province_events'] += 1
                stats['cross_province_distribution'][num_provinces] += 1
        
        stats['avg_properties_per_event'] = total_properties / len(traffic_logs) if traffic_logs else 0
        
        return stats

def main():
    """Generate large scale test data for benchmarking."""
    print("=== LARGE SCALE DOCUMENT AND TRAFFIC GENERATOR ===")
    
    # Configuration
    TARGET_DOCUMENTS = 5000  # Adjust this for different scales
    TARGET_TRAFFIC_EVENTS = 2000
    
    # Generate documents
    doc_generator = LargeScaleDocumentGenerator(target_document_count=TARGET_DOCUMENTS)
    documents = doc_generator.generate_documents()
    
    # Print generation statistics
    stats = doc_generator.get_generation_stats()
    print(f"\nDocument Generation Statistics:")
    print(f"  Total Documents: {stats['total_documents']}")
    print(f"  Total Properties: {stats['total_properties']}")
    print(f"  Provinces: {stats['provinces']}")
    print(f"  Avg Documents per Province: {stats['total_documents'] / stats['provinces']:.1f}")
    
    # Generate traffic logs
    traffic_generator = RealisticTrafficGenerator(documents, doc_generator.properties_by_province)
    traffic_logs = traffic_generator.generate_traffic_logs(TARGET_TRAFFIC_EVENTS)
    
    # Analyze traffic patterns
    traffic_stats = traffic_generator.analyze_traffic_patterns(traffic_logs)
    print(f"\nTraffic Generation Statistics:")
    print(f"  Total Events: {traffic_stats['total_events']}")
    print(f"  Intra-Province Events: {traffic_stats['intra_province_events']} ({traffic_stats['intra_province_events']/traffic_stats['total_events']*100:.1f}%)")
    print(f"  Cross-Province Events: {traffic_stats['cross_province_events']} ({traffic_stats['cross_province_events']/traffic_stats['total_events']*100:.1f}%)")
    print(f"  Avg Properties per Event: {traffic_stats['avg_properties_per_event']:.1f}")
    print(f"  Max Properties per Event: {traffic_stats['max_properties_per_event']}")
    
    # Save generated data
    doc_generator.save_documents('large_scale_documents.json')
    traffic_generator.save_traffic_logs(traffic_logs, 'large_scale_traffic_logs.json')
    
    print(f"\nGenerated data saved successfully!")
    print(f"  Documents: large_scale_documents.json")
    print(f"  Traffic Logs: large_scale_traffic_logs.json")
    
    return documents, traffic_logs

if __name__ == "__main__":
    main()

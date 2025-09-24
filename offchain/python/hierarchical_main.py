import json
import os
from web3 import Web3
from optimized_tree_builder import Document
from jurisdiction_tree_manager import JurisdictionTreeManager

# --- CONFIGURATION ---
HARDHAT_URL = "http://127.0.0.1:8545"
ARTIFACT_PATH = '../../artifacts/contracts/HierarchicalMerkleVerifier.sol/HierarchicalMerkleVerifier.json'
DOCUMENTS_DB_FILE = 'hierarchical_documents.json'
TRAFFIC_LOGS_FILE = 'hierarchical_traffic_logs.json'

# --- WEB3 SETUP ---
web3 = Web3(Web3.HTTPProvider(HARDHAT_URL))

if not web3.is_connected():
    print("Failed to connect to Hardhat node. Please ensure it's running.")
    exit()

try:
    with open(ARTIFACT_PATH, 'r') as f:
        artifact = json.load(f)
    contract_abi = artifact['abi']
    contract_address = artifact['networks']['31337']['address']
except FileNotFoundError:
    print(f"Error: Artifact file not found at '{ARTIFACT_PATH}'.")
    print("Please deploy the HierarchicalMerkleVerifier contract first.")
    exit()
except KeyError:
    print("Error: Could not find contract address for network 31337 in artifact.")
    exit()

web3.eth.default_account = web3.eth.accounts[0]
hierarchical_contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# --- DATA MANAGEMENT ---
def load_documents():
    if not os.path.exists(DOCUMENTS_DB_FILE): 
        return []
    with open(DOCUMENTS_DB_FILE, 'r') as f:
        data = json.load(f)
        documents = []
        for d in data:
            documents.append(Document(d['doc_id'], d['content'], d['province'], d['property_id']))
        return documents

def save_documents(documents):
    docs_data = [{'doc_id': d.doc_id, 'content': d.content, 'province': d.province, 'property_id': d.property_id} for d in documents]
    with open(DOCUMENTS_DB_FILE, 'w') as f: 
        json.dump(docs_data, f, indent=4)

def load_traffic_logs():
    if not os.path.exists(TRAFFIC_LOGS_FILE): 
        return []
    with open(TRAFFIC_LOGS_FILE, 'r') as f: 
        return json.load(f)

def log_verification_event(properties_verified):
    logs = load_traffic_logs()
    logs.append(tuple(sorted(properties_verified)))
    with open(TRAFFIC_LOGS_FILE, 'w') as f: 
        json.dump(logs, f, indent=4)

# --- HIERARCHICAL SYSTEM WORKFLOWS ---
def build_and_update_hierarchical_system():
    print("--- [HIERARCHICAL SYSTEM] Building complete hierarchical tree system ---")
    all_docs = load_documents()
    logs = load_traffic_logs()
    print(f"Loaded {len(all_docs)} documents and {len(logs)} traffic log entries.")
    
    # Create and build the jurisdiction tree manager
    jurisdiction_manager = JurisdictionTreeManager(all_docs, logs)
    jurisdiction_root = jurisdiction_manager.build_all_trees()
    
    # Get system information
    system_info = jurisdiction_manager.get_system_info()
    print(f"\nHierarchical system built successfully:")
    print(f"  - Jurisdiction Root: {jurisdiction_root}")
    print(f"  - Total Provinces: {system_info['total_provinces']}")
    print(f"  - Total Documents: {system_info['total_documents']}")
    
    # Update the smart contract with new roots
    print("\n--- [ON-CHAIN] Updating hierarchical roots ---")
    
    try:
        # Prepare data for contract update
        provinces = sorted(jurisdiction_manager.provinces)
        province_roots = [jurisdiction_manager.province_builders[p].merkle_root for p in provinces]
        jurisdiction_root_hex = "0x" + jurisdiction_root
        province_roots_hex = ["0x" + root for root in province_roots]
        
        # Update all roots in a single transaction
        tx_hash = hierarchical_contract.functions.updateHierarchicalRoots(
            jurisdiction_root_hex,
            provinces,
            province_roots_hex
        ).transact()
        
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Successfully updated hierarchical roots! Tx hash: {receipt.transactionHash.hex()}")
        
        return jurisdiction_manager
        
    except Exception as e:
        print(f"ERROR updating hierarchical roots: {e}")
        return None

def simulate_cross_province_verification(jurisdiction_manager, verification_request):
    """
    Simulate cross-province verification using the hierarchical system.
    
    verification_request example:
    {
        'Jakarta': ['Jakarta.PropA', 'Jakarta.PropD'],
        'Jawa_Barat': ['Jawa_Barat.PropB', 'Jawa_Barat.PropE'],
        'Sumatera_Utara': ['Sumatera_Utara.PropC']
    }
    """
    if not jurisdiction_manager:
        return
    
    print(f"\n--- [CROSS-PROVINCE BATCH VERIFICATION] ---")
    
    # Log this verification event for traffic analysis
    all_properties = []
    for province_props in verification_request.values():
        all_properties.extend(province_props)
    log_verification_event(all_properties)
    
    # Generate hierarchical proof package
    proof_package = jurisdiction_manager.verify_cross_province_batch(verification_request)
    
    # Verify locally first
    is_valid_locally, reason = jurisdiction_manager.verify_proof_package_locally(proof_package)
    
    if not is_valid_locally:
        print(f"🔴 Local hierarchical verification failed: {reason}")
        return
    
    print(f"✅ Local hierarchical verification passed!")
    
    # Now verify on-chain using the hierarchical contract
    print("\n--- [ON-CHAIN HIERARCHICAL VERIFICATION] ---")
    
    try:
        # Prepare data for on-chain verification
        province_proofs = []
        province_flags = []
        province_leaves_arrays = []
        provinces_involved = proof_package['jurisdiction_proof']['provinces_involved']
        
        for province in provinces_involved:
            province_proof_data = proof_package['province_proofs'][province]
            
            # Convert to bytes32 format
            proof_bytes = [bytes.fromhex(p) for p in province_proof_data['proof']]
            leaves_bytes = [bytes.fromhex(l) for l in province_proof_data['document_hashes']]
            
            province_proofs.append(proof_bytes)
            province_flags.append(province_proof_data['flags'])
            province_leaves_arrays.append(leaves_bytes)
        
        # Jurisdiction-level proof
        jurisdiction_proof_bytes = [bytes.fromhex(p) for p in proof_package['jurisdiction_proof']['proof']]
        jurisdiction_flags = proof_package['jurisdiction_proof']['flags']
        
        # Call the hierarchical verification function
        is_valid_onchain = hierarchical_contract.functions.verifyHierarchicalBatch(
            province_proofs,
            province_flags,
            province_leaves_arrays,
            provinces_involved,
            jurisdiction_proof_bytes,
            jurisdiction_flags
        ).call()
        
        print(f"On-chain hierarchical verification result: {is_valid_onchain}")
        
        if is_valid_onchain:
            print("🎉 HIERARCHICAL BATCH VERIFICATION SUCCESSFUL!")
            print(f"   ✓ Verified {proof_package['total_documents']} documents")
            print(f"   ✓ Across {proof_package['total_provinces']} provinces")
            print(f"   ✓ In a single on-chain transaction")
        else:
            print("❌ On-chain hierarchical verification failed")
            
    except Exception as e:
        print(f"❌ ERROR during on-chain hierarchical verification: {e}")

def simulate_single_province_verification(jurisdiction_manager, province, properties):
    """Simulate verification within a single province."""
    if not jurisdiction_manager:
        return
        
    print(f"\n--- [SINGLE PROVINCE VERIFICATION] {province} ---")
    
    # This is effectively the same as cross-province with just one province
    verification_request = {province: properties}
    simulate_cross_province_verification(jurisdiction_manager, verification_request)

def initialize_hierarchical_genesis_documents():
    print("Initializing hierarchical genesis documents with Indonesian provinces...")
    docs = [
        # Jakarta properties (high-value commercial)
        Document("deed_jkt_001", "Commercial property deed in Menteng business district", "Jakarta", "PropA"),
        Document("title_jkt_001", "Commercial property title in Menteng business district", "Jakarta", "PropA"),
        Document("permit_jkt_001", "Building permit for Menteng commercial property", "Jakarta", "PropA"),
        Document("deed_jkt_002", "Luxury apartment deed in Kemang", "Jakarta", "PropD"),
        Document("title_jkt_002", "Luxury apartment title in Kemang", "Jakarta", "PropD"),
        
        # Jawa Barat properties (mixed residential/industrial)
        Document("deed_jabar_001", "Residential property deed in Bandung", "Jawa_Barat", "PropB"),
        Document("title_jabar_001", "Residential property title in Bandung", "Jawa_Barat", "PropB"),
        Document("survey_jabar_001", "Land survey for Bandung residential property", "Jawa_Barat", "PropB"),
        Document("deed_jabar_002", "Industrial factory deed in Bekasi", "Jawa_Barat", "PropE"),
        Document("permit_jabar_002", "Industrial permit for Bekasi factory", "Jawa_Barat", "PropE"),
        
        # Sumatera Utara properties (agricultural)
        Document("deed_sumut_001", "Agricultural land deed in Medan", "Sumatera_Utara", "PropC"),
        Document("survey_sumut_001", "Agricultural land survey in Medan", "Sumatera_Utara", "PropC"),
        
        # Bali properties (tourism/hospitality)
        Document("deed_bali_001", "Resort property deed in Ubud", "Bali", "PropF"),
        Document("license_bali_001", "Tourism license for Ubud resort", "Bali", "PropF"),
        Document("deed_bali_002", "Beachfront property deed in Sanur", "Bali", "PropG"),
    ]
    
    save_documents(docs)
    if os.path.exists(TRAFFIC_LOGS_FILE): 
        os.remove(TRAFFIC_LOGS_FILE)
    print(f"Genesis documents saved: {len(docs)} documents across multiple provinces.")

if __name__ == "__main__":
    print("=== HIERARCHICAL MERKLE TREE SYSTEM FOR INDONESIAN PROVINCES ===")
    print("--- Verifying Hardhat Connection and Contract Setup ---")
    print(f"Connected to blockchain: {web3.is_connected()}")
    print(f"Contract address loaded: {hierarchical_contract.address}")
    print(f"Default account: {web3.eth.default_account}")
    
    if not os.path.exists(DOCUMENTS_DB_FILE):
        initialize_hierarchical_genesis_documents()
    
    print("\n\n==================== BUILDING HIERARCHICAL SYSTEM ====================")
    jurisdiction_manager = build_and_update_hierarchical_system()
    
    if jurisdiction_manager:
        print("\n\n==================== TESTING CROSS-PROVINCE VERIFICATION ====================")
        
        # Test 1: Cross-province verification (your thesis scenario)
        cross_province_request = {
            'Jakarta': ['Jakarta.PropA', 'Jakarta.PropD'],  # 3 documents
            'Jawa_Barat': ['Jawa_Barat.PropB', 'Jawa_Barat.PropE'],  # 4 documents  
            'Sumatera_Utara': ['Sumatera_Utara.PropC']  # 2 documents
        }
        simulate_cross_province_verification(jurisdiction_manager, cross_province_request)
        
        # Test 2: Single province verification
        simulate_single_province_verification(jurisdiction_manager, 'Bali', ['Bali.PropF', 'Bali.PropG'])
        
        # Test 3: Large cross-province batch (like your 100-document example)
        large_batch_request = {
            'Jakarta': ['Jakarta.PropA', 'Jakarta.PropD'],
            'Jawa_Barat': ['Jawa_Barat.PropB', 'Jawa_Barat.PropE'],
            'Sumatera_Utara': ['Sumatera_Utara.PropC'],
            'Bali': ['Bali.PropF', 'Bali.PropG']
        }
        print("\n\n==================== LARGE BATCH CROSS-PROVINCE TEST ====================")
        simulate_cross_province_verification(jurisdiction_manager, large_batch_request)

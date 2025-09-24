import json
import os
import subprocess
from web3 import Web3
from optimized_tree_builder import HierarchicalTreeBuilder, Document

# --- CONFIGURATION ---
HARDHAT_URL = "http://127.0.0.1:8545"
ARTIFACT_PATH = '../../artifacts/contracts/MerkleVerifier.sol/MerkleVerifier.json'
DOCUMENTS_DB_FILE = 'documents.json'
TRAFFIC_LOGS_FILE = 'traffic_logs.json'

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
    print("Please run the `npx hardhat run scripts/deploy.ts` command first.")
    exit()
except KeyError:
    print("Error: Could not find contract address for network 31337 in artifact.")
    exit()

web3.eth.default_account = web3.eth.accounts[0]
verifier_contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# --- DATA MANAGEMENT ---
def load_documents():
    if not os.path.exists(DOCUMENTS_DB_FILE): return []
    with open(DOCUMENTS_DB_FILE, 'r') as f:
        data = json.load(f)
        documents = []
        for d in data:
            # Handle both old format (property_id) and new format (province + property_id)
            if 'province' in d:
                documents.append(Document(d['doc_id'], d['content'], d['province'], d['property_id']))
            else:
                # Default to Jakarta for backward compatibility
                documents.append(Document(d['doc_id'], d['content'], 'Jakarta', d['property_id']))
        return documents

def save_documents(documents):
    docs_data = [{'doc_id': d.doc_id, 'content': d.content, 'province': d.province, 'property_id': d.property_id} for d in documents]
    with open(DOCUMENTS_DB_FILE, 'w') as f: json.dump(docs_data, f, indent=4)

def load_traffic_logs():
    if not os.path.exists(TRAFFIC_LOGS_FILE): return []
    with open(TRAFFIC_LOGS_FILE, 'r') as f: return json.load(f)

def log_verification_event(properties_verified):
    logs = load_traffic_logs()
    logs.append(tuple(sorted(properties_verified)))
    with open(TRAFFIC_LOGS_FILE, 'w') as f: json.dump(logs, f, indent=4)

# --- MAIN WORKFLOWS ---
def run_off_chain_build_and_update():
    print("--- [SYSTEM] Starting periodic build process... ---")
    all_docs = load_documents()
    logs = load_traffic_logs()
    print(f"Loaded {len(all_docs)} documents and {len(logs)} log entries.")

    builder = HierarchicalTreeBuilder(all_docs, logs)
    new_merkle_root = builder.build()
    
    print(f"-> New Merkle Root calculated: 0x{new_merkle_root}")
    print("\n--- [ON-CHAIN] Sending transaction to update Merkle Root... ---")
    root_hex = "0x" + new_merkle_root
    
    try:
        tx_hash = verifier_contract.functions.updateRoot(root_hex).transact()
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"-> Successfully updated root! Tx hash: {receipt.transactionHash.hex()}")
        return builder
    except Exception as e:
        print(f"-> ERROR updating root: {e}")
        return None

def simulate_client_verification(builder, properties_to_verify):
    if not builder: return

    print(f"\n--- [CLIENT SIMULATION] Starting BATCH VERIFICATION for properties: {properties_to_verify} ---")
    log_verification_event(properties_to_verify)
    
    leaves_to_prove_hex = []
    for prop_id in properties_to_verify:
        if prop_id in builder.property_clusters:
            leaves_to_prove_hex.extend(builder.property_clusters[prop_id].get_leaf_hashes_hex())
    
    if not leaves_to_prove_hex:
        print("-> No documents found.")
        return

    leaves_to_prove_hex.sort(key=lambda leaf: builder.ordered_leaves_hex.index(leaf))
    
    print(f"-> Found and sorted {len(leaves_to_prove_hex)} total documents. Generating a single optimized multiproof...")
    
    batched_proof_hex, proof_flags = builder.generate_batched_proof_with_flags(leaves_to_prove_hex)
    
    print(f"-> Generated proof with {len(batched_proof_hex)} nodes and {len(proof_flags)} flags.")

    # --- LOCAL VERIFICATION STEP ---
    # We call the builder's internal, compatible verifier
    reconstructed_root, reason = builder._process_multiproof(batched_proof_hex, proof_flags, leaves_to_prove_hex)
    is_valid_locally = (reconstructed_root == builder.merkle_root)
    
    print(f"\n--- [LOCAL CHECK] Python-based verification result: {is_valid_locally} ---")
    if not is_valid_locally:
        print(f"-> 🔴 Local check failed! Reason: {reason}. Aborting on-chain call.")
        return
    else:
        print("-> ✅ Local check passed! Proceeding with on-chain call.")


    print("\n--- [ON-CHAIN] Calling verifyBatch function with ONE transaction... ---")
    proof_hex_list = ["0x" + p for p in batched_proof_hex]
    leaves_hex_list = ["0x" + l for l in leaves_to_prove_hex]
    
    try:
        is_valid_onchain = verifier_contract.functions.verifyBatch(proof_hex_list, proof_flags, leaves_hex_list).call()
        print(f"-> On-chain verification result from smart contract: {is_valid_onchain}")
        if is_valid_onchain: print("-> ✅ Batch successfully verified on-chain.")
        else: print("-> ❌ Batch verification failed on-chain.")
    except Exception as e:
        print(f"-> ❌ ERROR during on-chain verification: {e}")

def simulate_province_verification(builder, province_name):
    """Verify all properties within a specific province."""
    if not builder: return
    
    print(f"\n--- [PROVINCE VERIFICATION] Verifying ALL properties in {province_name} ---")
    
    # Get all properties for this province
    properties_to_verify = []
    if province_name in builder.province_clusters:
        properties_to_verify = builder.province_clusters[province_name].get_property_ids()
    
    if not properties_to_verify:
        print(f"-> No properties found in {province_name}")
        return
    
    print(f"-> Found {len(properties_to_verify)} properties in {province_name}: {properties_to_verify}")
    
    # Use the existing verification logic
    simulate_client_verification(builder, properties_to_verify)

def initialize_genesis_documents():
    print("Initializing genesis documents with Indonesian provinces...")
    docs = [
        # Jakarta properties
        Document("deed_jkt_001", "Deed for commercial property in Menteng", "Jakarta", "PropA"),
        Document("title_jkt_001", "Title for commercial property in Menteng", "Jakarta", "PropA"),
        
        # Jawa Barat properties  
        Document("deed_jabar_001", "Deed for residential property in Bandung", "Jawa_Barat", "PropB"),
        Document("title_jabar_001", "Title for residential property in Bandung", "Jawa_Barat", "PropB"),
        Document("survey_jabar_001", "Survey for residential property in Bandung", "Jawa_Barat", "PropB"),
        
        # Sumatera Utara properties
        Document("deed_sumut_001", "Deed for agricultural land in Medan", "Sumatera_Utara", "PropC"),
        
        # Cross-province verification testing
        Document("deed_jkt_002", "Deed for apartment in Kemang", "Jakarta", "PropD"),
        Document("deed_jabar_002", "Deed for factory in Bekasi", "Jawa_Barat", "PropE"),
    ]
    save_documents(docs)
    if os.path.exists(TRAFFIC_LOGS_FILE): os.remove(TRAFFIC_LOGS_FILE)
    print("Genesis documents saved with Indonesian province hierarchy.")

if __name__ == "__main__":
    print("--- Verifying Hardhat Connection and Contract Setup ---")
    print(f"Connected to blockchain: {web3.is_connected()}")
    print(f"Contract address loaded: {verifier_contract.address}")
    print(f"Default account: {web3.eth.default_account}\n")

    if not os.path.exists(DOCUMENTS_DB_FILE):
        initialize_genesis_documents()

    print("\n\n==================== DAY 1 - PROVINCE-BASED TESTING ====================")
    builder = run_off_chain_build_and_update()
    
    if builder:
        # Test intra-province verification (Jakarta)
        simulate_client_verification(builder, ["Jakarta.PropA", "Jakarta.PropD"])
        
        # Test single province verification (Jawa Barat)
        simulate_client_verification(builder, ["Jawa_Barat.PropB"])
        
        # Test cross-province verification
        simulate_client_verification(builder, ["Jakarta.PropA", "Jawa_Barat.PropB", "Sumatera_Utara.PropC"])

    print("\n\n==================== DAY 2 - MIXED PROVINCE PATTERNS ====================")
    builder = run_off_chain_build_and_update()
    
    if builder:
        # Test frequent cross-province pair
        simulate_client_verification(builder, ["Jakarta.PropA", "Jawa_Barat.PropE"])
        
        # Test all Jawa Barat properties
        simulate_client_verification(builder, ["Jawa_Barat.PropB", "Jawa_Barat.PropE"])
        
        # Test province-level batch verification
        simulate_province_verification(builder, "Jakarta")
        simulate_province_verification(builder, "Jawa_Barat")


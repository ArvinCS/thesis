#!/usr/bin/env python3
"""
Cross-Province Benchmark Suite
Tests the hierarchical Merkle tree system in scenarios where it truly shines:
- Multiple provinces in single verification
- Cross-jurisdiction document verification
- Large-scale multi-province scenarios
"""

from benchmark_suite import ComprehensiveBenchmarkSuite
from large_scale_generator import LargeScaleDocumentGenerator, RealisticTrafficGenerator
from web3 import Web3
import time
import json

class CrossProvinceBenchmarkSuite:
    """Specialized benchmark suite for cross-province scenarios."""
    
    def __init__(self, web3_instance=None):
        self.web3 = web3_instance
        self.profiler = None
        
        if web3_instance:
            self._setup_contracts()
    
    def _setup_contracts(self):
        """Setup smart contracts for gas cost analysis."""
        try:
            import json
            
            # Load stateless hierarchical contract
            with open('../../artifacts/contracts/HierarchicalMerkleVerifier.sol/HierarchicalMerkleVerifier.json', 'r') as f:
                hierarchical_artifact = json.load(f)
            
            # Use the deployed stateless contract address directly
            hierarchical_address = "0x9D40c21ff3BD14d671BB7c00Dcc1aDD0a4C9Bd41"
            self.hierarchical_contract = self.web3.eth.contract(
                address=hierarchical_address,
                abi=hierarchical_artifact['abi']
            )
            
            # Load traditional contract
            with open('../../artifacts/contracts/MerkleVerifier.sol/MerkleVerifier.json', 'r') as f:
                traditional_artifact = json.load(f)
            
            traditional_address = traditional_artifact['networks']['31337']['address']
            self.traditional_contract = self.web3.eth.contract(
                address=traditional_address,
                abi=traditional_artifact['abi']
            )
            
            print("✅ Smart contracts loaded for cross-province testing")
            
        except Exception as e:
            print(f"⚠️ Could not load smart contracts: {e}")
            self.hierarchical_contract = None
            self.traditional_contract = None
    
    def create_cross_province_scenarios(self, properties_by_province):
        """Create verification scenarios that span multiple provinces."""
        scenarios = {}
        
        # Get all provinces
        provinces = list(properties_by_province.keys())
        
        if len(provinces) < 2:
            print("⚠️ Need at least 2 provinces for cross-province testing")
            return scenarios
        
        # Scenario 1: Small Cross-Province (2 provinces, few documents)
        if len(provinces) >= 2:
            scenario_name = "small_cross_province_2_provinces"
            scenario = {}
            for i, province in enumerate(provinces[:2]):
                # Take 1-2 properties from each province
                province_props = properties_by_province[province][:2]
                scenario[province] = [prop['full_id'] for prop in province_props]
            scenarios[scenario_name] = scenario
        
        # Scenario 2: Medium Cross-Province (3 provinces, moderate documents)
        if len(provinces) >= 3:
            scenario_name = "medium_cross_province_3_provinces"
            scenario = {}
            for i, province in enumerate(provinces[:3]):
                # Take 2-3 properties from each province
                province_props = properties_by_province[province][:3]
                scenario[province] = [prop['full_id'] for prop in province_props]
            scenarios[scenario_name] = scenario
        
        # Scenario 3: Large Cross-Province (4 provinces, many documents)
        if len(provinces) >= 4:
            scenario_name = "large_cross_province_4_provinces"
            scenario = {}
            for i, province in enumerate(provinces[:4]):
                # Take 3-4 properties from each province
                province_props = properties_by_province[province][:4]
                scenario[province] = [prop['full_id'] for prop in province_props]
            scenarios[scenario_name] = scenario
        
        # Scenario 4: Massive Cross-Province (6 provinces, many documents)
        if len(provinces) >= 6:
            scenario_name = "massive_cross_province_6_provinces"
            scenario = {}
            for i, province in enumerate(provinces[:6]):
                # Take 2-3 properties from each province
                province_props = properties_by_province[province][:3]
                scenario[province] = [prop['full_id'] for prop in province_props]
            scenarios[scenario_name] = scenario
        
        # Scenario 5: Extreme Cross-Province (8+ provinces, distributed documents)
        if len(provinces) >= 8:
            scenario_name = "extreme_cross_province_8_provinces"
            scenario = {}
            for i, province in enumerate(provinces[:8]):
                # Take 1-2 properties from each province
                province_props = properties_by_province[province][:2]
                scenario[province] = [prop['full_id'] for prop in province_props]
            scenarios[scenario_name] = scenario
        
        # Scenario 6: All Provinces (if manageable)
        if len(provinces) <= 10:  # Only if not too many provinces
            scenario_name = "all_provinces_cross_verification"
            scenario = {}
            for province in provinces:
                # Take 1 property from each province
                province_props = properties_by_province[province][:1]
                scenario[province] = [prop['full_id'] for prop in province_props]
            scenarios[scenario_name] = scenario
        
        return scenarios
    
    def benchmark_cross_province_scenarios(self, jurisdiction_manager, traditional_multiproof_builder, verification_scenarios):
        """Benchmark cross-province scenarios where hierarchical approach shines."""
        print("\n=== CROSS-PROVINCE BENCHMARK SUITE ===")
        print("🎯 Testing scenarios where hierarchical approach should excel")
        
        results = {}
        
        for scenario_name, verification_request in verification_scenarios.items():
            print(f"\n🔄 Testing: {scenario_name}")
            print(f"   Provinces involved: {list(verification_request.keys())}")
            print(f"   Total provinces: {len(verification_request)}")
            
            # Count total documents
            total_docs = 0
            for province, properties in verification_request.items():
                if province in jurisdiction_manager.province_builders:
                    builder = jurisdiction_manager.province_builders[province]
                    for prop_id in properties:
                        if prop_id in builder.property_clusters:
                            prop_cluster = builder.property_clusters[prop_id]
                            total_docs += len(prop_cluster.documents)
            print(f"   Total documents: {total_docs}")
            
            # Update stateless contract state for hierarchical verification
            if self.hierarchical_contract:
                try:
                    print(f"      🔄 Updating stateless contract with jurisdiction root...")
                    self.hierarchical_contract.functions.updateJurisdictionRoot(
                        bytes.fromhex(jurisdiction_manager.jurisdiction_root)
                    ).transact()
                    
                    print("      ✅ Stateless contract jurisdiction root updated for cross-province testing")
                except Exception as e:
                    print(f"      ⚠️ Stateless contract state update failed: {e}")
                    print(f"      ⚠️ Continuing with local verification only")
            else:
                print("  ⚠️ No stateless hierarchical contract available - skipping state update")
            
            # Test Hierarchical Approach
            hierarchical_gas_used = 0
            hierarchical_verification_time = 0
            hierarchical_success = False
            
            try:
                # Generate hierarchical proof
                hierarchical_proof = jurisdiction_manager.verify_cross_province_batch(verification_request)
                
                # Check if proof was generated successfully
                if hierarchical_proof and hierarchical_proof.get('total_documents', 0) > 0:
                    # Prepare stateless hierarchical proof data
                    # CRITICAL: Use the same ordering as the jurisdiction proof generation (alphabetical)
                    provinces_involved = hierarchical_proof['jurisdiction_proof']['provinces_involved']
                    claimed_province_roots = []
                    province_proofs = []
                    province_flags = []
                    province_leaves_arrays = []
                    
                    for province in provinces_involved:
                        if province in hierarchical_proof['province_proofs']:
                            # Get the province root from the jurisdiction manager
                            if province in jurisdiction_manager.province_builders:
                                builder = jurisdiction_manager.province_builders[province]
                                claimed_province_roots.append(bytes.fromhex(builder.merkle_root))
                            
                            province_proof_data = hierarchical_proof['province_proofs'][province]
                            province_proofs.append([bytes.fromhex(p) for p in province_proof_data['proof']])
                            province_flags.append(province_proof_data['flags'])
                            province_leaves_arrays.append([bytes.fromhex(l) for l in province_proof_data['document_hashes']])
                    
                    jurisdiction_proof_bytes = [bytes.fromhex(p) for p in hierarchical_proof['jurisdiction_proof']['proof']]
                    jurisdiction_flags = hierarchical_proof['jurisdiction_proof']['flags']
                    
                    # Test local verification first
                    is_valid_locally, reason = jurisdiction_manager.verify_proof_package_locally(hierarchical_proof)
                    if is_valid_locally:
                        print(f"  ✅ Hierarchical proof generation successful: {hierarchical_proof.get('total_documents', 0)} docs across {hierarchical_proof.get('total_provinces', 0)} provinces")
                        
                        # Now do REAL on-chain verification
                        if self.hierarchical_contract:
                            try:
                                print(f"      🔄 Sending stateless hierarchical verification to contract...")
                                start_time = time.time()
                                tx_hash = self.hierarchical_contract.functions.verifyHierarchicalBatch(
                                    claimed_province_roots,
                                    jurisdiction_proof_bytes,
                                    jurisdiction_flags,
                                    province_proofs,
                                    province_flags,
                                    province_leaves_arrays
                                ).transact()
                                
                                receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                                hierarchical_verification_time = time.time() - start_time
                                hierarchical_gas_used = receipt.gasUsed
                                hierarchical_success = receipt.status == 1
                                
                                if hierarchical_success:
                                    print(f"      ✅ Hierarchical on-chain verification: {hierarchical_gas_used:,} gas in {hierarchical_verification_time:.4f}s")
                                    print(f"      🎯 SINGLE TRANSACTION for cross-province verification!")
                                else:
                                    print(f"      ❌ Hierarchical on-chain verification failed")
                                    hierarchical_success = False
                                    
                            except Exception as onchain_error:
                                print(f"      ❌ Hierarchical on-chain verification failed: {onchain_error}")
                                hierarchical_success = False
                        else:
                            print(f"      ⚠️ No hierarchical contract available - using local verification only")
                            hierarchical_success = True
                            hierarchical_verification_time = 0.001  # Placeholder
                            hierarchical_gas_used = 50000  # Placeholder
                    else:
                        print(f"  ❌ Hierarchical local verification failed: {reason}")
                        hierarchical_success = False
                        
                else:
                    print(f"  ❌ Hierarchical proof generation failed")
                    
            except Exception as e:
                print(f"  ❌ Hierarchical approach failed: {e}")
            
            # Test Traditional Multiproof Approach
            traditional_gas_used = 0
            traditional_verification_time = 0
            traditional_success = False
            
            try:
                # Collect all documents for traditional approach
                all_doc_hashes = []
                for province, properties in verification_request.items():
                    if province in jurisdiction_manager.province_builders:
                        builder = jurisdiction_manager.province_builders[province]
                        for prop_id in properties:
                            if prop_id in builder.property_clusters:
                                prop_cluster = builder.property_clusters[prop_id]
                                all_doc_hashes.extend(prop_cluster.get_leaf_hashes_hex())
                
                # Remove duplicates
                all_doc_hashes = list(dict.fromkeys(all_doc_hashes))
                
                # Generate traditional multiproof
                traditional_proof, traditional_flags = traditional_multiproof_builder.generate_proof_for_documents(all_doc_hashes)
                traditional_proof_bytes = [bytes.fromhex(p) for p in traditional_proof]
                traditional_leaves_bytes = [bytes.fromhex(l) for l in reversed(all_doc_hashes)]
                
                # Measure traditional verification
                if self.traditional_contract:
                    start_time = time.time()
                    tx_hash = self.traditional_contract.functions.verifyBatch(
                        traditional_proof_bytes,
                        traditional_flags,
                        traditional_leaves_bytes
                    ).transact()
                    
                    receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                    traditional_verification_time = time.time() - start_time
                    traditional_gas_used = receipt.gasUsed
                    traditional_success = receipt.status == 1
                    
                    if traditional_success:
                        print(f"  ✅ Traditional: {traditional_gas_used:,} gas in {traditional_verification_time:.4f}s")
                    else:
                        print(f"  ❌ Traditional verification failed")
                else:
                    print(f"  ⚠️ No contract available for traditional testing")
                    
            except Exception as e:
                print(f"  ❌ Traditional approach failed: {e}")
            
            # Calculate savings
            if hierarchical_success and traditional_success and traditional_gas_used > 0:
                gas_savings = traditional_gas_used - hierarchical_gas_used
                savings_percent = (gas_savings / traditional_gas_used) * 100
                
                print(f"  📊 Cross-Province Results:")
                print(f"     Hierarchical: {hierarchical_gas_used:,} gas")
                print(f"     Traditional: {traditional_gas_used:,} gas")
                print(f"     Savings: {gas_savings:,} gas ({savings_percent:.1f}%)")
                
                if savings_percent > 0:
                    print(f"  🎉 Hierarchical approach WINS by {savings_percent:.1f}%!")
                else:
                    print(f"  ⚠️ Traditional approach still more efficient by {abs(savings_percent):.1f}%")
            
            # Store results
            results[scenario_name] = {
                'provinces_involved': len(verification_request),
                'total_documents': total_docs,
                'hierarchical_gas_used': hierarchical_gas_used,
                'hierarchical_verification_time': hierarchical_verification_time,
                'hierarchical_success': hierarchical_success,
                'traditional_gas_used': traditional_gas_used,
                'traditional_verification_time': traditional_verification_time,
                'traditional_success': traditional_success,
                'gas_savings': traditional_gas_used - hierarchical_gas_used if traditional_success and hierarchical_success else 0,
                'savings_percent': ((traditional_gas_used - hierarchical_gas_used) / traditional_gas_used) * 100 if traditional_success and hierarchical_success and traditional_gas_used > 0 else 0
            }
        
        return results

def main():
    """Run cross-province benchmark suite."""
    print("🚀 Starting Cross-Province Benchmark Suite")
    print("🎯 Testing scenarios where hierarchical approach should excel")
    
    # Setup Web3 connection
    web3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
    if web3.is_connected():
        web3.eth.default_account = web3.eth.accounts[0]
        print('✅ Connected to Hardhat for cross-province testing')
    else:
        print('❌ No Web3 connection - cannot test cross-province scenarios')
        return
    
    # Generate documents with multiple provinces
    print('\n🔄 Generating documents across multiple provinces...')
    generator = LargeScaleDocumentGenerator(target_document_count=2000, seed=42)
    documents = generator.generate_documents()
    
    print(f'✅ Generated {len(documents)} documents across {len(generator.properties_by_province)} provinces')
    for province, props in generator.properties_by_province.items():
        print(f'   {province}: {len(props)} properties')
    
    # Generate traffic logs
    traffic_generator = RealisticTrafficGenerator(documents, generator.properties_by_province)
    traffic_logs = traffic_generator.generate_traffic_logs(50)
    print(f'✅ Generated {len(traffic_logs)} traffic events')
    
    # Build trees
    print('\n🔄 Building hierarchical tree system...')
    benchmark_suite = ComprehensiveBenchmarkSuite(web3)
    construction_results, jurisdiction_manager, traditional_multiproof_builder, traditional_single_proof_builder = benchmark_suite.benchmark_tree_construction(documents, traffic_logs)
    
    # Create cross-province scenarios
    print('\n🔄 Creating cross-province test scenarios...')
    cross_province_suite = CrossProvinceBenchmarkSuite(web3)
    cross_province_scenarios = cross_province_suite.create_cross_province_scenarios(generator.properties_by_province)
    
    print(f'✅ Created {len(cross_province_scenarios)} cross-province scenarios:')
    for scenario_name, scenario in cross_province_scenarios.items():
        total_docs = sum(len(props) for props in scenario.values())
        print(f'   {scenario_name}: {len(scenario)} provinces, ~{total_docs} properties')
    
    # Run cross-province benchmarks
    cross_province_results = cross_province_suite.benchmark_cross_province_scenarios(
        jurisdiction_manager, traditional_multiproof_builder, cross_province_scenarios
    )
    
    # Summary
    print('\n🎉 Cross-Province Benchmark Complete!')
    print('\n📊 SUMMARY:')
    
    hierarchical_wins = 0
    traditional_wins = 0
    
    for scenario_name, result in cross_province_results.items():
        if result['hierarchical_success'] and result['traditional_success']:
            if result['savings_percent'] > 0:
                hierarchical_wins += 1
                print(f'✅ {scenario_name}: Hierarchical WINS by {result["savings_percent"]:.1f}%')
            else:
                traditional_wins += 1
                print(f'❌ {scenario_name}: Traditional still better by {abs(result["savings_percent"]):.1f}%')
        else:
            print(f'⚠️ {scenario_name}: Test failed')
    
    print(f'\n🏆 FINAL SCORE:')
    print(f'   Hierarchical wins: {hierarchical_wins}')
    print(f'   Traditional wins: {traditional_wins}')
    
    if hierarchical_wins > traditional_wins:
        print('🎉 HIERARCHICAL APPROACH DOMINATES in cross-province scenarios!')
    elif hierarchical_wins == traditional_wins:
        print('🤝 TIE - Both approaches perform similarly')
    else:
        print('⚠️ Traditional approach still more efficient even in cross-province scenarios')

if __name__ == "__main__":
    main()

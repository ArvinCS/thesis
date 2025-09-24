import { ethers } from "hardhat";

async function main() {
  console.log("Deploying all Merkle Tree Verifier contracts...");

  // Get the ContractFactory and Signers here.
  const [deployer] = await ethers.getSigners();
  console.log("Deploying contracts with the account:", deployer.address);
  console.log("Account balance:", (await deployer.provider.getBalance(deployer.address)).toString());

  const contracts = {};

  // 1. Deploy HierarchicalMerkleVerifier
  console.log("\nDeploying HierarchicalMerkleVerifier...");
  const HierarchicalMerkleVerifier = await ethers.getContractFactory("HierarchicalMerkleVerifier");
  const hierarchicalVerifier = await HierarchicalMerkleVerifier.deploy(deployer.address);
  await hierarchicalVerifier.waitForDeployment();
  const hierarchicalAddress = await hierarchicalVerifier.getAddress();
  contracts.hierarchical = hierarchicalAddress;
  console.log("✅ HierarchicalMerkleVerifier deployed to:", hierarchicalAddress);

  // 2. Deploy TraditionalMultiproofMerkleVerifier (Traditional Multiproof)
  console.log("\nDeploying TraditionalMultiproofMerkleVerifier (Traditional Multiproof)...");
  const TraditionalMultiproofMerkleVerifier = await ethers.getContractFactory("TraditionalMultiproofMerkleVerifier");
  const merkleVerifier = await TraditionalMultiproofMerkleVerifier.deploy(deployer.address);
  await merkleVerifier.waitForDeployment();
  const merkleAddress = await merkleVerifier.getAddress();
  contracts.traditional_multiproof = merkleAddress;
  console.log("✅ TraditionalMultiproofMerkleVerifier deployed to:", merkleAddress);

  // 3. Deploy TraditionalSingleProofMerkleVerifier (Traditional Single Proof)
  console.log("\nDeploying TraditionalSingleProofMerkleVerifier (Traditional Single Proof)...");
  const TraditionalSingleProofMerkleVerifier = await ethers.getContractFactory("TraditionalSingleproofMerkleVerifier");
  const singleProofVerifier = await TraditionalSingleProofMerkleVerifier.deploy(deployer.address);
  await singleProofVerifier.waitForDeployment();
  const singleProofAddress = await singleProofVerifier.getAddress();
  contracts.traditional_single_proof = singleProofAddress;
  console.log("✅ TraditionalSingleproofMerkleVerifier deployed to:", singleProofAddress);

  // 3. Deploy TraditionalHuffmanMerkleVerifier (Traditional Single Proof)
  console.log("\nDeploying TraditionalHuffmanMerkleVerifier (Traditional Single Proof)...");
  const TraditionalHuffmanMerkleVerifier = await ethers.getContractFactory("TraditionalHuffmanMerkleVerifier");
  const huffmanVerifier = await TraditionalHuffmanMerkleVerifier.deploy(deployer.address);
  await huffmanVerifier.waitForDeployment();
  const huffmanAddress = await huffmanVerifier.getAddress();
  contracts.traditional_huffman = huffmanAddress;
  console.log("✅ TraditionalHuffmanMerkleVerifier deployed to:", huffmanAddress);

  // 3. Deploy ClusteredFlatBasicVerifier (Traditional Single Proof)
  console.log("\nDeploying ClusteredFlatBasicVerifier (Traditional Single Proof)...");
  const ClusteredFlatBasicVerifier = await ethers.getContractFactory("ClusteredFlatBasicVerifier");
  const clusteredBasicVerifier = await ClusteredFlatBasicVerifier.deploy(deployer.address);
  await clusteredBasicVerifier.waitForDeployment();
  const clusteredBasicAddress = await clusteredBasicVerifier.getAddress();
  contracts.clustered_flat_basic = clusteredBasicAddress;
  console.log("✅ ClusteredFlatBasicVerifier deployed to:", clusteredBasicAddress);

  // 3. Deploy ClusteredFlatVerifier (Traditional Single Proof)
  console.log("\nDeploying ClusteredFlatVerifier (Traditional Single Proof)...");
  const ClusteredFlatVerifier = await ethers.getContractFactory("ClusteredFlatVerifier");
  const clusteredVerifier = await ClusteredFlatVerifier.deploy(deployer.address);
  await clusteredVerifier.waitForDeployment();
  const clusteredAddress = await clusteredVerifier.getAddress();
  contracts.clustered_flat = clusteredAddress;
  console.log("✅ ClusteredFlatVerifier deployed to:", clusteredAddress);

  // Save contract addresses to artifacts
  console.log("\nSaving contract addresses to artifacts...");
  const fs = require('fs');
  const path = require('path');
  
  const contractConfigs = [
    {
      name: 'HierarchicalMerkleVerifier',
      address: hierarchicalAddress,
      contract: hierarchicalVerifier
    },
    {
      name: 'TraditionalMultiproofMerkleVerifier',
      address: merkleAddress,
      contract: merkleVerifier
    },
    {
      name: 'TraditionalSingleproofMerkleVerifier',
      address: singleProofAddress,
      contract: singleProofVerifier
    },
    {
      name: 'ClusteredFlatBasicVerifier',
      address: clusteredBasicAddress,
      contract: clusteredBasicVerifier
    },
    {
      name: 'ClusteredFlatVerifier',
      address: clusteredAddress,
      contract: clusteredVerifier
    },
    {
      name: 'TraditionalHuffmanMerkleVerifier',
      address: huffmanAddress,
      contract: huffmanVerifier
    }
  ];

  for (const config of contractConfigs) {
    const artifactPath = path.join(__dirname, `../artifacts/contracts/${config.name}.sol/${config.name}.json`);
    
    if (fs.existsSync(artifactPath)) {
      const artifact = JSON.parse(fs.readFileSync(artifactPath, 'utf8'));
      
      // Add network information
      if (!artifact.networks) {
        artifact.networks = {};
      }
      
      artifact.networks['31337'] = {
        address: config.address,
        transactionHash: config.contract.deploymentTransaction()?.hash
      };
      
      // Write updated artifact back
      fs.writeFileSync(artifactPath, JSON.stringify(artifact, null, 2));
      console.log(`✅ ${config.name} address saved to artifact file.`);
    } else {
      console.log(`⚠️ Warning: ${config.name} artifact file not found.`);
    }
  }

  // Print deployment summary
  console.log("\n" + "=".repeat(60));
  console.log("🎉 ALL CONTRACTS DEPLOYED SUCCESSFULLY!");
  console.log("=".repeat(60));
  console.log(`Network: Hardhat Local (31337)`);
  console.log(`Deployer: ${deployer.address}`);
  console.log(`Account Balance: ${(await deployer.provider.getBalance(deployer.address)).toString()} wei`);
  console.log("\nContract Addresses:");
  console.log(`1. HierarchicalMerkleVerifier:     ${hierarchicalAddress}`);
  console.log(`2. TraditionalMultiproofMerkleVerifier:    ${merkleAddress}`);
  console.log(`3. TraditionalSingleProofMerkleVerifier:      ${singleProofAddress}`);
  console.log(`4. ClusteredFlatBasicVerifier:      ${clusteredBasicAddress}`);
  console.log(`5. ClusteredFlatVerifier:      ${clusteredAddress}`);
  console.log(`6. TraditionalHuffmanMerkleVerifier:      ${huffmanAddress}`);
  
  console.log("\nNext Steps:");
  console.log("1. Run: python offchain/python/benchmark_suite.py");
  console.log("2. Test three-way comparison with gas cost analysis");
  console.log("3. Verify all contracts are working correctly");
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

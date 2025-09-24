import { ethers } from "hardhat";

async function main() {
  console.log("Deploying HierarchicalMerkleVerifier contract...");

  // Get the ContractFactory and Signers here.
  const [deployer] = await ethers.getSigners();
  console.log("Deploying contracts with the account:", deployer.address);

  // Deploy the HierarchicalMerkleVerifier contract
  const HierarchicalMerkleVerifier = await ethers.getContractFactory("HierarchicalMerkleVerifier");
  const hierarchicalVerifier = await HierarchicalMerkleVerifier.deploy(deployer.address);

  await hierarchicalVerifier.waitForDeployment();

  const contractAddress = await hierarchicalVerifier.getAddress();
  console.log("HierarchicalMerkleVerifier deployed to:", contractAddress);
  console.log("Initial owner:", deployer.address);

  // Save the contract address and ABI to artifacts for the Python client
  const fs = require('fs');
  const path = require('path');
  
  const artifactPath = path.join(__dirname, '../artifacts/contracts/HierarchicalMerkleVerifier.sol/HierarchicalMerkleVerifier.json');
  
  if (fs.existsSync(artifactPath)) {
    const artifact = JSON.parse(fs.readFileSync(artifactPath, 'utf8'));
    
    // Add network information
    if (!artifact.networks) {
      artifact.networks = {};
    }
    
    artifact.networks['31337'] = {
      address: contractAddress,
      transactionHash: hierarchicalVerifier.deploymentTransaction()?.hash
    };
    
    // Write updated artifact back
    fs.writeFileSync(artifactPath, JSON.stringify(artifact, null, 2));
    console.log("Contract address saved to artifact file.");
  } else {
    console.log("Warning: Artifact file not found. You may need to compile first.");
  }

  console.log("\n=== Deployment Summary ===");
  console.log(`Contract: HierarchicalMerkleVerifier`);
  console.log(`Address: ${contractAddress}`);
  console.log(`Owner: ${deployer.address}`);
  console.log(`Network: Hardhat Local (31337)`);
  console.log("\nNext steps:");
  console.log("1. Run: python offchain/python/hierarchical_main.py");
  console.log("2. Test cross-province verification");
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

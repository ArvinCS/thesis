import { ethers } from "hardhat";

async function main() {
  console.log("Deploying SingleProofMerkleVerifier contract...");

  // Get the ContractFactory and Signers here.
  const [deployer] = await ethers.getSigners();
  console.log("Deploying contracts with the account:", deployer.address);

  // Deploy the SingleProofMerkleVerifier contract
  const SingleProofMerkleVerifier = await ethers.getContractFactory("SingleProofMerkleVerifier");
  const singleProofVerifier = await SingleProofMerkleVerifier.deploy(deployer.address);

  await singleProofVerifier.waitForDeployment();

  const contractAddress = await singleProofVerifier.getAddress();
  console.log("SingleProofMerkleVerifier deployed to:", contractAddress);
  console.log("Initial owner:", deployer.address);

  // Save the contract address and ABI to artifacts for the Python client
  const fs = require('fs');
  const path = require('path');
  
  const artifactPath = path.join(__dirname, '../artifacts/contracts/SingleProofMerkleVerifier.sol/SingleProofMerkleVerifier.json');
  
  if (fs.existsSync(artifactPath)) {
    const artifact = JSON.parse(fs.readFileSync(artifactPath, 'utf8'));
    
    // Add network information
    if (!artifact.networks) {
      artifact.networks = {};
    }
    
    artifact.networks['31337'] = {
      address: contractAddress,
      transactionHash: singleProofVerifier.deploymentTransaction()?.hash
    };
    
    // Write updated artifact back
    fs.writeFileSync(artifactPath, JSON.stringify(artifact, null, 2));
    console.log("Contract address saved to artifact file.");
  } else {
    console.log("Warning: Artifact file not found. You may need to compile first.");
  }

  console.log("\n=== Deployment Summary ===");
  console.log(`Contract: SingleProofMerkleVerifier`);
  console.log(`Address: ${contractAddress}`);
  console.log(`Owner: ${deployer.address}`);
  console.log(`Network: Hardhat Local (31337)`);
  console.log("\nNext steps:");
  console.log("1. Run: python offchain/python/benchmark_suite.py");
  console.log("2. Test three-way comparison (Hierarchical, Traditional Multiproof, Traditional Single Proof)");
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

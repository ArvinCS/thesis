import { ethers } from "hardhat";

async function main() {
  console.log("Deploying ClusteredFlatVerifier...");

  const ClusteredFlatVerifier = await ethers.getContractFactory("ClusteredFlatVerifier");
  const verifier = await ClusteredFlatVerifier.deploy();

  await verifier.waitForDeployment();
  const address = await verifier.getAddress();

  console.log(`ClusteredFlatVerifier deployed to: ${address}`);

  // Save the contract address for artifacts
  const fs = require('fs');
  const path = require('path');
  
  // Update the artifacts with network information
  const artifactPath = path.join(__dirname, '../artifacts/contracts/ClusteredFlatVerifier.sol/ClusteredFlatVerifier.json');
  if (fs.existsSync(artifactPath)) {
    const artifact = JSON.parse(fs.readFileSync(artifactPath, 'utf8'));
    if (!artifact.networks) {
      artifact.networks = {};
    }
    artifact.networks['31337'] = {
      address: address,
      transactionHash: verifier.deploymentTransaction()?.hash || '',
    };
    fs.writeFileSync(artifactPath, JSON.stringify(artifact, null, 2));
    console.log(`✅ Contract address saved to artifacts: ${address}`);
  }

  // Test basic functionality
  console.log("Testing basic functionality...");
  
  // Test empty verification
  const emptyResult = await verifier.testClusteredFlatMultiproof(
    [], // proof
    [], // proofFlags  
    [], // leaves
    ethers.ZeroHash // root
  );
  console.log(`Empty tree verification: ${emptyResult}`);

  // Estimate gas for different scenarios
  console.log("Gas estimates:");
  const estimates = [
    { leaves: 1, proof: 0, description: "Single leaf" },
    { leaves: 10, proof: 15, description: "Small batch (10 leaves)" },
    { leaves: 50, proof: 75, description: "Medium batch (50 leaves)" },
    { leaves: 100, proof: 150, description: "Large batch (100 leaves)" },
    { leaves: 500, proof: 750, description: "Very large batch (500 leaves)" }
  ];

  for (const estimate of estimates) {
    const gasEstimate = await verifier.estimateVerificationGas(
      estimate.proof,
      estimate.leaves
    );
    console.log(`  ${estimate.description}: ~${gasEstimate.toString()} gas`);
  }

  console.log("Deployment completed successfully!");
  
  return {
    verifier,
    address
  };
}

main()
  .then((result) => {
    console.log("Deployment successful!");
    process.exit(0);
  })
  .catch((error) => {
    console.error("Deployment failed:", error);
    process.exit(1);
  });

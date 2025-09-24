import { ethers } from "hardhat";

async function main() {
  console.log("🚀 Deploying StatelessHierarchicalVerifier...");

  // Get the contract factory
  const StatelessHierarchicalVerifier = await ethers.getContractFactory("StatelessHierarchicalVerifier");

  // Get the deployer address
  const [deployer] = await ethers.getSigners();
  
  // Deploy the contract
  const statelessVerifier = await StatelessHierarchicalVerifier.deploy(deployer.address);
  await statelessVerifier.waitForDeployment();

  const address = await statelessVerifier.getAddress();
  console.log(`✅ StatelessHierarchicalVerifier deployed to: ${address}`);

  console.log(`📝 Deployer address: ${deployer.address}`);

  const fs = require('fs');
  const path = require('path');
  
  const artifactPath = path.join(__dirname, '../artifacts/contracts/StatelessHierarchicalVerifier.sol/StatelessHierarchicalVerifier.json');
  
  if (fs.existsSync(artifactPath)) {
    const artifact = JSON.parse(fs.readFileSync(artifactPath, 'utf8'));
    
    // Add network information
    if (!artifact.networks) {
      artifact.networks = {};
    }
    
    artifact.networks['31337'] = {
      address: address,
      transactionHash: statelessVerifier.deploymentTransaction()?.hash
    };
    
    // Write updated artifact back
    fs.writeFileSync(artifactPath, JSON.stringify(artifact, null, 2));
    console.log("Contract address saved to artifact file.");
  } else {
    console.log("Warning: Artifact file not found. You may need to compile first.");
  }

  // Verify the jurisdiction root is initially empty
  const initialRoot = await statelessVerifier.jurisdictionRoot();
  console.log(`🌳 Initial jurisdiction root: ${initialRoot}`);

  console.log("🎉 Deployment complete!");
  console.log("\n📋 Next steps:");
  console.log("1. Update the jurisdiction root using updateJurisdictionRoot()");
  console.log("2. Test the stateless verification with your Python scripts");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("❌ Deployment failed:", error);
    process.exit(1);
  });

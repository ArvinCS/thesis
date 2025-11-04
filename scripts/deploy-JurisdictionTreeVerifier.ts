import { ethers } from "hardhat";
import fs from "fs";
import path from "path";

async function main() {
  console.log("🏗️  Deploying JurisdictionTreeVerifier...");

  // Deploy the contract
  const JurisdictionTreeVerifier = await ethers.getContractFactory("JurisdictionTreeVerifier");
  const verifier = await JurisdictionTreeVerifier.deploy();
  
  await verifier.waitForDeployment();
  const address = await verifier.getAddress();

  console.log(`✅ JurisdictionTreeVerifier deployed to: ${address}`);

  // Verify contract capabilities
  try {
    const versionResult = await verifier.getFunction("getVersion")();
    const version = versionResult[0];
    const capabilities = versionResult[1];
    console.log(`📋 Contract version: ${version}`);
    console.log(`🔧 Capabilities: ${capabilities} (${capabilities.toString(2).padStart(8, '0')})`);
    
    // Test gas estimation
    const estimatedGas1Province = await verifier.getFunction("estimateGas")(1, 5);
    const estimatedGas3Provinces = await verifier.getFunction("estimateGas")(3, 15);
    console.log(`⛽ Estimated gas (1 province, 5 docs): ${estimatedGas1Province}`);
    console.log(`⛽ Estimated gas (3 provinces, 15 docs): ${estimatedGas3Provinces}`);
  } catch (error) {
    console.log(`⚠️  Could not test contract methods: ${error}`);
  }

  // Save contract info to artifacts
  const network = await ethers.provider.getNetwork();
  const chainId = network.chainId.toString();
  
  const artifactPath = path.join(__dirname, "../artifacts/contracts/JurisdictionTreeVerifier.sol/JurisdictionTreeVerifier.json");
  
  if (fs.existsSync(artifactPath)) {
    const artifact = JSON.parse(fs.readFileSync(artifactPath, 'utf8'));
    
    // Add network info
    if (!artifact.networks) {
      artifact.networks = {};
    }
    
    artifact.networks[chainId] = {
      address: address,
      transactionHash: verifier.deploymentTransaction()?.hash || "",
    };
    
    fs.writeFileSync(artifactPath, JSON.stringify(artifact, null, 2));
    console.log(`💾 Updated artifact with deployment info for chain ${chainId}`);
  }

  console.log("\n🎉 JurisdictionTreeVerifier deployment completed!");
  console.log("📊 Key Features:");
  console.log("   • Two-phase verification (province + jurisdiction)");
  console.log("   • Optimized single-province verification");
  console.log("   • Gas-efficient batch processing");
  console.log("   • OpenZeppelin-compatible multiproof");
  console.log("   • Event logging for analytics");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("❌ Deployment failed:", error);
    process.exit(1);
  });
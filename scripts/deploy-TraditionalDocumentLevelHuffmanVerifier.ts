import { ethers } from "hardhat";

async function main() {
  console.log("🚀 Deploying TraditionalDocumentLevelHuffmanVerifier...");
  
  const [deployer] = await ethers.getSigners();
  console.log("Deploying with account:", deployer.address);
  console.log("Account balance:", (await ethers.provider.getBalance(deployer.address)).toString());

  // Deploy the contract
  const TraditionalDocumentLevelHuffmanVerifier = await ethers.getContractFactory("TraditionalDocumentLevelHuffmanVerifier");
  const contract = await TraditionalDocumentLevelHuffmanVerifier.deploy(deployer.address);

  await contract.waitForDeployment();
  const contractAddress = await contract.getAddress();

  console.log("✅ TraditionalDocumentLevelHuffmanVerifier deployed to:", contractAddress);
  
  // Display initial state
  const merkleRoot = await contract.merkleRoot();
  const stats = await contract.getOptimizationStats();
  
  console.log("📊 Initial Contract State:");
  console.log("  Merkle Root:", merkleRoot);
  console.log("  Total Verifications:", stats.totalVerifications.toString());
  console.log("  Intra-Property Verifications:", stats.intraPropertyVerifications.toString());
  console.log("  Cross-Property Verifications:", stats.crossPropertyVerifications.toString());
  console.log("  Average Proof Size:", stats.averageProofSize.toString());
  console.log("  Last Update:", new Date(Number(stats.lastUpdateTimestamp) * 1000).toISOString());

  // Test setting an initial root
  console.log("\n🔧 Setting initial test root...");
  const testRoot = "0x1234567890123456789012345678901234567890123456789012345678901234";
  await contract.updateRoot(testRoot, "initial_deployment_test");
  
  const updatedRoot = await contract.merkleRoot();
  console.log("Updated root:", updatedRoot);
  
  console.log("\n🎯 Contract Features:");
  console.log("  ✅ Document-level Huffman optimization support");
  console.log("  ✅ Intra-property verification tracking");
  console.log("  ✅ Cross-property verification tracking");
  console.log("  ✅ Optimization statistics collection");
  console.log("  ✅ Compatible with OpenZeppelin multiproof");
  console.log("  ✅ Emergency root update capability");

  return contractAddress;
}

main()
  .then((address) => {
    console.log(`\n📝 Deployment Summary:`);
    console.log(`Contract Address: ${address}`);
    console.log(`Network: ${process.env.HARDHAT_NETWORK || 'localhost'}`);
    console.log(`Timestamp: ${new Date().toISOString()}`);
    process.exit(0);
  })
  .catch((error) => {
    console.error("❌ Deployment failed:", error);
    process.exit(1);
  });
import { ethers, artifacts, network } from "hardhat";
import fs from "fs";
import path from "path";
import { MerkleVerifier } from "../typechain-types";

async function main(): Promise<void> {
  // Get the deployer account
  const [deployer] = await ethers.getSigners();

  // Get the contract factory
  const MerkleVerifierFactory = await ethers.getContractFactory("MerkleVerifier");

  // Deploy the contract
  const verifier = (await MerkleVerifierFactory.deploy(deployer.address)) as MerkleVerifier;
  await verifier.waitForDeployment();

  console.log("MerkleVerifier deployed to:", await verifier.getAddress());

  // --- This part is crucial for connecting the Python script ---
  // It saves the contract's address and ABI to the artifacts directory,
  // which is necessary for the Python script to find it.
  const artifactsDir = path.join(__dirname, '..', 'artifacts', 'contracts', 'MerkleVerifier.sol');
  if (!fs.existsSync(artifactsDir)) {
    fs.mkdirSync(artifactsDir, { recursive: true });
  }

  // Cast artifact to `any` to allow adding the 'networks' property
  const artifact: any = artifacts.readArtifactSync("MerkleVerifier");
  const chainId = network.config.chainId || 31337; // Default to 31337 for local hardhat network
  
  // Add network information to the artifact
  artifact.networks = {
    ...artifact.networks,
    [chainId]: {
      address: await verifier.getAddress()
    }
  };

  // Write the updated artifact back to the file system
  fs.writeFileSync(
    path.join(artifactsDir, 'MerkleVerifier.json'),
    JSON.stringify(artifact, null, 2)
  );

  console.log("Artifact with deployment address saved successfully.");
}

main()
  .then(() => process.exit(0))
  .catch((error: Error) => {
    console.error(error);
    process.exit(1);
  });


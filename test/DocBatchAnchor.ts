import { expect } from "chai";
import { ethers } from "hardhat";
import { DocBatchAnchor } from "../typechain-types";

describe("DocBatchAnchor", function () {
  let contract: DocBatchAnchor;

  beforeEach(async function () {
    const DocBatchAnchorFactory = await ethers.getContractFactory("DocBatchAnchor");
    contract = (await DocBatchAnchorFactory.deploy()) as DocBatchAnchor;
    await contract.waitForDeployment(); // This replaces contract.deployed()
  });

  it("should store a batch root and retrieve it", async function () {
    const root = ethers.keccak256(ethers.toUtf8Bytes("batch-root-1"));
    const tx = await contract.storeBatchRoot(root);
    await tx.wait();

    const batchCount = await contract.batchCount();
    expect(batchCount).to.equal(1);

    const storedRoot = await contract.batchRoots(0);
    expect(storedRoot).to.equal(root);
  });

  it("should verify a simple Merkle proof (2 leaves)", async function () {
    // Leaves: hash("A"), hash("B")
    const leafA = ethers.keccak256(ethers.toUtf8Bytes("A"));
    const leafB = ethers.keccak256(ethers.toUtf8Bytes("B"));

    // Root = keccak256(leafA || leafB)
    const root = ethers.keccak256(ethers.concat([leafA, leafB]));

    // Store root
    await contract.storeBatchRoot(root);

    // Proof for A = [leafB], index=0
    const proofForA = [leafB];
    const verifiedA = await contract.verifyMembership(leafA, proofForA, 0, 0);
    expect(verifiedA).to.equal(true);

    // Proof for B = [leafA], index=1
    const proofForB = [leafA];
    const verifiedB = await contract.verifyMembership(leafB, proofForB, 1, 0);
    expect(verifiedB).to.equal(true);
  });

  it("should fail verification if proof is wrong", async function () {
    const leafX = ethers.keccak256(ethers.toUtf8Bytes("X"));
    const leafY = ethers.keccak256(ethers.toUtf8Bytes("Y"));

    const root = ethers.keccak256(ethers.concat([leafX, leafY]));
    await contract.storeBatchRoot(root);

    // Wrong proof: provide leafX as sibling for leafX
    const badProof = [leafX];
    const result = await contract.verifyMembership(leafX, badProof, 0, 0);
    expect(result).to.equal(false);
  });

  it("should revert if batch index not found", async function () {
    const leaf = ethers.keccak256(ethers.toUtf8Bytes("A"));
    const proof: string[] = [];
    await expect(
      contract.verifyMembership(leaf, proof, 0, 99)
    ).to.be.revertedWith("batch not found");
  });
});

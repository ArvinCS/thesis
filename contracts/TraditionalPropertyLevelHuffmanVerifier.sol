// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// This contract uses the MerkleProof library that includes the multiProofVerify function.
// This is compatible with OpenZeppelin Contracts v4.x and some versions of v5.x.
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title TraditionalPropertyLevelHuffmanVerifier
 * @author Your Name
 * @notice This contract acts as an on-chain anchor for an off-chain document registry.
 * It stores a single Merkle root representing the state of all documents.
 * It allows for the efficient batch verification of multiple documents in a single transaction
 * using a multiproof compatible with OpenZeppelin's v4.x standard.
 */
contract TraditionalPropertyLevelHuffmanVerifier is Ownable {
    // The single, authoritative Merkle root representing the current state of the entire document registry.
    bytes32 public merkleRoot;

    // Event to be emitted whenever the Merkle root is updated.
    event RootUpdated(bytes32 indexed newRoot, uint256 timestamp);

    /**
     * @notice Constructor to set the initial owner of the contract.
     * @param initialOwner The address of the account that will have administrative privileges.
     */
    constructor(address initialOwner) {
        _transferOwnership(initialOwner);
    }

    /**
     * @notice Updates the Merkle root. Can only be called by the contract owner.
     * @param _newRoot The new Merkle root to be stored.
     */
    function updateRoot(bytes32 _newRoot) external onlyOwner {
        require(_newRoot != bytes32(0), "New root cannot be empty");
        merkleRoot = _newRoot;
        emit RootUpdated(_newRoot, block.timestamp);
    }

    /**
     * @notice Verifies a multiproof using the bottom-up reconstruction method.
     * @param leaves The leaf hashes being proven, sorted lexicographically off-chain.
     * @param proofHashes The minimal, de-duplicated set of sibling hashes, sorted.
     * @param pathMap The "recipe" for rebuilding the tree. An array of pairs of indices.
     * Each index points to a hash in the combined `leaves` and `proofHashes` array,
     * or to a previously computed hash.
     * @return A boolean indicating if the proof is valid.
     */
    function verifyBatch(
        bytes32[] memory leaves,
        bytes32[] memory proofHashes,
        uint256[] memory pathMap
    ) public view returns (bool) {
        require(pathMap.length % 2 == 0, "Invalid path map length");

        // The working set starts with all the known base hashes.
        uint256 leavesLen = leaves.length;
        uint256 proofHashesLen = proofHashes.length;
        uint256 numComputed = pathMap.length / 2;
        
        // This array will hold all leaves, all proof hashes, and all hashes we compute.
        bytes32[] memory workingSet = new bytes32[](leavesLen + proofHashesLen + numComputed);

        // 1. Fill the working set with the known hashes.
        for (uint i = 0; i < leavesLen; i++) {
            workingSet[i] = leaves[i];
        }
        for (uint i = 0; i < proofHashesLen; i++) {
            workingSet[leavesLen + i] = proofHashes[i];
        }

        // 2. Execute the recipe to compute the internal hashes.
        uint256 computedIndex = leavesLen + proofHashesLen;
        for (uint i = 0; i < numComputed; i++) {
            uint256 leftIndex = pathMap[i * 2];
            uint256 rightIndex = pathMap[i * 2 + 1];

            // The indices in pathMap refer to the full workingSet array.
            require(leftIndex < computedIndex && rightIndex < computedIndex, "Invalid path map index");

            bytes32 leftHash = workingSet[leftIndex];
            bytes32 rightHash = workingSet[rightIndex];
            
            // Ensure canonical ordering for hashing
            if (leftHash < rightHash) {
                 workingSet[computedIndex] = keccak256(abi.encodePacked(leftHash, rightHash));
            } else {
                 workingSet[computedIndex] = keccak256(abi.encodePacked(rightHash, leftHash));
            }
            computedIndex++;
        }

        // 3. The final hash computed must be the root.
        // If the tree is not empty, the last element in the workingSet should be the root.
        if (workingSet.length == 0) return merkleRoot == keccak256(abi.encodePacked());
        
        return workingSet[workingSet.length - 1] == merkleRoot;
    }
}


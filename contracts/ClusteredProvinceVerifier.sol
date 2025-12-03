// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";

/**
 * @title ClusteredProvinceVerifier
 * @notice Verifies documents in a balanced Merkle tree with province-based leaf ordering.
 * 
 * This contract uses standard OpenZeppelin multiproof verification for a balanced tree
 * where leaves are grouped by province for jurisdictional efficiency.
 * 
 * Architecture:
 * - Balanced Merkle tree (standard binary tree structure)
 * - Custom leaf ordering: documents grouped by province
 * - Uses OpenZeppelin's multiProofVerify for efficient batch verification
 */
contract ClusteredProvinceVerifier is Ownable {
    /// @notice The Merkle root representing the current state of all documents
    bytes32 public merkleRoot;

    /// @notice Emitted when the Merkle root is updated
    event RootUpdated(bytes32 indexed newRoot, uint256 timestamp);

    /**
     * @notice Constructor to set the initial owner
     * @param initialOwner The address that will have administrative privileges
     */
    constructor(address initialOwner) {
        _transferOwnership(initialOwner);
    }

    /**
     * @notice Updates the Merkle root (only owner)
     * @param _newRoot The new Merkle root
     */
    function updateRoot(bytes32 _newRoot) external onlyOwner {
        require(_newRoot != bytes32(0), "New root cannot be empty");
        merkleRoot = _newRoot;
        emit RootUpdated(_newRoot, block.timestamp);
    }

    /**
     * @notice Verifies multiple documents using OpenZeppelin multiproof
     * @param proof Array of sibling hashes needed for verification
     * @param proofFlags Boolean flags indicating which hashes to use
     * @param leaves Array of leaf hashes being verified (must be sorted)
     * @return bool True if all leaves are valid members of the tree
     */
    function verifyBatch(
        bytes32[] memory proof,
        bool[] memory proofFlags,
        bytes32[] memory leaves
    ) public view returns (bool) {
        return MerkleProof.multiProofVerify(proof, proofFlags, merkleRoot, leaves);
    }
}


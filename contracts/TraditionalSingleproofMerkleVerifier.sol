// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title TraditionalSingleproofMerkleVerifier
 * @author Your Name
 * @notice This contract implements single proof verification for traditional Merkle trees.
 * It allows verification of individual documents one at a time.
 */
contract TraditionalSingleproofMerkleVerifier is Ownable {
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
     * @notice Verifies a single document (leaf) against the stored Merkle root using a single proof.
     * @dev This function uses OpenZeppelin's verify function for single proof verification.
     * @param proof An array of Merkle proof elements for the single leaf.
     * @param leaf The document hash (leaf) to be verified.
     * @return bool True if the leaf is part of the Merkle tree, false otherwise.
     */
    function verifySingle(
        bytes32[] calldata proof,
        bytes32 leaf
    ) public view returns (bool) {
        return MerkleProof.verify(proof, merkleRoot, leaf);
    }

    /**
     * @notice Verifies multiple documents individually using single proofs.
     * @dev This function calls verifySingle for each document, simulating multiple transactions.
     * @param proofs An array of proof arrays, one for each document.
     * @param leaves An array of document hashes to be verified.
     * @return bool True if all leaves are part of the Merkle tree, false otherwise.
     */
    function verifyMultiple(
        bytes32[][] calldata proofs,
        bytes32[] calldata leaves
    ) public view returns (bool) {
        require(proofs.length == leaves.length, "Proofs and leaves arrays must have same length");
        
        for (uint i = 0; i < leaves.length; i++) {
            if (!MerkleProof.verify(proofs[i], merkleRoot, leaves[i])) {
                return false;
            }
        }
        
        return true;
    }

    /**
     * @notice Gets the current Merkle root.
     * @return bytes32 The current Merkle root.
     */
    function getRoot() public view returns (bytes32) {
        return merkleRoot;
    }
}

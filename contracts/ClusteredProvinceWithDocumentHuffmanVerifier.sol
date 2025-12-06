// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// This contract uses the MerkleProof library that includes the multiProofVerify function.
// This is compatible with OpenZeppelin Contracts v4.x and some versions of v5.x.
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title ClusteredProvinceWithDocumentHuffmanVerifier
 * @author Your Name
 * @notice This contract acts as an on-chain anchor for an off-chain document registry.
 * It stores a single Merkle root representing the state of all documents.
 * It allows for the efficient batch verification of multiple documents in a single transaction
 * using a multiproof compatible with OpenZeppelin's v4.x standard.
 */
contract ClusteredProvinceWithDocumentHuffmanVerifier is Ownable {
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
     * @notice Verifies a multiproof using optimized bitmap-based stack approach.
     * @param inputs Combined array of all input hashes (leaves and siblings) in post-order.
     * @param bitmap Bitmap where each bit controls the operation:
     *               - bit=1: Push next hash from inputs array onto stack
     *               - bit=0: Pop two hashes from stack, merge them, push result
     * @return A boolean indicating if the proof is valid.
     * 
     * OPTIMIZED VERSION:
     * - Uses unchecked arithmetic for bounded counters
     * - Removed redundant early termination check from loop
     * - Cached merkleRoot to avoid storage read in hot path
     * - Optimized merge with inline assembly
     */
    function verifyBatch(
        bytes32[] memory inputs,
        uint256[] memory bitmap
    ) public view returns (bool) {
        uint256 inputLen = inputs.length;
        if (inputLen == 0) return false;
        if (bitmap.length == 0) return false;
        
        // Cache storage variable to avoid repeated SLOAD
        bytes32 expectedRoot = merkleRoot;
        
        // Stack to hold intermediate computed hashes
        bytes32[] memory stack = new bytes32[](inputLen);
        uint256 stackSize;
        uint256 inputIndex;
        
        // Total operations = pushes + merges = inputs + (inputs - 1) = 2*inputs - 1
        uint256 totalOps = inputLen * 2 - 1;
        uint256 bitmapIndex;
        uint256 bitPosition;
        uint256 currentWord = bitmap[0];
        
        // Use unchecked for bounded arithmetic - these cannot overflow
        unchecked {
            for (uint256 op = 0; op < totalOps; op++) {
                // Get next bit from bitmap
                bool isPush = (currentWord & (1 << bitPosition)) != 0;
                
                if (isPush) {
                    // Push next input onto stack
                    stack[stackSize] = inputs[inputIndex];
                    stackSize++;
                    inputIndex++;
                } else {
                    // Merge: pop two, hash, push result
                    if (stackSize < 2) return false;
                    
                    stackSize--;
                    bytes32 right = stack[stackSize];
                    stackSize--;
                    bytes32 left = stack[stackSize];
                    
                    // Canonical ordering and hash using assembly for efficiency
                    bytes32 merged;
                    assembly {
                        // Allocate memory for hashing (64 bytes)
                        let ptr := mload(0x40)
                        
                        // Compare and order: smaller hash goes first
                        switch lt(left, right)
                        case 1 {
                            mstore(ptr, left)
                            mstore(add(ptr, 32), right)
                        }
                        default {
                            mstore(ptr, right)
                            mstore(add(ptr, 32), left)
                        }
                        
                        // Compute keccak256
                        merged := keccak256(ptr, 64)
                    }
                    
                    stack[stackSize] = merged;
                    stackSize++;
                }
                
                // Advance bit position
                bitPosition++;
                
                // Move to next bitmap word if needed (every 256 bits)
                if (bitPosition == 256) {
                    bitPosition = 0;
                    bitmapIndex++;
                    if (bitmapIndex < bitmap.length) {
                        currentWord = bitmap[bitmapIndex];
                    }
                }
            }
        }
        
        // After processing, stack should have exactly one element: the root
        return stackSize == 1 && stack[0] == expectedRoot;
    }
}


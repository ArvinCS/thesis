// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title HierarchicalMerkleVerifier
 * @author Your Name
 * @notice This contract implements hierarchical batch verification for Indonesian province-based document registry.
 * 
 * The system uses a two-level hierarchy:
 * 1. Province Trees: Each province has its own Merkle tree of property documents
 * 2. Jurisdiction Tree: A top-level tree where each leaf is a province root
 * 
 * This enables efficient cross-province verification in a single transaction.
 */
contract HierarchicalMerkleVerifierStateful is Ownable {
    // The single, authoritative Jurisdiction Tree root stored on-chain
    bytes32 public jurisdictionRoot;
    
    // Mapping to store individual province roots for reference
    mapping(string => bytes32) public provinceRoots;
    string[] public provinceList;
    
    // Events
    event JurisdictionRootUpdated(bytes32 indexed newRoot, uint256 timestamp);
    event ProvinceRootUpdated(string indexed province, bytes32 indexed newRoot, uint256 timestamp);
    
    /**
     * @notice Constructor to set the initial owner of the contract.
     * @param initialOwner The address of the account that will have administrative privileges.
     */
    constructor(address initialOwner) {
        _transferOwnership(initialOwner);
    }
    
    /**
     * @notice Updates the jurisdiction root and all province roots in a single transaction.
     * @param _newJurisdictionRoot The new jurisdiction tree root
     * @param _provinces Array of province names
     * @param _provinceRoots Array of corresponding province roots
     */
    function updateHierarchicalRoots(
        bytes32 _newJurisdictionRoot,
        string[] calldata _provinces,
        bytes32[] calldata _provinceRoots
    ) external onlyOwner {
        require(_newJurisdictionRoot != bytes32(0), "Jurisdiction root cannot be empty");
        require(_provinces.length == _provinceRoots.length, "Provinces and roots arrays must have same length");
        
        // Update jurisdiction root
        jurisdictionRoot = _newJurisdictionRoot;
        emit JurisdictionRootUpdated(_newJurisdictionRoot, block.timestamp);
        
        // Clear existing province list
        delete provinceList;
        
        // Update province roots
        for (uint i = 0; i < _provinces.length; i++) {
            require(_provinceRoots[i] != bytes32(0), "Province root cannot be empty");
            provinceRoots[_provinces[i]] = _provinceRoots[i];
            provinceList.push(_provinces[i]);
            emit ProvinceRootUpdated(_provinces[i], _provinceRoots[i], block.timestamp);
        }
    }
    
    /**
     * @notice Verifies documents within a single province.
     * @param province The province name
     * @param proof Merkle proof for the documents within the province
     * @param proofFlags Boolean flags for multiproof verification
     * @param leaves Document hashes to verify within the province
     * @return bool True if verification succeeds
     */
    function verifyProvinceDocuments(
        string calldata province,
        bytes32[] calldata proof,
        bool[] calldata proofFlags,
        bytes32[] calldata leaves
    ) public view returns (bool) {
        bytes32 provinceRoot = provinceRoots[province];
        require(provinceRoot != bytes32(0), "Province not found");
        
        return MerkleProof.multiProofVerify(proof, proofFlags, provinceRoot, leaves);
    }
    
    /**
     * @notice Verifies that specific province roots are part of the jurisdiction tree.
     * @param proof Merkle proof for the province roots
     * @param proofFlags Boolean flags for multiproof verification  
     * @param provinceRootsToVerify Array of province roots to verify
     * @return bool True if verification succeeds
     */
    function verifyProvinceRoots(
        bytes32[] calldata proof,
        bool[] calldata proofFlags,
        bytes32[] calldata provinceRootsToVerify
    ) public view returns (bool) {
        return MerkleProof.multiProofVerify(proof, proofFlags, jurisdictionRoot, provinceRootsToVerify);
    }
    
    /**
     * @notice Complete hierarchical verification for cross-province document batches.
     * This is the main function for your thesis - it verifies documents across multiple provinces
     * in a single transaction using the hierarchical proof structure.
     * 
     * @param provinceProofs Array of province-level proofs
     * @param provinceFlags Array of province-level proof flags
     * @param provinceLeavesArrays Array of document hashes for each province
     * @param provincesInvolved Array of province names being verified
     * @param jurisdictionProof Proof that the province roots are in the jurisdiction tree
     * @param jurisdictionFlags Flags for the jurisdiction-level proof
     * @return bool True if complete hierarchical verification succeeds
     */
    function verifyHierarchicalBatch(
        bytes32[][] calldata provinceProofs,
        bool[][] calldata provinceFlags,
        bytes32[][] calldata provinceLeavesArrays,
        string[] calldata provincesInvolved,
        bytes32[] calldata jurisdictionProof,
        bool[] calldata jurisdictionFlags
    ) public view returns (bool) {
        require(provinceProofs.length == provincesInvolved.length, "Province proofs length mismatch");
        require(provinceFlags.length == provincesInvolved.length, "Province flags length mismatch");
        require(provinceLeavesArrays.length == provincesInvolved.length, "Province leaves length mismatch");
        
        // Step 1: Verify each province's documents and collect verified roots
        bytes32[] memory verifiedProvinceRoots = _verifyAllProvinces(
            provinceProofs,
            provinceFlags,
            provinceLeavesArrays,
            provincesInvolved
        );
        
        // Step 2: Verify jurisdiction-level proof
        return MerkleProof.multiProofVerify(
            jurisdictionProof,
            jurisdictionFlags,
            jurisdictionRoot,
            verifiedProvinceRoots
        );
    }
    
    /**
     * @notice Internal function to verify all province-level proofs.
     * Separated to avoid stack too deep error.
     */
    function _verifyAllProvinces(
        bytes32[][] calldata provinceProofs,
        bool[][] calldata provinceFlags,
        bytes32[][] calldata provinceLeavesArrays,
        string[] calldata provincesInvolved
    ) internal view returns (bytes32[] memory) {
        bytes32[] memory verifiedRoots = new bytes32[](provincesInvolved.length);
        
        for (uint i = 0; i < provincesInvolved.length; i++) {
            bytes32 expectedRoot = provinceRoots[provincesInvolved[i]];
            require(expectedRoot != bytes32(0), "Province not found");
            
            bool isValid = MerkleProof.multiProofVerify(
                provinceProofs[i],
                provinceFlags[i],
                expectedRoot,
                provinceLeavesArrays[i]
            );
            
            require(isValid, "Province verification failed");
            verifiedRoots[i] = expectedRoot;
        }
        
        return verifiedRoots;
    }
    
    /**
     * @notice Get the current province root for a specific province.
     * @param province The province name
     * @return bytes32 The current root hash for the province
     */
    function getProvinceRoot(string calldata province) external view returns (bytes32) {
        return provinceRoots[province];
    }
    
    /**
     * @notice Get all registered provinces.
     * @return string[] Array of all province names
     */
    function getAllProvinces() external view returns (string[] memory) {
        return provinceList;
    }
    
    /**
     * @notice Get the total number of registered provinces.
     * @return uint256 The count of provinces
     */
    function getProvinceCount() external view returns (uint256) {
        return provinceList.length;
    }
}

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title HierarchicalMerkleVerifier
 * @notice Versi ini berisi DUA implementasi:
 * 1. verifyHierarchicalBatch: Implementasi N+1 call asli Anda.
 * 2. verifyHierarchicalBatch_Optimized: Implementasi "inline" (monolithic).
 */
contract HierarchicalMerkleVerifier is Ownable {
    // The single, authoritative Jurisdiction Tree root stored on-chain
    bytes32 public jurisdictionRoot;

    // ... (Events tetap sama) ...
    event JurisdictionRootUpdated(bytes32 indexed newRoot, uint256 timestamp);
    event HierarchicalBatchVerified(
        uint256 indexed totalDocuments,
        uint256 indexed provincesInvolved,
        bytes32 indexed jurisdictionRoot
    );

    constructor(address initialOwner) {
        _transferOwnership(initialOwner);
    }

    function updateJurisdictionRoot(bytes32 _newJurisdictionRoot) external onlyOwner {
        require(_newJurisdictionRoot != bytes32(0), "Jurisdiction root cannot be empty");
        jurisdictionRoot = _newJurisdictionRoot;
        emit JurisdictionRootUpdated(_newJurisdictionRoot, block.timestamp);
    }

    /**
     * @notice IMPLEMENTASI OPTIMISASI (INLINED) - "IDE A"
     * @dev Melakukan verifikasi yang sama dengan di atas, tetapi menghindari N+1 call
     * overhead dengan meng-inline-kan logika verifikasi provinsi.
     */
    function verifyHierarchicalBatch(
        bytes32[] calldata claimedProvinceRoots,
        bytes32[] calldata jurisdictionProof,
        bool[] calldata jurisdictionFlags,
        bytes32[][] calldata provinceProofs,
        bool[][] calldata provinceFlags,
        bytes32[][] calldata provinceLeavesArrays
    ) external view returns (bool success) {
        uint256 numProvinces = claimedProvinceRoots.length;
        require(numProvinces > 0, "No province roots provided");
        require(numProvinces == provinceProofs.length, "Mismatch in province data arrays");
        require(numProvinces == provinceFlags.length, "Mismatch in province data arrays");
        require(numProvinces == provinceLeavesArrays.length, "Mismatch in province data arrays");
        
        // --- OPTIMISASI: PANGGILAN #1 ---
        // Gunakan versi 'Calldata' untuk menghemat 2x copy (proof & flags)
        // Copy calldata->memory untuk `claimedProvinceRoots` tidak bisa dihindari di sini.
        bool jurisdictionVerification = MerkleProof.multiProofVerifyCalldata(
            jurisdictionProof,
            jurisdictionFlags,
            jurisdictionRoot,
            claimedProvinceRoots
        );
        
        require(jurisdictionVerification, "Jurisdiction verification failed");
        
        // --- OPTIMISASI: ALOKASI MEMORI SATU KALI ---
        // 1. Temukan buffer terbesar yang kita butuhkan dari semua N provinsi
        uint256 maxTotalHashes = 0;
        for (uint256 i = 0; i < numProvinces; i++) {
            uint256 leavesLen = provinceLeavesArrays[i].length;
            uint256 proofLen = provinceProofs[i].length;
            uint256 totalHashes = provinceFlags[i].length;

            require(leavesLen + proofLen - 1 == totalHashes, "MerkleProof: invalid multiproof");
            
            if (totalHashes > maxTotalHashes) {
                maxTotalHashes = totalHashes;
            }
        }

        // 2. Alokasikan buffer 'hashes' SATU KALI SAJA.
        bytes32[] memory hashes = new bytes32[](maxTotalHashes);

        // --- OPTIMISASI: LOOP INLINE (MENGHINDARI PANGGILAN N KALI) ---
        for (uint256 i = 0; i < numProvinces; i++) {
            // Ambil "slice" calldata untuk provinsi ini. Tidak ada copy!
            bytes32[] calldata proof = provinceProofs[i];
            bool[] calldata proofFlags = provinceFlags[i];
            bytes32[] calldata leaves = provinceLeavesArrays[i];
            bytes32 claimedRoot = claimedProvinceRoots[i];

            uint256 leavesLen = leaves.length;
            uint256 proofLen = proof.length;
            uint256 totalHashes = proofFlags.length;
            require(leavesLen > 0, "No documents provided for province");
            
            // --- MULAI LOGIKA 'processMultiProof' YANG DI-INLINE ---
            // Logika ini disalin dari OpenZeppelin tetapi dimodifikasi
            // untuk membaca dari 'calldata' dan menggunakan kembali buffer 'hashes'
            
            uint256 leafPos = 0;
            uint256 hashPos = 0;
            uint256 proofPos = 0;

            for (uint256 j = 0; j < totalHashes; j++) {
                // Baca 'a' (dari 'leaves' (calldata) atau 'hashes' (memory))
                bytes32 a = leafPos < leavesLen ? leaves[leafPos++] : hashes[hashPos++];
                
                // Baca 'b' (dari 'leaves' (calldata), 'hashes' (memory), atau 'proof' (calldata))
                bytes32 b = proofFlags[j]
                    ? (leafPos < leavesLen ? leaves[leafPos++] : hashes[hashPos++])
                    : proof[proofPos++];
                
                // Tulis ke buffer 'hashes' (memory) yang kita gunakan kembali
                hashes[j] = _hashPair(a, b);
            }

            bytes32 computedRoot;
            if (totalHashes > 0) {
                require(proofPos == proofLen, "MerkleProof: invalid multiproof");
                unchecked {
                    computedRoot = hashes[totalHashes - 1];
                }
            } else {
                // leavesLen pasti > 0 karena 'require' di atas
                computedRoot = leaves[0];
            }
            
            require(computedRoot == claimedRoot, "Province verification failed");
            // --- SELESAI LOGIKA 'processMultiProof' YANG DI-INLINE ---
        }
        
        // ... (Emitting event) ...
        
        return true;
    }

    // ... (Fungsi helper lainnya seperti verifyProvinceDocuments bisa tetap ada) ...


    // --- FUNGSI HELPER BARU (Disalin dari OpenZeppelin 'MerkleProof.sol') ---
    // Kita butuh ini agar logika 'inline' kita berfungsi

    function _hashPair(bytes32 a, bytes32 b) private pure returns (bytes32) {
        return a < b ? _efficientHash(a, b) : _efficientHash(b, a);
    }

    function _efficientHash(bytes32 a, bytes32 b) private pure returns (bytes32 value) {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, a)
            mstore(0x20, b)
            value := keccak256(0x00, 0x40)
        }
    }
}
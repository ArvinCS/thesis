#!/usr/bin/env python3
"""
Bitmap-Based Proof Generator

Converts pathMap format proofs to bitmap-based format for more efficient on-chain verification.

Bitmap format:
- inputs: Combined array of all hashes (leaves and siblings) in proper post-order
- bitmap: Array of uint256 where each bit controls an operation:
  * bit=1: Push next hash from inputs onto stack
  * bit=0: Pop two hashes from stack, merge them, push result
"""

def pathmap_to_bitmap(leaves, proof_hashes, path_map):
    """
    Convert pathMap format to bitmap format.
    
    Args:
        leaves: List of leaf hashes
        proof_hashes: List of sibling/proof hashes
        path_map: List of index pairs [left_idx, right_idx, ...]
    
    Returns:
        dict with 'inputs' (combined hashes in post-order) and 'bitmap' (operation flags)
    """
    if not path_map:
        # Single leaf case
        return {'inputs': leaves, 'bitmap': [1]}  # Just push the leaf
    
    # Build combined hash array (base inputs)
    all_hashes = leaves + proof_hashes
    
    # Track which hashes have been used and their positions
    inputs = []  # Hashes in the order they should be pushed
    operations = []  # 1 for push, 0 for merge
    
    # Simulate stack execution to generate proper sequence
    # Key insight: pathMap is already in post-order, so we process sequentially
    # and track which values are on the stack
    
    stack = []  # Simulated stack: each element is an index (from all_hashes or computed)
    on_stack = set()  # Track what indices are currently on stack
    next_computed_idx = len(all_hashes)  # Computed values start after all_hashes
    
    num_instructions = len(path_map) // 2
    
    # Validation: N inputs should produce N-1 merges to get 1 root
    # This will be checked after we process all instructions
    
    for i in range(num_instructions):
        left_idx = path_map[i * 2]
        right_idx = path_map[i * 2 + 1]
        
        # Special case: if left_idx == right_idx, we need to push it twice
        if left_idx == right_idx:
            if left_idx < len(all_hashes):
                # Push the same value twice
                inputs.append(all_hashes[left_idx])
                operations.append(1)  # Push
                inputs.append(all_hashes[left_idx])
                operations.append(1)  # Push again
                # Don't add to on_stack since we'll immediately consume both
            else:
                # Computed value can't be used twice
                raise ValueError(f"Computed index {left_idx} cannot be merged with itself")
        else:
            # Normal case: different left and right indices
            # Ensure left operand is on stack
            if left_idx not in on_stack:
                if left_idx < len(all_hashes):
                    # Push from inputs
                    inputs.append(all_hashes[left_idx])
                    operations.append(1)  # Push
                    stack.append(left_idx)
                    on_stack.add(left_idx)
                else:
                    # This is a computed value - should already be on stack
                    # If not, there's an error in pathMap ordering
                    raise ValueError(f"Computed index {left_idx} referenced before being computed")
            
            # Ensure right operand is on stack
            if right_idx not in on_stack:
                if right_idx < len(all_hashes):
                    # Push from inputs
                    inputs.append(all_hashes[right_idx])
                    operations.append(1)  # Push
                    stack.append(right_idx)
                    on_stack.add(right_idx)
                else:
                    # This is a computed value - should already be on stack
                    raise ValueError(f"Computed index {right_idx} referenced before being computed")
        
        # Now both operands are on stack, perform merge
        operations.append(0)  # Merge
        
        # Simulate merge: pop right, pop left, push result
        if left_idx == right_idx:
            # Special case: same index twice means both copies are on stack
            # No need to update on_stack since we never added them
            pass
        else:
            # Find and remove right_idx from stack
            if right_idx in stack:
                stack.remove(right_idx)
                on_stack.remove(right_idx)
            
            # Find and remove left_idx from stack
            if left_idx in stack:
                stack.remove(left_idx)
                on_stack.remove(left_idx)
        
        # Push computed result
        stack.append(next_computed_idx)
        on_stack.add(next_computed_idx)
        next_computed_idx += 1
    
    # Validation: check operation counts
    num_pushes = operations.count(1)
    num_merges = operations.count(0)
    expected_merges = num_pushes - 1
    
    if num_merges != expected_merges:
        # This is a critical error - pathMap has incorrect structure
        raise ValueError(f"Invalid pathMap: {num_pushes} pushes requires {expected_merges} merges, but pathMap has {num_merges} merge instructions. "
                        f"PathMap length: {len(path_map)} values = {len(path_map)//2} instructions. "
                        f"This indicates a bug in proof generation.")
    
    if len(stack) != 1:
        raise ValueError(f"After processing pathMap, stack should have 1 element (root) but has {len(stack)}. "
                        f"This indicates pathMap does not correctly compute to root.")
    
    # Convert operations list to bitmap format (array of uint256)
    bitmap = []
    current_bits = 0
    bit_position = 0
    
    for op in operations:
        if op == 1:
            current_bits |= (1 << bit_position)
        # op == 0 means bit stays 0 (already is)
        
        bit_position += 1
        
        # Each uint256 holds 256 bits
        if bit_position == 256:
            bitmap.append(current_bits)
            current_bits = 0
            bit_position = 0
    
    # Add remaining bits if any
    if bit_position > 0:
        bitmap.append(current_bits)
    
    return {
        'inputs': inputs,
        'bitmap': bitmap
    }


def generate_bitmap_proof_from_pathmap(pathmap_proof):
    """
    Helper function to convert a pathmap proof dict to bitmap format.
    
    Args:
        pathmap_proof: Dict with 'leaves', 'proofHashes', 'pathMap'
    
    Returns:
        Dict with 'inputs' and 'bitmap'
    """
    return pathmap_to_bitmap(
        pathmap_proof['leaves'],
        pathmap_proof['proofHashes'],
        pathmap_proof['pathMap']
    )

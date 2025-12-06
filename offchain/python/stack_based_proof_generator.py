"""
Generate bitmap proofs directly from tree structure using depth-first traversal.

The key insight: Stack-based verification can only merge the top two elements.
We need to traverse the tree such that when we want to merge two values,
they are both at the top of the stack.

This requires a depth-first post-order traversal where we fully process
left and right subtrees before processing the parent.
"""

def generate_stack_based_proof(root, leaves_to_prove, all_leaves):
    """
    Generate bitmap proof for stack-based verification.
    
    Args:
        root: The root node of the tree
        leaves_to_prove: Set of leaf hashes we want to prove
        all_leaves: List of all leaf nodes in the tree
        
    Returns:
        dict with 'inputs' (list of hex hashes) and 'bitmap' (list of uint256)
    """
    
    # Step 1: Find which nodes are needed for the proof
    needed_nodes = set()
    leaf_nodes = {leaf.hash: leaf for leaf in all_leaves}
    
    # Start from leaves we want to prove
    for leaf_hash in leaves_to_prove:
        if leaf_hash in leaf_nodes:
            current = leaf_nodes[leaf_hash]
            # Walk up to root, marking all nodes and siblings
            while current:
                needed_nodes.add(current)
                parent = get_parent(current)
                if parent:
                    sibling = get_sibling(current, parent)
                    if sibling:
                        needed_nodes.add(sibling)
                current = parent
    
    # Step 2: Build proof using depth-first traversal
    # This ensures that when we merge, both operands are at top of stack
    inputs = []
    operations = []  # 1 = push, 0 = merge
    
    def traverse(node):
        """
        Depth-first post-order traversal.
        Returns True if this node's value is now on the stack.
        """
        if not node or node not in needed_nodes:
            return False
            
        # If leaf, just push it
        if is_leaf(node):
            inputs.append(node.hash)
            operations.append(1)  # push
            return True
        
        # Process children first (post-order)
        left_pushed = traverse(node.left) if node.left else False
        right_pushed = traverse(node.right) if node.right else False
        
        # If both children pushed values, merge them
        if left_pushed and right_pushed:
            operations.append(0)  # merge
            return True
        
        # If only one child, push sibling and merge
        if left_pushed and not right_pushed:
            # Right child wasn't in proof, so push it as sibling
            inputs.append(node.right.hash)
            operations.append(1)  # push
            operations.append(0)  # merge
            return True
        
        if right_pushed and not left_pushed:
            # Left child wasn't in proof, so push it as sibling
            inputs.append(node.left.hash)
            operations.append(1)  # push
            operations.append(0)  # merge
            return True
        
        # If neither child pushed, this node shouldn't be in proof
        return False
    
    traverse(root)
    
    # Convert operations to bitmap
    bitmap = encode_bitmap(operations)
    
    return {
        'inputs': inputs,
        'bitmap': bitmap
    }

def encode_bitmap(operations):
    """Convert list of operations (1=push, 0=merge) to bitmap format."""
    bitmap = []
    current_word = 0
    bit_position = 0
    
    for op in operations:
        if op == 1:
            current_word |= (1 << bit_position)
        bit_position += 1
        
        if bit_position == 256:
            bitmap.append(current_word)
            current_word = 0
            bit_position = 0
    
    if bit_position > 0:
        bitmap.append(current_word)
    
    return bitmap

# Helper functions (to be implemented based on tree structure)
def get_parent(node):
    """Get parent of a node."""
    return node.parent if hasattr(node, 'parent') else None

def get_sibling(node, parent):
    """Get sibling of a node given its parent."""
    if parent.left == node:
        return parent.right
    elif parent.right == node:
        return parent.left
    return None

def is_leaf(node):
    """Check if node is a leaf."""
    return node.left is None and node.right is None

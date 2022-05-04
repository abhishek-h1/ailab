class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# RECURSIVE APPROACH

def rangeSumBST029(root, L, R):
    ans = 0
    stack = [root]
    while stack:
        node = stack.pop()
        if node:
            if L <= node.val <= R:
                ans += node.val
            if L < node.val:
                stack.append(node.left)
            if node.val < R:
                stack.append(node.right)
    return ans

bst = TreeNode(10)
bst.left = TreeNode(5)
bst.right = TreeNode(15)
bst.left.left = TreeNode(3)
bst.left.right = TreeNode(7)
bst.right.left = TreeNode(13)
bst.right.right = TreeNode(18)
bst.left.left.left = TreeNode(1)
bst.left.right.left = TreeNode(6)
mn = int(input("Enter the Lower value of the range : "))
mx = int(input("Enter the Higher value of the range : "))

dfs = rangeSumBST029(bst, mn, mx)
print(f"The sum of the nodes in the range {mn} and {mx} is {dfs}")

# Definition for a binary tree node.

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution029(object):
    def rangeSumBST(self, root, L, R):        
        if root == None:
            return 0
        res = 0
        q = [root]
        while q:
            next = []
            for node in q:
                if L <= node.val <= R:
                    res += node.val
                if node.left:
                    next.append(node.left)
                if node.right:
                    next.append(node.right)
            q = next
            
        return res

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
bfs=Solution029().rangeSumBST(bst,mn,mx)
print(f"The sum of the nodes in the range {mn} and {mx} is {bfs}")
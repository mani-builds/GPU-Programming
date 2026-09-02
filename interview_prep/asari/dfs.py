#!/usr/bin/env python

class TreeNode:
    def __init__(self,key,val,left=None,right=None,parent=None):
        self.key = key
        self.payload = val
        self.leftChild = left
        self.rightChild = right
        self.parent = parent

    def hasLeftChild(self):
        return self.leftChild

    def hasRightChild(self):
        return self.rightChild

    def isLeftChild(self):
        return self.parent and self.parent.leftChild == self

    def isRightChild(self):
        return self.parent and self.parent.rightChild == self

    def isRoot(self):
        return not self.parent

    def isLeaf(self):
        return not (self.rightChild or self.leftChild)

    def hasAnyChildren(self):
        return self.rightChild or self.leftChild

    def hasBothChildren(self):
        return self.rightChild and self.leftChild

    def spliceOut(self):
        if self.isLeaf():
            if self.isLeftChild():
                self.parent.leftChild = None
            else:
                self.parent.rightChild = None
        elif self.hasAnyChildren():
            if self.hasLeftChild():
                if self.isLeftChild():
                    self.parent.leftChild = self.leftChild
                else:
                    self.parent.rightChild = self.leftChild
                self.leftChild.parent = self.parent
            else:
                if self.isLeftChild():
                    self.parent.leftChild = self.rightChild
                else:
                    self.parent.rightChild = self.rightChild
                self.rightChild.parent = self.parent

    def findSuccessor(self):
        succ = None
        if self.hasRightChild():
            succ = self.rightChild.findMin()
        else:
            if self.parent:
                   if self.isLeftChild():
                       succ = self.parent
                   else:
                       self.parent.rightChild = None
                       succ = self.parent.findSuccessor()
                       self.parent.rightChild = self
        return succ

    def findMin(self):
        current = self
        while current.hasLeftChild():
            current = current.leftChild
        return current

    def replaceNodeData(self,key,value,lc,rc):
        self.key = key
        self.payload = value
        self.leftChild = lc
        self.rightChild = rc
        if self.hasLeftChild():
            self.leftChild.parent = self
        if self.hasRightChild():
            self.rightChild.parent = self


n0 = TreeNode(0, "zero")
n1 = TreeNode(1, "one")
n2 = TreeNode(2, "two")
n3 = TreeNode(3, "three")
n4 = TreeNode(4, "four")
n5 = TreeNode(5, "five")
n2.leftChild = n1
n2.rightChild = n4
n1.parent = n2
n4.parent = n2

n1.leftChild = n0
n0.parent = n1

n4.leftChild = n3
n4.rightChild = n5
n3.parent = n4
n5.parent = n4

root = n2

# from collections import stack
result = []

# recursive DFS with backtracking
def dfs(node:TreeNode):
    if node is not None:
        dfs(node.leftChild)
        result.append(node)
        dfs(node.rightChild)

# def dfs_interative(root):
#     node = root
#     stack = []
#     result = []
#     while stack or node:
#         if node:
#             stack.append(node)
#             node = node.leftChild
#         else:
#             node = stack.pop()
#             result.append(node)
#             node = node.rightChild

#     return result

dfs(root)
print([i.key for i in result])
print(result[0].parent.key)

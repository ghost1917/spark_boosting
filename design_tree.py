# -*- coding: utf-8 -*-
import json

# Здесь определены решающее дерево и его ноды

class DesignTreeNode (object):
    def __init__ (self):
        pass

    def evaluate (self, row):
        pass

    def getDictionary (self):
        return self.__dict__

    def __repr__(self):
        return json.dumps (self.getDictionary (),
                           separators=(",",":"),
                           indent=2,
                           sort_keys=True)


class InternalNode (DesignTreeNode):
    def __init__ (self,
                  feature_index,
                  feature_threshold,
                  left_child,
                  right_child):
        self.feature_index = feature_index
        self.feature_threshold = feature_threshold
        self.left_child = left_child
        self.right_child = right_child


    def getDictionary (self):
        result = self.__dict__.copy()
        result["left_child"]  = self.left_child.getDictionary()
        result["right_child"] = self.right_child.getDictionary()
        return result


    def evaluate (self, row):
        if (row [self.feature_index] > self.feature_threshold):
            return self.right_child.evaluate(row)
        else:
            return self.left_child.evaluate(row)


class LeafNode (DesignTreeNode):
    def __init__ (self, value):
        self.value = value

    def evaluate (self, row):
        return self.value





# Само решающее дерево
class DesignTree (object):
    def __init__ (self, rootNode):
            self.rootNode = rootNode

    def evaluate (self, row):
        return self.rootNode.evaluate (row)

    def __repr__ (self):
        return repr(self.rootNode)

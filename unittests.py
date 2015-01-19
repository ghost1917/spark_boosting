# -*- coding: utf-8 -*-
from pyspark import SparkContext, SparkConf

import unittest
from design_tree import *
import id3

spark_context = SparkContext ()

class TestDesignTree (unittest.TestCase):
    def setUp (self):
        self.leaf_node  = LeafNode (value = -1)
        right_leaf_node = LeafNode (value = 1)
        self.design_node = InternalNode (
                feature_index = 0,
                feature_threshold = 0,
                left_child = self.leaf_node,
                right_child = right_leaf_node)

        self.design_tree = DesignTree (self.design_node)


    def test_leaf (self):
        self.assertEqual (self.leaf_node.evaluate (437), -1)
        self.assertEqual (self.leaf_node.getDictionary(), {"value":-1})


    def test_design_node (self):
        self.assertEqual (self.design_node.evaluate ([1]), 1)
        self.assertEqual (self.design_node.evaluate ([2]), 1)
        self.assertEqual (self.design_node.evaluate ([3]), 1)
        self.assertEqual (self.design_node.evaluate ([-1]), -1)
        self.assertEqual (self.design_node.evaluate ([-2]), -1)
        self.assertEqual (self.design_node.evaluate ([-3]), -1)

        self.assertEqual (self.design_node.feature_index, 0)
        self.assertEqual (self.design_node.feature_threshold, 0)
        self.assertEqual (self.design_node.left_child.value, -1)
        self.assertEqual (self.design_node.right_child.value, 1)

    def test_design_tree (self):
        self.assertEqual (self.design_tree.evaluate ([1]), 1)
        self.assertEqual (self.design_tree.evaluate ([2]), 1)
        self.assertEqual (self.design_tree.evaluate ([-1]), -1)
        self.assertEqual (self.design_tree.evaluate ([-2]), -1)


# Надо протестировать алгоритм построения решающего дерева
# Как мы будем его тестировать
# Сначала проверим, что правильно считается энтропия
class TestCalcEntropy (unittest.TestCase):
    def test_all_zeros (self):
        test_sample = [
                id3.DataPoint ([],0,1),
                id3.DataPoint ([],0,1),
                id3.DataPoint ([],0,1),
                id3.DataPoint ([],0,1),
                id3.DataPoint ([],0,1),
                id3.DataPoint ([],0,1),
                id3.DataPoint ([],0,1),
                ]
        test_sample = spark_context.parallelize (test_sample)
        self.assertEqual (id3.calcEntropy (test_sample), 0)

    def test_all_ones (self):
        test_sample = [
                id3.DataPoint ([],1,1),
                id3.DataPoint ([],1,1),
                id3.DataPoint ([],1,1),
                id3.DataPoint ([],1,1),
                id3.DataPoint ([],1,1),
                id3.DataPoint ([],1,1),
                id3.DataPoint ([],1,1),
                ]
        test_sample = spark_context.parallelize (test_sample)
        self.assertEqual (id3.calcEntropy (test_sample), 0)


    def test_half (self):
        test_sample = [
                id3.DataPoint ([],1,1),
                id3.DataPoint ([],0,1),
                id3.DataPoint ([],1,1),
                id3.DataPoint ([],0,1),
                ]
        test_sample = spark_context.parallelize (test_sample)
        self.assertEqual (id3.calcEntropy (test_sample), 1)


    def test_inequal (self):
        test_sample = [
                id3.DataPoint ([],1,1),
                id3.DataPoint ([],0,1),
                id3.DataPoint ([],1,1),
                ]
        test_sample = spark_context.parallelize (test_sample)
        self.assertAlmostEqual (id3.calcEntropy (test_sample), 0.9182958, delta=0.0001)


    def test_different_weights (self):
        test_sample = [
                id3.DataPoint ([],1,1),
                id3.DataPoint ([],0,2),
                id3.DataPoint ([],1,1),
                ]
        test_sample = spark_context.parallelize (test_sample)
        self.assertAlmostEqual (id3.calcEntropy (test_sample), 1, delta=0.0001)

    def test_different_weights2 (self):
        test_sample = [
                id3.DataPoint ([],1,1),
                id3.DataPoint ([],0,2),
                ]

        test_sample = spark_context.parallelize (test_sample)
        self.assertAlmostEqual (id3.calcEntropy (test_sample), 0.9182958, delta=0.0001)

class TestFildBestThreshold (unittest.TestCase):
    # Надо проверить работу и с признаками 0-1 и с непрерывными признаками
    # Начнем с тривиального граничного случая, когда фича совпадает с классом
    def test_case1 (self):
        test_sample = [
                id3.DataPoint ([0],0,1),
                id3.DataPoint ([1],1,1),
                id3.DataPoint ([0], 0, 1),
                id3.DataPoint ([1], 1, 1),
                ]
        test_sample = spark_context.parallelize (test_sample)
        feature_index, best_threshold, information_gain = id3.findBestThreshold (test_sample, 0, 1)
        self.assertEqual (feature_index, 0)
        self.assertEqual (best_threshold, 0)
        self.assertAlmostEqual (information_gain, 1, delta=0.0001)


    def test_case2 (self):
        test_sample = [
                id3.DataPoint ([1], 0, 1),
                id3.DataPoint ([0], 1, 1),
                ]
        test_sample = spark_context.parallelize (test_sample)
        feature_index, best_threshold, information_gain = id3.findBestThreshold (test_sample, 0, 1)
        self.assertEqual (feature_index, 0)
        self.assertEqual (best_threshold, 0)
        self.assertAlmostEqual (information_gain, 1, delta=0.0001)

    def test_case3 (self):
        test_sample = [
                id3.DataPoint ([0], 0, 1),
                id3.DataPoint ([1], 1, 1),
                id3.DataPoint ([2], 1, 1),
                ]
        test_sample = spark_context.parallelize (test_sample)
        feature_index, best_threshold, information_gain = id3.findBestThreshold (test_sample, 0, 1)
        self.assertEqual (feature_index, 0)
        self.assertEqual (best_threshold, 0)
        self.assertAlmostEqual (information_gain, 1, delta=0.0001)

    def test_case4 (self):
        test_sample = [
                id3.DataPoint ([0], 0, 1),
                id3.DataPoint ([1], 0, 1),
                id3.DataPoint ([2], 1, 2),
                ]
        test_sample = spark_context.parallelize (test_sample)
        feature_index, best_threshold, information_gain = id3.findBestThreshold (test_sample, 0, 1)
        self.assertEqual (feature_index, 0)
        self.assertEqual (best_threshold, 1)
        self.assertAlmostEqual (information_gain, 1, delta=0.0001)


    # Дробный information gain, чтобы проверить округление
    def test_case5 (self):
        test_sample = [
                id3.DataPoint ([0], 0, 1),
                id3.DataPoint ([1], 1, 1),
                id3.DataPoint ([2], 0, 1),
                ]
        test_sample = spark_context.parallelize (test_sample)
        feature_index, best_threshold, information_gain = id3.findBestThreshold (test_sample, 0, 1)
        self.assertEqual (feature_index, 0)
        self.assertEqual (best_threshold, 0)
        self.assertAlmostEqual (information_gain, 0.3333, delta=0.0001)


# Как тестировать id3 ?
#   - проверяем тривиальные случаи
#   - проверяем задачу с футболистами
class TestId3Implementation (unittest.TestCase):
    def test_all_targets_equals (self):
        test_sample = [
                id3.DataPoint ([0], 0, 1),
                id3.DataPoint ([1], 0, 1),
                id3.DataPoint ([2], 0, 1),
                ]
        data_rdd = spark_context.parallelize (test_sample)
        tree = id3.id3Implementation (data_rdd, 2)
        self.assertEqual (tree.getDictionary(), {"value":0})



    def test_all_targets_equals2 (self):
        test_sample = [
                id3.DataPoint ([0], 1, 1),
                id3.DataPoint ([1], 1, 1),
                id3.DataPoint ([2], 1, 1),
                ]
        data_rdd = spark_context.parallelize (test_sample)
        tree = id3.id3Implementation (data_rdd, 2)
        self.assertEqual (tree.getDictionary(), {"value":1})



    def test_tree_size_zero (self):
        test_sample = [
                id3.DataPoint ([0], 0, 1),
                id3.DataPoint ([1], 0, 1),
                id3.DataPoint ([2], 1, 1),
                ]
        data_rdd = spark_context.parallelize (test_sample)
        tree = id3.id3Implementation (data_rdd, 0)
        self.assertEqual (tree.getDictionary(), {"value":0})

    def test_tree_size_zero2 (self):
        test_sample = [
                id3.DataPoint ([0], 0, 1),
                id3.DataPoint ([1], 1, 1),
                id3.DataPoint ([2], 1, 1),
                ]
        data_rdd = spark_context.parallelize (test_sample)
        tree = id3.id3Implementation (data_rdd, 0)
        self.assertEqual (tree.getDictionary(), {"value":1})


    def test_all_features_used(self):
        test_sample = [
                id3.DataPoint ([0], 0, 1),
                id3.DataPoint ([1], 0, 1),
                id3.DataPoint ([2], 1, 1),
                ]
        data_rdd = spark_context.parallelize (test_sample)
        tree = id3.id3Implementation (data_rdd, 1, excluded_features=set([0]))
        self.assertEqual (tree.getDictionary(), {"value":0})


    def test_simple_tree(self):
        test_sample = [
                id3.DataPoint ([0], 0, 1),
                id3.DataPoint ([1], 0, 1),
                id3.DataPoint ([2], 1, 1),
                ]
        data_rdd = spark_context.parallelize (test_sample)
        tree = id3.id3Implementation (data_rdd, 4)
        self.assertEqual (tree.getDictionary(), {
            'feature_index': 0,
            'feature_threshold': 1,
            'left_child': {'value': 0},
            'right_child': {'value': 1}})

    def test_2level_tree(self):
        test_sample = [
                id3.DataPoint ([0,0], 0, 1),
                id3.DataPoint ([1,0], 1, 1),
                id3.DataPoint ([0,1], 1, 1),
                id3.DataPoint ([1,1], 0, 1),
                ]
        data_rdd = spark_context.parallelize (test_sample)
        tree = id3.id3Implementation (data_rdd, 4)
        self.assertEqual (tree.getDictionary(), {
            "feature_index":0,
            "feature_threshold":0,
            "left_child":{
                "feature_index":1,
                "feature_threshold":0,
                "left_child":{
                    "value":0
                    },
                "right_child":{
                    "value":1
                    }
                },
            "right_child":{
                "feature_index":1,
                "feature_threshold":0,
                "left_child":{
                    "value":1
                    },
                "right_child":{
                    "value":0
                    }
                }})

    def test_book_example (self):
        test_sample = [
                id3.DataPoint ([1,1,1,1], 1, 1),
                id3.DataPoint ([1,1,0,1], 1, 1),
                id3.DataPoint ([1,1,0,0], 1, 1),
                id3.DataPoint ([0,1,0,0], 1, 1),
                id3.DataPoint ([0,0,0,0], 0, 1),
                id3.DataPoint ([0,1,0,1], 1, 1),
                id3.DataPoint ([1,0,1,1], 0, 1),
                ]

        data_rdd = spark_context.parallelize (test_sample)
        tree = id3.id3Implementation (data_rdd, 4)
        print tree




# Потом что правильно находится лучший разделитель для разных датасетов

# Потом что правильно строится само дерево


if __name__ == '__main__':
    unittest.main()























# -*- coding: utf-8 -*-
# Простой пример, как пользоваться решающим деревом.
#
import id3
from collections import defaultdict
from pyspark import SparkContext, SparkConf


sc = SparkContext ()

# Загруажем обучающие даныне из hdfs и приводим к виду [вектор фич], целевая фукнция, вес
input_rdd = sc.textFile ("/testdata/kaggle")
data_step1_rdd = input_rdd.map (lambda x: x.split("\t"))
data_rdd = data_step1_rdd.map (lambda x: id3.DataPoint (features = map (lambda y: float(y), x[1:]),
                                                        target   = int (x[0]),
                                                        weight   = 1))

# Создаем дерево алгоритмом id3
tree  = id3.id3 (data_rdd)
classification_results_rdd = data_rdd.map (lambda x: tree.evaluate(x.features))

def evaluate_quality (dataset, classification_results):
    expected_targets = map (lambda x: x.target, dataset)
    comparition = zip (expected_targets, classification_results)
    statistics = defaultdict (int)
    for element in comparition:
        statistics [element] += 1

    return  {
             "true positive" : statistics [(1,1)],
             "false positive" : statistics [(0,1)],
             "true negative" : statistics [(0,0)],
             "false negative" : statistics [(1,0)],
             "precision": 1.*statistics [(1,1)] / (statistics [(1,1)]+statistics [(0,1)]),
             "recall": 1.*statistics [(1,1)] / (statistics [(1,1)]+statistics [(1,0)]),
            }

print evaluate_quality (data_rdd.collect(), classification_results_rdd.collect())

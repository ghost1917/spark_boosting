# -*- coding: utf-8 -*-
from design_tree import *
import math


# То что подается на вход алгоритму при построении решающего дерева
class DataPoint (object):
    def __init__ (self, features, target, weight):
        self.features = features
        self.weight = weight
        self.target = target

    def __repr__ (self):
        return str(self.__dict__)



def pLogP (p):
    if (p > 0):
        return p * math.log(p)/math.log(2)
    else:
        return 0



def calcEntropy (data_rdd):
     weights_for_each_target = (data_rdd
            .map (lambda x: (x.target, x.weight))
            .reduceByKey (lambda x,y: x+y))

     targets_weights = weights_for_each_target.collect ()
     total_weight = float(sum (map (lambda x: x[1], targets_weights)))
     H = sum (map (lambda x: -pLogP (x[1]/total_weight), targets_weights))
     return H

def calcWeightedEntropy (targets_list, total_weight):
    current_weight = float(sum (map (lambda x: x[1], targets_list)))
    H = sum (map (lambda x: -pLogP(x[1]/current_weight), targets_list))*current_weight/total_weight
    return H





def strList (inputList):
    return "\n".join (map (lambda x:str(x), inputList))


def calcCumulativeDistribution (partition_zeros_ones_distribution):
    if (len (partition_zeros_ones_distribution) == 0):
        return [(0,0,0)]
    elif (len (partition_zeros_ones_distribution) == 1):
        return [partition_zeros_ones_distribution[0]]
    else:
        sorted_distribution = sorted (partition_zeros_ones_distribution, key=lambda x:x[0])
        cummulative_distribution = [ sorted_distribution [0] ]
        for (partition, zeros, ones) in sorted_distribution[1:]:
            last_index = len (cummulative_distribution) - 1
            cummulative_zero_ones = (partition,
                                     cummulative_distribution [last_index][1] + zeros,
                                     cummulative_distribution [last_index][2] + ones)
            cummulative_distribution.append (cummulative_zero_ones)

        return cummulative_distribution



# Для числовых признаков надо находить порог, который лучше всего разделяет датасеты
# Получает на вход датасет и индекс фичи, для которой ищется порог
# На выход выдается
#  - индекс фичи,
#  - значение порога
#  - соответствующий им information gain
def findBestThreshold (data_rdd, feature_index, total_entropy, verbose=False):
    PARTITIONS_COUNT = 10

    # sort
    feature_target_weight_rdd = (data_rdd.map (lambda x: (x.features[feature_index],(x.target,x.weight)))
                                         .sortByKey(PARTITIONS_COUNT)
                                         .map (lambda x: (x[0],x[1][0],x[1][1])))

    if verbose:
        print "feature_target_weight_rdd", feature_target_weight_rdd.collect()

    def calcTotalPartitionZerosOnesWeights (partitionIndex, iterator):
        zeros_weight = 0
        ones_weight = 0
        for i in iterator:
            if (i[1] == 0):
                zeros_weight += i[2]
            if (i[1] == 1):
                ones_weight += i[2]

        yield (partitionIndex, zeros_weight, ones_weight)


    partitioned_zeros_ones_weights = (feature_target_weight_rdd
            .mapPartitionsWithIndex (calcTotalPartitionZerosOnesWeights)
            .collect ())
    if  verbose:
        print "partitioned_zeros_ones_weights", partitioned_zeros_ones_weights

    cumulative_zeros_ones_weights = calcCumulativeDistribution (partitioned_zeros_ones_weights)
    if verbose:
        print "cumulative_zeros_ones_weights", cumulative_zeros_ones_weights

    total_weights = reduce (lambda x,y: (x[0]+y[0],x[1]+y[1],x[2]+y[2]), partitioned_zeros_ones_weights)
    if verbose:
        print "total_weights", total_weights

    def calcCurrentPartitionZerosOnesWeights (partitionIndex, iterator):
        zeros_weight = 0
        ones_weight = 0
        current_threshold = None
        if (partitionIndex != 0):
            zeros_weight = cumulative_zeros_ones_weights [partitionIndex-1][1]
            ones_weight = cumulative_zeros_ones_weights [partitionIndex-1][2]

        for i in iterator:
            if verbose:
                print "i", i
                print "zeros_weight", zeros_weight
                print "ones_weight", ones_weight
            if current_threshold is None:
                current_threshold = i[0]
            elif current_threshold != i[0]:
                yield (current_threshold,
                       zeros_weight,
                       ones_weight,
                       total_weights[1] - zeros_weight,
                       total_weights[2] - ones_weight)
                current_threshold = i[0]

            if (i[1] == 0):
                zeros_weight += i[2]
            if (i[1] == 1):
                ones_weight += i[2]


        if current_threshold is not None:
            yield (current_threshold,
                   zeros_weight,
                   ones_weight,
                   total_weights[1] - zeros_weight,
                   total_weights[2] - ones_weight)

    threshold_less_equal_weights = feature_target_weight_rdd.mapPartitionsWithIndex (calcCurrentPartitionZerosOnesWeights)

    if verbose:
        print "threshold_less_equal_weights", threshold_less_equal_weights.collect()

    def calcThresholdEntropy (threshold_weights):
        threshold, left_zeros, left_ones, right_zeros, right_ones = threshold_weights

        left_weight = left_zeros + left_ones
        right_weight = right_zeros + right_ones
        total_weight = left_weight + right_weight

        p_left_zeros = 1.0 * left_zeros / left_weight if (left_weight > 0) else 0
        p_left_ones  = 1.0 * left_ones  / left_weight if (left_weight > 0) else 0

        p_right_zeros = 1.0 * right_zeros / right_weight if (right_weight > 0) else 0
        p_right_ones  = 1.0 * right_ones  / right_weight if (right_weight > 0) else 0

        H = -((pLogP (p_left_zeros)  + pLogP (p_left_ones)) * left_weight
             +(pLogP (p_right_zeros) + pLogP (p_right_ones)) * right_weight) / total_weight

        return (threshold, H)


    threshold_entropy = threshold_less_equal_weights.map (calcThresholdEntropy)

    if verbose:
        print "threshold_entropy", threshold_entropy.collect()
        print len (threshold_entropy.collect())
    # Теперь, зная total_weights рассчитываем энтропию для каждого из порогов

    # вычисляем information gain
    threshold_gain = threshold_entropy.map (lambda x:(x[0],total_entropy - x[1]))

    # находим порог с максимальным information gain
    max_threshold = threshold_gain.reduce (lambda x,y: max(x,y,key=lambda z:z[1]))
    return (feature_index, max_threshold[0],max_threshold[1])



def id3Implementation (data_rdd,
                       tree_depth,
                       excluded_features = set(),
                       verbose=False):
    number_of_examples = data_rdd.count()
    number_of_positive_examples = data_rdd.filter (lambda x: x.target==1).count()
    number_of_features = len(data_rdd.first().features)

    # Нода должна быть листом, если:
    if (number_of_positive_examples == number_of_examples  #  - в обучающей выборке остались только примеры одного класса
        or number_of_positive_examples == 0                #
        or len (excluded_features) >= number_of_features   # - в дереве сиспользованы уже все фичи
        or tree_depth <= 0):                               #  - глубина дерева достигла максимума
        answer = 1 if number_of_positive_examples > number_of_examples/2 else 0
        return LeafNode (value=answer)


    entropy_of_dataset = calcEntropy (data_rdd)
    if verbose:
        print "\n\n-----\nEntropy of dataset: ", entropy_of_dataset
    available_features = set (range (0,number_of_features)) - excluded_features
    features_best_thresholds = map (lambda x: findBestThreshold(data_rdd, x, entropy_of_dataset, verbose), available_features)
    best_feature_threshold = max (features_best_thresholds, key=lambda t:t[2])

    new_excluded_features = excluded_features.copy()
    new_excluded_features.add (best_feature_threshold[0])

    left_tree_dataset = data_rdd.filter(lambda x:x.features[best_feature_threshold[0]] <= best_feature_threshold[1])
    right_tree_dataset = data_rdd.filter(lambda x:x.features[best_feature_threshold[0]] > best_feature_threshold[1])


    if verbose:
        print "==================="
        print "Data: ", json.dumps(data_rdd.map(lambda x:json.dumps(x)).collect (), indent=2, separators=(",",": "))
        print "Best threshold: ", best_feature_threshold
        print "Left dataset: ", json.dumps(left_tree_dataset.map(lambda x:json.dumps(x)).collect (), indent=2, separators=(",",": "))
        print "Right dataset: ", json.dumps(right_tree_dataset.map(lambda x:json.dumps(x)).collect (), indent=2, separators=(",",": "))
        print "===================\n\n\n"

    node = InternalNode (
            feature_index = best_feature_threshold [0],
            feature_threshold = best_feature_threshold [1],
            left_child  = id3Implementation (left_tree_dataset,  tree_depth-1, new_excluded_features,verbose),
            right_child = id3Implementation (right_tree_dataset, tree_depth-1, new_excluded_features,verbose))

    return node


# Строит решающее дерево на основе датасета
def id3 (data_rdd, verbose=False):
    return DesignTree (id3Implementation (data_rdd, tree_depth=4, verbose=verbose))


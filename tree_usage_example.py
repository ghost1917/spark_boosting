# Простой пример, как пользоваться решающим деревом.
#
import id3
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

# Применяем обученное дерево ко всем элементам
for element in data_rdd.take(10):
    result = tree.evaluate (element.features)
    print "expected result", element.target,  "classification result:", result

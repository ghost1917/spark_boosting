import id3

input_rrd = sc.textFile ("/testdata/kaggle")
data_step1_rdd = input_rdd.map (lambda x: x.split("\t"))
data_rdd = data_step1_rdd.map (lambda x: {"f":map (lambda y: float(y), x[1:]), "t": int (x[0]), "w":1})


tree  = id3.id3 (data_rdd)


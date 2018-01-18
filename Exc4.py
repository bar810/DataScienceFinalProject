import os
from pyspark.mllib import linalg as mllib_linalg
from pyspark.ml import linalg as ml_linalg
import pandas as pd
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.sql import SparkSession
import pandasql as pdsql
from sklearn.cross_validation import train_test_split
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint

os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"]="python3"

def as_old(v):
    if isinstance(v, ml_linalg.SparseVector):
        return mllib_linalg.SparseVector(v.size, v.indices, v.values)
    if isinstance(v, ml_linalg.DenseVector):
        return mllib_linalg.DenseVector(v.values)
    raise ValueError("Unsupported type {0}".format(type(v)))

lambda row: LabeledPoint(row.label, as_old(row.features))

pysql = lambda q: pdsql.sqldf(q, globals())

spark = SparkSession.builder.appName("FinalProject").master("local[*]").getOrCreate()

cf = spark.read.csv('Hotels_data_Changed.csv', header=True)\
    .withColumnRenamed('Snapshot ID', 'SnapshotID')\
    .withColumnRenamed('Snapshot Date', 'SnapshotDate')\
    .withColumnRenamed('Checkin Date', 'CheckinDate')\
    .withColumnRenamed('Original Price', 'OriginalPrice')\
    .withColumnRenamed('Discount Price', 'DiscountPrice')\
    .withColumnRenamed('Discount Code', 'DiscountCode')\
    .withColumnRenamed('Available Rooms', 'AvailableRooms')\
    .withColumnRenamed('Hotel Name', 'HotelName')\
    .withColumnRenamed('Hotel Stars', 'HotelStars')

cf.createOrReplaceTempView('cf')

query = 'select a.SnapshotDate, a.CheckinDate, a.DiscountCode, a.HotelName, a.DayDiff, a.WeekDay, a.DiscountDiff ' \
        'from cf a ' \
        '   left outer join cf b ' \
        '       on a.HotelName = b.HotelName ' \
        '           and a.DiscountDiff < b.DiscountDiff ' \
        'where b.HotelName is null'

df = spark.sql(query)

snapshotIndexer = StringIndexer(inputCol="SnapshotDate", outputCol="SnapshotDateIndex")
checkinDateIndexer = StringIndexer(inputCol="CheckinDate", outputCol="CheckinDateIndex")
hotelNameIndexer = StringIndexer(inputCol="HotelName", outputCol="HotelNameIndex")
weekDayIndexer = StringIndexer(inputCol="WeekDay", outputCol="WeekDayIndex")
df = snapshotIndexer.fit(df).transform(df)
df = checkinDateIndexer.fit(df).transform(df)
df = hotelNameIndexer.fit(df).transform(df)
df = weekDayIndexer.fit(df).transform(df)
df = df.drop('SnapshotDate')
df = df.drop('CheckinDate')
df = df.drop('HotelName')
df = df.drop('WeekDay')
df = df.withColumnRenamed('SnapshotDateIndex', 'SnapshotDate')\
    .withColumnRenamed('CheckinDateIndex','CheckinDate')\
    .withColumnRenamed('HotelNameIndex','HotelName')\
    .withColumnRenamed('WeekDayIndex','WeekDay')

assembler = VectorAssembler(
  inputCols=['SnapshotDate',
             'CheckinDate',
             'HotelName',
             'WeekDay'], outputCol="features")

output = assembler.transform(df).select('DiscountCode','features').withColumnRenamed('DiscountCode', 'label')

output = output.rdd.map(lambda row: LabeledPoint(row.label, as_old(row.features)))

# DECISION TREE CLASSIFIER
print("------------------------DECISION TREE-----------------------")
training, test = output.randomSplit([0.6, 0.4], seed=0)
treeModel = DecisionTree.trainClassifier(training, numClasses=5, categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=5, maxBins=32)


# print("-------------------------NAIVE BAYES------------------------")
# # Split data aproximately into training (60%) and test (40%)
# training, test = output.randomSplit([0.6, 0.4], seed=0)
# #training.show()
# # Train a naive Bayes model.
# from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
# naiveModel = NaiveBayes.train(training, 1.0)
#
# # Make prediction and test accuracy.
# predictionAndLabel = test.map(lambda p: (naiveModel.predict(p.features), p.label))
# accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
# print('accuracy is: {}'.format(accuracy))

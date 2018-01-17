import pandas as pd
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.sql import SparkSession
import pandasql as pdsql
from sklearn.cross_validation import train_test_split
from pyspark.ml.feature import VectorAssembler

from pyspark.mllib.regression import LabeledPoint


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

output.show()


from pyspark.mllib import linalg as mllib_linalg
from pyspark.ml import linalg as ml_linalg

def as_old(v):
    if isinstance(v, ml_linalg.SparseVector):
        return mllib_linalg.SparseVector(v.size, v.indices, v.values)
    if isinstance(v, ml_linalg.DenseVector):
        return mllib_linalg.DenseVector(v.values)
    raise ValueError("Unsupported type {0}".format(type(v)))

lambda row: LabeledPoint(row.label, as_old(row.features))

output = output.rdd.map(lambda row: LabeledPoint(row.label, as_old(row.features)))

# DECISION TREE CLASSIFIER
print("------------------------DECISION TREE-----------------------")
print()
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# (trainingData, testData) = output.randomSplit([0.7, 0.3])
# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
# model = DecisionTree.trainClassifier(trainingData, numClasses=4, categoricalFeaturesInfo={},
#                                      impurity='gini', maxDepth=5, maxBins=32)
# Evaluate model on test instances and compute test error
#predictions = model.predict(testData.map(lambda x: x.features))
# from pyspark.ml.regression import LinearRegression
# Create a Linear Regression Model object
#lr = LinearRegression(labelCol='DiscountCode')
# Fit the model to the data and call this model lrModel
#lrModel = lr.fit(trainingData)
# encoder = OneHotEncoder(includeFirst=False, inputCol="SnapshotDateIndex", outputCol="SnapshotDateVec")
# encoded = encoder.transform(indexed)
#print(encoded)
# df['WeekDay'] = df['WeekDay'].apply(translate1)
# df['HotelName'] = df['HotelName'].apply(translate2)
# df['CheckinDate'] = df['CheckinDate'].apply(translate3)
# df['SnapshotDate'] = df['SnapshotDate'].apply(translate4)
#df.show()
print("-------------------------NAIVE BAYES------------------------")
print(output)
# Split data aproximately into training (60%) and test (40%)
training, test = output.randomSplit([0.6, 0.4], seed=0)
#training.show()
# Train a naive Bayes model.
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
model = NaiveBayes.train(training, 1.0)
print(model)






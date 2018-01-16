import pandas as pd
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.mllib.tree import DecisionTree
from pyspark.sql import SparkSession
import pandasql as pdsql
from sklearn.cross_validation import train_test_split

pysql = lambda q: pdsql.sqldf(q, globals())

dfTemp = pd.read_csv('test.csv')

dfTemp2 = pysql('select name,code, min(price) from dfTemp group by name')

#print(dfTemp2)

spark = SparkSession.builder.appName("FinalProject").master("local[*]").getOrCreate()


dfTemp = spark.read.csv('test.csv', header=True)
dfTemp.createOrReplaceTempView('dfTemp')

dfTemp2 = spark.sql('select a.name,a.code,a.price '
                    'from dfTemp a '
                    '   left outer join dfTemp b '
                    '   on a.name = b.name and b.price < a.price '
                    'where b.name is null')
#
#dfTemp2.show()

#keep_col = ['Snapshot Date', 'Checkin Date', 'Discount Code', 'Hotel Name', 'DayDiff', 'WeekDay', 'DiscountDiff']

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

#srcDf.show()

query = 'select a.SnapshotDate, a.CheckinDate, a.DiscountCode, a.HotelName, a.DayDiff, a.WeekDay, a.DiscountDiff ' \
        'from cf a ' \
        '   left outer join cf b ' \
        '       on a.HotelName = b.HotelName ' \
        '           and a.DiscountDiff < b.DiscountDiff ' \
        'where b.HotelName is null'

df = spark.sql(query)

df.show()

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

df.show()

features = ['SnapshotDate', 'CheckinDate', 'HotelName', 'WeekDay', 'DayDiff']

X = df[features]
y = df["DiscountCode"]
columns_names=X.schema.names

# DECISION TREE CLASSIFIER
print("------------------------DECISION TREE-----------------------")
print()
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

(trainingData, testData) = df.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
model = DecisionTree.trainClassifier(trainingData, numClasses=4, categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=5, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))



# encoder = OneHotEncoder(includeFirst=False, inputCol="SnapshotDateIndex", outputCol="SnapshotDateVec")
# encoded = encoder.transform(indexed)
#print(encoded)

# df['WeekDay'] = df['WeekDay'].apply(translate1)
# df['HotelName'] = df['HotelName'].apply(translate2)
# df['CheckinDate'] = df['CheckinDate'].apply(translate3)
# df['SnapshotDate'] = df['SnapshotDate'].apply(translate4)

#df.show()

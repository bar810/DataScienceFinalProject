import pandas as pd
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql import SparkSession
import pandasql as pdsql

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

#stringIndexer = StringIndexer(inputCol="SnapshotDate", outputCol="SnapshotDateIndex")
#model = stringIndexer.fit(df)
#print(model)
#indexed = model.transform(df)
#print(indexed)
#encoder = OneHotEncoder(includeFirst=False, inputCol="SnapshotDateIndex", outputCol="SnapshotDateVec")
#encoded = encoder.transform(indexed)
#print(encoded)

# df['WeekDay'] = df['WeekDay'].apply(translate1)
# df['HotelName'] = df['HotelName'].apply(translate2)
# df['CheckinDate'] = df['CheckinDate'].apply(translate3)
# df['SnapshotDate'] = df['SnapshotDate'].apply(translate4)

df.show()

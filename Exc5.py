import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt

os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"]="python3"
spark = SparkSession.builder.appName("FinalProject").master("local[*]").getOrCreate()

origDf = spark.read.csv('hotels_data.csv', header=True)\
        .withColumnRenamed('Snapshot ID', 'SnapshotID')\
        .withColumnRenamed('Snapshot Date', 'SnapshotDate')\
        .withColumnRenamed('Checkin Date', 'CheckinDate')\
        .withColumnRenamed('Original Price', 'OriginalPrice')\
        .withColumnRenamed('Discount Price', 'DiscountPrice')\
        .withColumnRenamed('Discount Code', 'DiscountCode')\
        .withColumnRenamed('Available Rooms', 'AvailableRooms')\
        .withColumnRenamed('Hotel Name', 'HotelName')\
        .withColumnRenamed('Hotel Stars', 'HotelStars')
origDf.createOrReplaceTempView('origDf')
#origDf.show()

hotelsDf = spark.sql('select HotelName, count(HotelName) ' \
                        'from origDf '\
                        'group by HotelName '\
                        'order by count(HotelName) desc '\
                        'limit 150')
hotelsDf.createOrReplaceTempView('hotelsDf')
#hotelsDf.show()

checkinDf = spark.sql('select CheckinDate, count(CheckinDate) ' \
                        'from origDf '\
                        'group by CheckinDate '\
                        'order by count(CheckinDate) desc '\
                        'limit 40')
checkinDf.createOrReplaceTempView('checkinDf')
#checkinDf.show()

bestDiscountCodesDf = spark.sql('select HotelName, CheckinDate,DiscountCode, min(DiscountPrice) '\
                                'from origDf '\
                                'where HotelName in (select HotelName from hotelsDf) '\
                                'and CheckinDate in (select CheckinDate from checkinDf) '\
                                'group by HotelName, CheckinDate, DiscountCode')
bestDiscountCodesDf = bestDiscountCodesDf.withColumnRenamed('min(DiscountPrice)', 'DiscountPrice')
bestDiscountCodesDf.createOrReplaceTempView('bestDiscountCodesDf')
#bestDiscountCodesDf.show()

discountCodeDf = spark.sql('select DiscountCode '
                           'from origDf '
                           'group by DiscountCode')
discountCodeDf.createOrReplaceTempView('discountCodeDf')
#discountCodeDf.show()

checkinDateEditedDf = spark.sql('select CheckinDate '
                                'from checkinDf '
                                'group by CheckinDate')
checkinDateEditedDf.createOrReplaceTempView('checkinDateEditedDf')

crossJoinDf = spark.sql('select a.DiscountCode as DiscountCode, b.CheckinDate as CheckinDate '
                       'from discountCodeDf a '
                       '    cross join checkinDateEditedDf b')
crossJoinDf.createOrReplaceTempView('crossJoinDf')
#crossJoinDf.show()

def concat(value1, value2):
   return value1 + '_' + value2

udfconcat = udf(concat, StringType())

crossJoinDf = crossJoinDf.withColumn('combo', udfconcat('CheckinDate', 'DiscountCode'))
crossJoinDf.createOrReplaceTempView('crossJoinDf')
#crossJoinDf.show()

prePivotDf = spark.sql('select a.HotelName, a.DiscountPrice, b.combo '\
                        'from bestDiscountCodesDf as a '\
                        '   inner join crossJoinDf as b '\
                        'on a.DiscountCode=b.DiscountCode '
                       '    and a.CheckinDate=b.CheckinDate ')

prePivotDf = prePivotDf.withColumn('DiscountPriceInt', prePivotDf.DiscountPrice.cast('int'))\
                        .drop('DiscountPrice')\
                        .withColumnRenamed('DiscountPriceInt', 'DiscountPrice')

prePivotDf.createOrReplaceTempView('prePivotDf')

pivotDf = prePivotDf.groupby('HotelName')\
                    .pivot('combo')\
                    .avg('DiscountPrice')
pivotDf = pivotDf.fillna(-1)
pivotDf.createOrReplaceTempView('pivotDf')
pivotDf.show()

prePivotDf.withColumn('norm_val', (prePivotDf.DiscountPrice-min)/(max-min))



# pivotDf = prePivotDf.groupby('HotelName')\
#                     .pivot('combo')\
#                     .avg('DiscountPrice')
# pivotDf = pivotDf.fillna(-1)
# pivotDf.createOrReplaceTempView('pivotDf')
# pivotDf.show()

# # dendrogram
# scaled_df=pivotDf.toPandas()
# scaled_df = scaled_df.set_index('HotelName')
# print()
# del scaled_df.index.name
# # Calculate the distance between each sample
# Z = hierarchy.linkage(scaled_df, 'ward')
# # Plot with Custom leaves
# hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=5,labels=scaled_df.index)
# plt.show()
#
#
#
#

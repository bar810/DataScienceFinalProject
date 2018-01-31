from pyspark.sql import SparkSession

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
origDf.show()

hotelsDf = spark.sql('select HotelName, count(HotelName) ' \
                        'from origDf '\
                        'group by HotelName '\
                        'order by count(HotelName) desc '\
                        'limit 150')
hotelsDf.createOrReplaceTempView('hotelsDf')
hotelsDf.show()

checkinDf = spark.sql('select CheckinDate, count(CheckinDate) ' \
                        'from origDf '\
                        'group by CheckinDate '\
                        'order by count(CheckinDate) desc '\
                        'limit 40')
checkinDf.createOrReplaceTempView('checkinDf')
checkinDf.show()

bestDiscountCodesDf = spark.sql('select HotelName, CheckinDate,DiscountCode, min(DiscountPrice) '\
                                'from origDf '\
                                'where HotelName in (select HotelName from hotelsDf) '\
                                'and CheckinDate in (select CheckinDate from checkinDf) '\
                                'group by HotelName, CheckinDate, DiscountCode')
bestDiscountCodesDf.createOrReplaceTempView('bestDiscountCodesDf')
bestDiscountCodesDf.show()

discountCodeDf = spark.sql('select DiscountCode '
                           'from origDf '
                           'group by DiscountCode')
discountCodeDf.createOrReplaceTempView('discountCodeDf')
discountCodeDf.show()

checkinDateEditedDf = spark.sql('select CheckinDate '
                                'from checkinDf '
                                'group by CheckinDate')
checkinDateEditedDf.createOrReplaceTempView('checkinDateEditedDf')

crossJoinDf = spark.sql('select a.DiscountCode, b.CheckinDate '
                       'from discountCodeDf a '
                       '    cross join checkinDateEditedDf b')

crossJoinDf.createOrReplaceTempView('crossJoinDf')
crossJoinDf.show()


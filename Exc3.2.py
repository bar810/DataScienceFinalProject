import pandas as pd
import pandasql as pdsql

colnames = ['SnapshotID','SnapshotDate','CheckinDate','Days','OriginalPrice','DiscountPrice','DiscountCode','AvailableRooms',
            'HotelName','HotelStars']

df = pd.read_csv('hotels_data.csv', names=colnames, header=None)

pysql = lambda q: pdsql.sqldf(q, globals())


#150 hotels most records
#40 check in dates most records
#fro each date -> take 4 costs with 4 discount codes
#total 160 columns: each date,price -> column
    # case no price on date -> -1
    #min price per snapshot
#normalize price(0-100)

#finally:
    #each row -> 161 columns: hotel name, 160 dates+prices

hotelQuery = 'select HotelName, count(HotelName) ' \
            'from df '\
            'group by HotelName '\
            'order by count(HotelName) desc '\
            'limit 150'

hotelNamesDf = pysql(hotelQuery)\

print(hotelNamesDf)

checkInQuery = 'select CheckinDate, count(CheckinDate) ' \
            'from df '\
            'where count(CheckinDate) > 40 '\
            'group by CheckinDate '\

checkInDf = pysql(hotelQuery)

print(checkInDf)


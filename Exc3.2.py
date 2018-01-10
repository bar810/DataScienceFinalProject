import pandas as pd
import pandasql as pdsql


colnames = ['SnapshotID','SnapshotDate','CheckinDate','Days','OriginalPrice','DiscountPrice','DiscountCode','AvailableRooms',
            'HotelName','HotelStars']

df = pd.read_csv('hotels_data.csv', names=colnames, header=None)

pysql = lambda q: pdsql.sqldf(q, globals())
print("-------------------QUEREY 5-------------------")
#150 hotels most records
hotelQuery = 'select HotelName, count(HotelName) ' \
            'from df '\
            'group by HotelName '\
            'order by count(HotelName) desc '\
            'limit 150'

hotelNamesDf = pysql(hotelQuery)
# print(hotelNamesDf)
#
print("-------------------QUEREY 6-------------------")
#40 check in dates most records
checkInQuery = 'select CheckinDate, count(CheckinDate) ' \
            'from df '\
            'group by CheckinDate '\
            'order by count(CheckinDate) desc '\
            'limit 40'

checkInDf = pysql(checkInQuery)
# print(checkInDf)

print("-------------------QUEREY 7-------------------")
#fro each date -> take 4 costs with 4 discount codes
#total 160 columns: each date,price -> column
    # case no price on date -> -1
    #min price per snapshot
# query='select CheckinDate, DiscountPrice, '\
#         'FROM df ' \
#         'where CheckinDate in (select CheckinDate from checkinDf)'


names = {'CheckinDate':'CheckinDate', 'DiscountCode':'DiscountCode', 'min(DiscountPrice)':'minDiscountPrice'}

minDiscountPriceQuery = 'select CheckinDate, DiscountCode , min(DiscountPrice) '\
                        'from df ' \
                        'where CheckinDate in (select CheckinDate from checkInDf) ' \
                        'group by CheckinDate, DiscountCode'

minDiscountPriceDf = pysql(minDiscountPriceQuery).rename(columns=names)

#print(minDiscountPriceDf)

#print(minDiscountPriceDf)

qq='select * from df where CheckinDate="10/1/2015 0:00"'
# query1 = 'select  SnapshotDate, CheckinDate, DiscountCode, min(DiscountPrice) '\
#         'from df where HotelName in(select HotelName from hotelNamesDf) '\
#         'and CheckinDate in (select CheckinDate from checkInDf) '\
#         'group by SnapshotDate, CheckinDate, DiscountCode'




#
# 0     10/1/2015 0:00            1               1063
# 1     10/1/2015 0:00            2               1060
# 2     10/1/2015 0:00            3               1069
# 3     10/1/2015 0:00            4               1145

dd=pysql(qq)
#print(dd)

# print("-------------------QUEREY 8-------------------")
# #normalize price(0-100)
#
#
#



#finally:
    #each row -> 161 columns: hotel name, 160 dates+prices



def df_crossjoin(df1, df2, **kwargs):
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    res.index = pd.MultiIndex.from_product((df1.index, df2.index))

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    return res

crossJoinedDf1 = df_crossjoin(hotelNamesDf[['HotelName']], checkInDf[['CheckinDate']])
#   print(crossJoinedDf)

codesDf = pysql('select distinct DiscountCode from df where DiscountCode in (1,2,3,4)') #TODO - fix!

#print(codesDf)

crossJoinDf2 = df_crossjoin(hotelNamesDf[['HotelName']], codesDf[["DiscountCode"]])

#print('got so far')

cols = {'a.HotelName':'HotelName', 'a.CheckinDate':'CheckinDate', 'c.DiscountCode':'DiscountCode', 'c.minDiscountPrice':'minDiscountPrice'}

joinedDf = pysql('select a.HotelName, a.CheckinDate, c.DiscountCode, c.minDiscountPrice '\
                 'from crossJoinedDf1 as a '\
                 # '  inner join crossJoinDf2 as b '\
                 # '      on a.HotelName = b.HotelName '\
                 '  inner join minDiscountPriceDf as c '\
                 '      on c.CheckinDate = a.CheckinDate').rename(columns=cols)

#print(joinedDf)

#joinedDf.to_csv('joinedDf.csv', sep=',')

#print(joinedDf)

joinedDf.pivot(index='HotelName', columns='CheckinDate', values='minDiscountPrice')

print(joinedDf)
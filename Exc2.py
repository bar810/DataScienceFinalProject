import pandas as pd
import pandasql as pdsql

pysql = lambda q: pdsql.sqldf(q, globals())

colnames = ['SnapshotID', 'SnapshotDate', 'CheckinDate', 'Days', 'OriginalPrice', 'DiscountPrice', 'DiscountCode', 'AvailableRooms', 'HotelName', 'HotelStars', 'DayDiff', 'WeekDay', 'DiscountDiff', 'DiscountPerc']
df = pd.read_csv('Hotels_data_Changed.csv', header=None)
#sprint(df)

#  query = 'select *' \
#           'from df'
#
# dfX = pysql(query)
#
# print(dfX)

df2 = df[['SnapshotDate', 'CheckinDate','DiscountCode','HotelName', 'DayDiff', 'WeekDay', 'DiscountDiff']]
keep_col = ['SnapshotDate', 'CheckinDate', 'DiscountCode', 'HotelName', 'DayDiff', 'WeekDay', 'DiscountDiff']
new_f = df[keep_col]
new_f.to_csv("Classification.csv")
Classification = pd.read_csv('Classification.csv', header=None, names=colnames)#.head(5)
#print(Classification)

df3 = Classification[['SnapshotDate', 'CheckinDate', 'DiscountCode', 'HotelName', 'DayDiff', 'WeekDay', 'DiscountDiff']]
print('-----------------------------------print df3---------------------------------------')
print(df3)
print('-----------------------------------print df3---------------------------------------')

query = 'select SnapshotDate, CheckinDate, DiscountCode, HotelName, DayDiff, WeekDay, max(DiscountDiff) from df3 group by SnapshotDate, CheckinDate, DiscountCode, HotelName, DayDiff, WeekDay'


df4 = pysql(query)
print('-----------------------------------print df4---------------------------------------')
print(df4)
print('-----------------------------------print df4---------------------------------------')
df4.to_csv("new.csv")
# a = pd.read_csv('new.csv')
# print(a)

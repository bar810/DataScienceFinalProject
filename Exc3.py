
import pandas as pd
import pandasql as pdsql

# RELOAD THE ORIGINAL CSV
colnames= ['index','SnapshotID','SnapshotDate','CheckinDate','Days','OriginalPrice','DiscountPrice','DiscountCode','AvailableRooms','HotelName','HotelStars','DayDiff',
           'WeekDay','DiscountDiff','DiscountPerc']
df = pd.read_csv('hotels_data_Changed.csv', names=colnames, header=None).head(10)

print(df)

df2 = df[['SnapshotDate','CheckinDate','DayDiff','HotelName','WeekDay', 'DiscountDiff', 'DiscountPerc']]

print(df2)

print("-------------------------------------------------")

# dfHotelNames = df.groupby(['HotelName', 'HotelStars']).count()
#
pysql = lambda q: pdsql.sqldf(q, globals())
#
# query = 'select SnapshotDate,CheckinDate,DayDiff,HotelName,WeekDay,max(DiscountDiff),DiscountPerc ' \
#         'from df2 '\
#         'group by SnapshotDate,CheckinDate,DayDiff,HotelName,WeekDay,DiscountDiff,DiscountPerc '\
#         'order by max(DiscountDiff) desc'
#
# df1 = pysql(query)

data2 = [
        {'a': 1, 'b': 2, 'c': 3},
        {'a': 2, 'b': 2, 'c': 3},
        {'a': 1, 'b': 2, 'c': 4},
        {'a': 3, 'b': 2, 'c': 3},
         ]
df3 = pd.DataFrame(data2, columns=['a','b','c'])

print(df3)

#query2 = 'select a,b,c from df3'

query2 = 'select a,b,max(c) ' \
        'from df3 '\
        'group by a,b '\
        #'order by c desc'

df4 = pysql(query2)

print(df4)

#dfHotelNames = df.groupby('Hotel Name').agg({'Hotel Name': pd.np.size}).sort_values(['occurences'], ascending=False)

#print(dfHotelNames)

#dfHotelNames = df.groupby('Hotel Name').agg({'Hotel Name': pd.np.size}).sort_values(['occurences'], ascending=False).head(150)

#print(dfHotelNames)






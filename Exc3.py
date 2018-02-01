import pandas as pd
import pandasql as pdsql
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt


def df_crossjoin(df1, df2, **kwargs):
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1
    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    res.index = pd.MultiIndex.from_product((df1.index, df2.index))
    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)
    return res


colnames = ['SnapshotID', 'SnapshotDate', 'CheckinDate', 'Days', 'OriginalPrice', 'DiscountPrice', 'DiscountCode',
            'AvailableRooms',
            'HotelName', 'HotelStars']
df = pd.read_csv('hotels_data.csv', names=colnames, header=None, low_memory=False)
pysql = lambda q: pdsql.sqldf(q, globals())
print("-------------------QUEREY 1-------------------")
# 150 hotels most records
q1 = 'select HotelName, count(HotelName) ' \
     'from df ' \
     'group by HotelName ' \
     'order by count(HotelName) desc ' \
     'limit 150'

df1 = pysql(q1)
print("-------------------QUEREY 2-------------------")
# 40 check in dates most records
q2 = 'select CheckinDate, count(CheckinDate) ' \
     'from df ' \
     'group by CheckinDate ' \
     'order by count(CheckinDate) desc ' \
     'limit 40'

df2 = pysql(q2)

# DROP THE UNRELEVANT DATA FROM THE MAIN DATA FRAME
mdfq = 'select HotelName, CheckinDate,DiscountCode, min(DiscountPrice) ' \
       'from df ' \
       'where HotelName in (select HotelName from df1) ' \
       'and CheckinDate in (select CheckinDate from df2) ' \
       'group by HotelName, CheckinDate, DiscountCode'

names = {'HotelName': 'HotelName', 'CheckinDate': 'CheckinDate', 'DiscountCode': 'DiscountCode',
         'min(DiscountPrice)': 'DiscountPrice'}
ndf = pysql(mdfq).rename(columns=names)

print("-------------------QUEREY 3.1-------------------")
df31 = pysql('select distinct DiscountCode from df where DiscountCode in (1,2,3,4)')

print("-------------------QUEREY 3.2-------------------")
df32 = df_crossjoin(df2[['CheckinDate']], df31[["DiscountCode"]])

print("-------------------QUEREY 3.3-------------------")
df33 = df32
df33['datePlusCode'] = df32['CheckinDate'] + '_' + df32['DiscountCode']

print("-------------------QUEREY 3.4-------------------")
q34 = 'select a.HotelName, a.DiscountPrice, b.datePlusCode ' \
      'from ndf as a ' \
      'inner join df33 as b ' \
      'on a.DiscountCode=b.DiscountCode and a.CheckinDate=b.CheckinDate '
df34 = pysql(q34)
df34['DiscountPrice'] = df34['DiscountPrice'].astype('int')

print("-------------------NORMALIZE-------------------")
minPrice = pysql('select min(DiscountPrice) from df34')['min(DiscountPrice)'][0]
print('minPrice: ' + str(minPrice))
maxPrice = pysql('select max(DiscountPrice) from df34')['max(DiscountPrice)'][0]
print('maxPrice: ' + str(maxPrice))

df34['DiscountPrice'] = ((df34['DiscountPrice'] - minPrice) / (maxPrice - minPrice) * 100)

print("-------------------QUEREY 4-------------------")
df35 = df34.pivot(index='HotelName', columns='datePlusCode', values='DiscountPrice')
df35.fillna(value=-1, inplace=True)

print(df35)

df35.to_csv('pivot.csv')

df35.drop(df35.index[0], inplace=True)

print("-------------------CLUSTERING AND DENDOGRAM-------------------")
print()
# Calculate the distance between each sample
Z = hierarchy.linkage(df35, 'ward')
# Plot with Custom leaves
hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=5, labels=df35.index)
plt.show()
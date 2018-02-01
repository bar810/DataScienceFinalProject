import pandas as pd
import pandasql as pdsql
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt
from dateutil import parser


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
# print(hotelNamesDf)
print("-------------------QUEREY 2-------------------")
# 40 check in dates most records
q2 = 'select CheckinDate, count(CheckinDate) ' \
     'from df ' \
     'group by CheckinDate ' \
     'order by count(CheckinDate) desc ' \
     'limit 40'

df2 = pysql(q2)
# print(df2)

# DROP THE UNRELEVANT DATA FROM THE MAIN DATA FRAME
mdfq = 'select HotelName, CheckinDate,DiscountCode, min(DiscountPrice) ' \
       'from df ' \
       'where HotelName in (select HotelName from df1) ' \
       'and CheckinDate in (select CheckinDate from df2) ' \
       'group by HotelName, CheckinDate, DiscountCode'

names = {'HotelName': 'HotelName', 'CheckinDate': 'CheckinDate', 'DiscountCode': 'DiscountCode',
         'min(DiscountPrice)': 'DiscountPrice'}
ndf = pysql(mdfq).rename(columns=names)
# print(ndf)

print("-------------------QUEREY 3.1-------------------")
df31 = pysql('select distinct DiscountCode from df where DiscountCode in (1,2,3,4)')
# print(df31)

print("-------------------QUEREY 3.2-------------------")
df32 = df_crossjoin(df2[['CheckinDate']], df31[["DiscountCode"]])
# print(df32)

print("-------------------QUEREY 3.3-------------------")
df33 = df32
df33['datePlusCode'] = df32['CheckinDate'] + '_' + df32['DiscountCode']
# print(df33)

print("-------------------QUEREY 3.4-------------------")
q34 = 'select a.HotelName, a.DiscountPrice, b.datePlusCode ' \
      'from ndf as a ' \
      'inner join df33 as b ' \
      'on a.DiscountCode=b.DiscountCode and a.CheckinDate=b.CheckinDate '
df34 = pysql(q34)
# print(df34)

print("-------------------QUEREY 4-------------------")
df35 = df34.pivot(index='HotelName', columns='datePlusCode', values='DiscountPrice')
df35.fillna(value=-1, inplace=True)  # TODO replace it to the step before.
df35.to_csv('pivot.csv')
dfx = pd.read_csv('pivot.csv')
# print(dfx)
print("-------------------NORMALIZE-------------------")
hotelNameList = dfx['HotelName']
columns_names = dfx.columns.values
dfn = dfx.drop('HotelName', 1)
scalar = preprocessing.MinMaxScaler(copy=True, feature_range=(0, 100))
scaled_df = scalar.fit_transform(dfn)
scaled_df = pd.DataFrame(scaled_df)
scaled_df.insert(0, 'HotelName', hotelNameList)
scaled_df.to_csv('pivot_normalize.csv')
scaled_df = pd.read_csv('pivot_normalize.csv', names=columns_names)
scaled_df.drop(scaled_df.index[0], inplace=True)
scaled_df.to_csv('pivot_normalize.csv')

print("-------------------CLUSTERING AND DENDOGRAM-------------------")
scaled_df = scaled_df.set_index('HotelName')
print()
del scaled_df.index.name
# Calculate the distance between each sample
Z = hierarchy.linkage(scaled_df, 'ward')
# Plot with Custom leaves
hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=5, labels=scaled_df.index)
plt.show()

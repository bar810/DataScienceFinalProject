import pandas as pd
import pandasql as pdsql
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

colnames = ['SnapshotID','SnapshotDate','CheckinDate','Days','OriginalPrice','DiscountPrice','DiscountCode','AvailableRooms',
            'HotelName','HotelStars']
df = pd.read_csv('hotels_data.csv', names=colnames, header=None)
pysql = lambda q: pdsql.sqldf(q, globals())
# df=df.head(10000)
print("-------------------QUEREY 1-------------------")
#150 hotels most records
q1 = 'select HotelName, count(HotelName) ' \
            'from df '\
            'group by HotelName '\
            'order by count(HotelName) desc '\
            'limit 150'

df1 = pysql(q1)
# print(hotelNamesDf)
print("-------------------QUEREY 2-------------------")
#40 check in dates most records
q2 = 'select CheckinDate, count(CheckinDate) ' \
            'from df '\
            'group by CheckinDate '\
            'order by count(CheckinDate) desc '\
            'limit 40'

df2 = pysql(q2)
# print(df2)

# DROP THE UNRELEVANT DATA FROM THE MAIN DATA FRAME
mdfq= 'select HotelName, CheckinDate,DiscountCode, min(DiscountPrice) '\
        'from df '\
        'where HotelName in (select HotelName from df1) '\
        'and CheckinDate in (select CheckinDate from df2) '\
        'group by HotelName, CheckinDate, DiscountCode'

names = {'HotelName':'HotelName', 'CheckinDate':'CheckinDate','DiscountCode':'DiscountCode', 'min(DiscountPrice)':'DiscountPrice'}
ndf = pysql(mdfq).rename(columns=names)
# print(ndf)

print("-------------------QUEREY 3.1-------------------")
df31 = pysql('select distinct DiscountCode from df where DiscountCode in (1,2,3,4)') #TODO - fix!
# print(df31)

print("-------------------QUEREY 3.2-------------------")
df32=df_crossjoin(df2[['CheckinDate']], df31[["DiscountCode"]])
# print(df32)

print("-------------------QUEREY 3.3-------------------")
df33=df32
df33['datePlusCode']=df32['CheckinDate']+'_'+df32['DiscountCode']
# print(df33)

print("-------------------QUEREY 3.4-------------------")
q34='select a.HotelName, a.DiscountPrice, b.datePlusCode '\
    'from ndf as a '\
    'inner join df33 as b '\
    'on a.DiscountCode=b.DiscountCode and a.CheckinDate=b.CheckinDate '
df34 = pysql(q34)
# print(df34)

print("-------------------QUEREY 4-------------------")
df35=df34.pivot(index='HotelName', columns='datePlusCode', values='DiscountPrice')
df35.fillna(value=-1,inplace=True) # TODO replace it to the step before.
df35.to_csv('pivot.csv')
dfx = pd.read_csv('pivot.csv')
# print(dfx)
print("-------------------NORMALIZE-------------------")
hotelNameList=dfx['HotelName']
columns_names=dfx.columns.values
dfn= dfx.drop('HotelName', 1)
scalar=preprocessing.MinMaxScaler(copy = True, feature_range=(0, 100))
scaled_df=scalar.fit_transform(dfn)
scaled_df=pd.DataFrame(scaled_df)
scaled_df.insert(0, 'HotelName', hotelNameList)
scaled_df.to_csv('pivot_normalize.csv')
scaled_df = pd.read_csv('pivot_normalize.csv',names=columns_names)
scaled_df.drop(scaled_df.index[0], inplace=True)
scaled_df.to_csv('pivot_normalize.csv')

print("-------------------CLUSTERING-------------------")
minDate = parser.parse('2015-10-02')
maxDate = parser.parse('2016-01-01')
# print(minDate)


# #TODO add this part
clslist=[]
for col in scaled_df.columns.values:
    for row in scaled_df.itertuples():
        i=scaled_df.columns.get_loc(col)
        if i>1 and row[scaled_df.columns.get_loc(col)] > 0 :#and row[scaled_df.columns.get_loc(col)] < 10:
            currentDate = parser.parse(str(col).split('_')[0])
            dayDiff = (currentDate - minDate).days
            clslist.append([dayDiff, format(row[scaled_df.columns.get_loc(col)], '.2f')])


# colors = [int(i % 3) for i in scaled_df['DiscountCode']]
# pylab.scatter(xy[0], xy[1], c=colors)
clslist = np.asarray(clslist)


N = 8207
area = np.pi * (15 * np.random.rand(N))**2 # 0 to 15 point radii
#print(clsList[:,1])
colors = np.random.rand(N)
plt.yticks(rotation=70)
plt.xticks(np.arange(0, 59, 1), rotation='vertical')
plt.scatter(clslist[:,0], clslist[:,1],alpha=0.5, c=colors)
# plt.show()

#DENDOGRAM
plt.figure(figsize=(25,10))
plt.title("nn")
plt.xlabel("jj")
plt.ylabel("hh")
z=linkage(clslist,'ward')
dendrogram(z)
plt.show()
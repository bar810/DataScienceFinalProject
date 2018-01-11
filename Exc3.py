import pandas as pd
import pandasql as pdsql
from sklearn import preprocessing


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
        'where HotelName in(select HotelName from df1) '\
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
    'left join df33 as b '\
    'on a.DiscountCode=b.DiscountCode and a.CheckinDate=b.CheckinDate '
df34 = pysql(q34)
# print(df34)

# print("-------------------QUEREY 4-------------------")
df35=df34.pivot(index='HotelName', columns='datePlusCode', values='DiscountPrice')
df35.fillna(value=-1,inplace=True)
# print(df35)

# print("-------------------NORMALIZE-------------------")
scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(df35)
scaled_df = pd.DataFrame(scaled_df) #TODO cancel header normalize
scaled_df=scaled_df*100
# print(scaled_df)

# print("-------------------CLUSTERING-------------------")
#TODO add this part

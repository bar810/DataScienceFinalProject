import pandas as pd

# RELOAD THE ORIGINAL CSV
df = pd.read_csv('hotels_data.csv')
# ADD COLUMNS DAY DIFF - THE DAYS BETWEEN SNAPSHOT DATE TO CHECKIN DATE
df['DayDiff1'] = pd.to_datetime(df['Checkin Date'])
df['DayDiff2'] = pd.to_datetime(df['Snapshot Date'])
df['DayDiff']=(df['DayDiff1'] - df['DayDiff2']).dt.days
del df['DayDiff1']
del df['DayDiff2']

# ADD COLUMNS WEEKDAY- FIND THE SPECIFIC DAY BY THE CHECKIN DATE
df = df.reset_index()
df['WeekDay'] = pd.to_datetime(df['Checkin Date']).dt.weekday_name

#ADD COLUMNS DISCOUNTDIFF - THE DIFFRENT BETWEEN ORIGINAL PRICE TO DISCOUNT PRICE
df['DiscountDiff']= pd.to_numeric(df['Original Price'])-pd.to_numeric(df['Discount Price'])

#ADD COLUMNS DISCOUNTPERC - THE PERCENT OF THE DISCOUNT
df['DiscountPerc']=(pd.to_numeric(df['DiscountDiff'])/pd.to_numeric(df['Original Price']))*100

df.to_csv('Hotels_data_Changed.csv')

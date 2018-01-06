import pandas as pd

df = pd.read_csv('hotels_data.csv')

df['DayDiff'] = df.assign(df['Snapshot Date'] - df['Checkin Date'])

print(df)

##df.to_csv('Hotels_data_Changed.csv')


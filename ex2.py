import pandas as pd
#
# # THIE PART TAKE THE WHOLE TABLE AND DROP THE UNNECCERY COLUMNS--> SAVE IT AS NEW CSV FILE
# df = pd.read_csv('Hotels_data_Changed.csv')
# keep_col = ['Snapshot Date', 'Checkin Date', 'Discount Code', 'Hotel Name', 'DayDiff', 'WeekDay', 'DiscountDiff']
# cf = df[keep_col]
# # TODO remove it!! - for testing only
# cf.to_csv("ClassificationTable.csv")
cf = pd.read_csv('ClassificationTable1.csv')
print(cf.count())

# THIS ROW MAKE GROUP BY QUERY AND DROP THE ROWS WITH LOW DISCOUNT
kk = cf.groupby(['Snapshot Date', 'Checkin Date', 'Discount Code', 'Hotel Name', 'DayDiff', 'WeekDay'])#.agg({'DiscountDiff':'max'})
print("---------")

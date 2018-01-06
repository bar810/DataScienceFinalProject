import Exc1
import pandas as pd

def main():
    keep_col = ['Snapshot Date', 'Checkin Date', 'Hotel Name']#, 'DayDiff',  'Hotel Name', 'WeekDay']
   #print(keep_col)
    new_f = Exc1.df[keep_col]
   #print(new_f)
    new_f.to_csv("Classification.csv")
    classi = pd.read_csv('Classification.csv')

    print(classi)

# df.to_csv('Hotels_data_Changed.csv')

main()
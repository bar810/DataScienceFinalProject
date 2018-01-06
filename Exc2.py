import pandas as pd
import csv
import operator

df = pd.read_csv('Hotels_data_Changed.csv')


def main():

    keep_col = ['Snapshot Date', 'Checkin Date', 'Hotel Name', 'DayDiff', 'WeekDay', 'Discount Code', 'DiscountDiff']
    new_f = df[keep_col].head(10)
    print(new_f)

    print("---------------------------------------------------------------")

    new_f.to_csv("Classification.csv")
    classi = pd.read_csv('Classification.csv')
    print(classi)



    print("---------------------------------------------------------------")

    columns = ['Snapshot Date', 'Checkin Date', 'Hotel Name', 'DayDiff', 'WeekDay', 'Discount Code']
    new_f = new_f.groupby(columns)

    print("---------------------------------------------------------------")

    sort = sorted(new_f,key=operator.itemgetter(6))
    print(sort)

    new_f.to_csv("new.csv")
    a = pd.read_csv('new.csv')
    print(a)


main()

import pandas as pd
import pandasql as pdsql

data2 = [
        {'a': 1, 'b': 2, 'c': 3},
        {'a': 2, 'b': 2, 'c': 3},
        {'a': 1, 'b': 2, 'c': 4},
        {'a': 3, 'b': 2, 'c': 3},
         ]

##df3 = pd.DataFrame(data2, columns=['a','b','c'])

df3 = pd.read_csv('test.csv')

print(df3)

#query2 = 'select a,b,c from df3'

query2 = 'select a,b,max(c) ' \
        'from df3 '\
        'group by a,b '\
        #'order by c desc'

pysql = lambda q: pdsql.sqldf(q, globals())

df4 = pysql(query2)

print(df4)
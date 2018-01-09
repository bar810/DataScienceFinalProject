import pandas as pd
import pandasql as pdsql
import pandas as pd
from sklearn import tree, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
pysql = lambda q: pdsql.sqldf(q, globals())

# THIS PART TAKE THE WHOLE TABLE AND DROP THE UNNECCERY COLUMNS
df = pd.read_csv('Hotels_data_Changed.csv')
keep_col = ['Snapshot Date', 'Checkin Date', 'Discount Code', 'Hotel Name', 'DayDiff', 'WeekDay', 'DiscountDiff']
df.rename(columns={'Snapshot Date': 'SnapshotDate'}, inplace=True)
df.rename(columns={'Checkin Date': 'CheckinDate'}, inplace=True)
df.rename(columns={'Discount Code': 'DiscountCode'}, inplace=True)
df.rename(columns={'Hotel Name': 'HotelName'}, inplace=True)
keep_col = ['SnapshotDate', 'CheckinDate', 'DiscountCode', 'HotelName', 'DayDiff', 'WeekDay', 'DiscountDiff']
cf = df[keep_col]

# GROUP BY QUERY- DROP THE EXPENSIVE VECTORS
query = 'select SnapshotDate, CheckinDate, DiscountCode, HotelName, DayDiff, WeekDay, max(DiscountDiff) from cf group by SnapshotDate, CheckinDate, HotelName, DayDiff, WeekDay'
df = pysql(query)

# PART 2.2
df=df.head(100)
# GET ONLY THE FEATURES I NEED
features = ['SnapshotDate', 'CheckinDate', 'HotelName', 'WeekDay', 'DayDiff']

# DROP THE ROWS WITH MISSING VALUES
df = df.dropna()

# CONVERT THE FEATUERS TO NUMERIC: to translte back: le.transform(THE FITCHER NAME)
translate1 = lambda row: wde.transform([row])[0]
wde = preprocessing.LabelEncoder()
wde.fit(df['WeekDay'])

translate2 = lambda row: hne.transform([row])[0]
hne = preprocessing.LabelEncoder()
hne.fit(df['HotelName'])

translate3 = lambda row: cde.transform([row])[0]
cde = preprocessing.LabelEncoder()
cde.fit(df['CheckinDate'])

translate4 = lambda row: sde.transform([row])[0]
sde = preprocessing.LabelEncoder()
sde.fit(df['SnapshotDate'])

df['WeekDay'] = df['WeekDay'].apply(translate1)
df['HotelName'] = df['HotelName'].apply(translate2)
df['CheckinDate'] = df['CheckinDate'].apply(translate3)
df['SnapshotDate'] = df['SnapshotDate'].apply(translate4)

# INSERT THE VALUES INTO VARIABLES
X = df[features]
y = df["DiscountCode"]

# DECISION TREE CLASSIFIER
print("------------------------DECISION TREE-----------------------")
print()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

# TEST THE ALGORITHEM AND SHOW STATISTICS
y_predict = model.predict(X_test)

#CONFUSION MATRIX
matrix=pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Predicted 1', 'Predicted 2','Predicted 3','Predicted 4'],
    index=['True 1', 'True 2','True 3','True 4']
)
print("-------------------------STATISTICS-------------------------")
print("confusion_matrix:")
print(matrix)
print("------------------------------------------------------------")
# accuracy
accuracy=accuracy_score(y_test, y_predict)
print("accuracy is: %s" %(accuracy))
print("------------------------------------------------------------")
# TP
tp = np.diag(matrix)
print("TP is: %s" %(tp))
print("------------------------------------------------------------")
# FP
fp=matrix.sum(axis=0)-np.diag(matrix)
print("FP:")
print(fp)
print("------------------------------------------------------------")
# FN
fn = matrix.sum(axis=1) - np.diag(matrix)
print("FN:")
print(fn)
print("------------------------------------------------------------")
# ROC
print("ROC:")
#TODO add this
print("------------------------------------------------------------")
print()

#TODO fix that. its look strange
# NAIVE BAYES CLASSIFIER
print("-------------------------NAIVE BAYES------------------------")
clf = GaussianNB()
print(clf.fit(X, y))
print(clf.predict([[10,46,274,21,4]]))

dlf = BernoulliNB()
dlf.fit(X, y)
BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True)
print(dlf.predict([[0,29,253,10,1]]))

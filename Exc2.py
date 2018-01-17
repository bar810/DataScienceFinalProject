import matplotlib.pyplot as plt
import nltk as nltk
import numpy as np
import pandas as pd
import pandasql as pdsql
import pydotplus
import scikitplot as skplt
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz

pysql = lambda q: pdsql.sqldf(q, globals())

# THIS PART TAKE THE WHOLE TABLE AND DROP THE UNNECCERY COLUMNS
df = pd.read_csv('Hotels_data_Changed.csv')
# df = pd.read_csv('Hotels_data_Changed_minimaized.csv') # for naive bayse after clean the data
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
# GET ONLY THE FEATURES I NEED
features = ['SnapshotDate', 'CheckinDate', 'HotelName', 'WeekDay', 'DayDiff']
df=df.head(1000)
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
columns_names=X.columns.values

# DECISION TREE CLASSIFIER
print("------------------------DECISION TREE-----------------------")
print()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

# TEST THE ALGORITHEM AND SHOW STATISTICS
y_predict = model.predict(X_test)

# CONFUSION MATRIX
matrix = pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Predicted 1', 'Predicted 2', 'Predicted 3', 'Predicted 4'],
    index=['True 1', 'True 2', 'True 3', 'True 4']
)

print("-------------------------STATISTICS-------------------------")
print("confusion_matrix:")
print(matrix)
print("------------------------------------------------------------")
# accuracy
accuracy = accuracy_score(y_test, y_predict)
print("accuracy is: %s" % (accuracy))
print("------------------------------------------------------------")
# TP
tp = np.diag(matrix)
print("TP is: %s" % (tp))
print("------------------------------------------------------------")
# FP
fp = matrix.sum(axis=0) - np.diag(matrix)
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
# This is the ROC curve
y_predict2 = model.predict_proba(X_test)
skplt.metrics.plot_roc_curve(y_test, y_predict2)
plt.show()
print("see diagram")
print("------------------------------------------------------------")
print()

# # DRAW THE TREE
# dot_data = tree.export_graphviz(model,
#                                 feature_names=features,
#                                 out_file=None,
#                                 filled=True,
#                                 rounded=True)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_png('tree.png')


# NAIVE BAYES CLASSIFIER
print("-------------------------NAIVE BAYES------------------------")
nb = GaussianNB()
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, random_state=1)
nb.fit(X1_train, y1_train)
predicted = nb.predict(X1_test)
predicted_probas = nb.predict_proba(X1_test)

matrix=pd.DataFrame(
    confusion_matrix(y1_test, predicted),
    columns=['Predicted 1', 'Predicted 2','Predicted 3','Predicted 4'],
    index=['True 1', 'True 2','True 3','True 4']
)

print("-------------------------STATISTICS-------------------------")
print("confusion_matrix:")
print(matrix)
print("------------------------------------------------------------")
# accuracy
accuracy=accuracy_score(y1_test, predicted)
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
# This is the ROC curve
skplt.metrics.plot_roc_curve(y1_test, predicted_probas)
plt.show()
print("see diagram")
print("------------------------------------------------------------")
print()

print("-------------------------KNN------------------------")
print()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
#
# TEST THE ALGORITHEM AND SHOW STATISTICS
y_predict = knn.predict(X_test)

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
# This is the ROC curve
y_predict2 = knn.predict_proba(X_test)
skplt.metrics.plot_roc_curve(y_test, y_predict2)
plt.show()
print("see diagram")
print("------------------------------------------------------------")
print()

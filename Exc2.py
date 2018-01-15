import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandasql as pdsql
import pydotplus
import scikitplot as skplt
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz


def visualize_tree(tree, feature_names):
    from sklearn.tree import  export_graphviz
    import subprocess
    with open("tree.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "tree.dot", "-o", "tree.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

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
df = df.head(10000)
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
# visualize_tree(model,features)















# # #TODO fix that. its look strange
# # NAIVE BAYES CLASSIFIER
# print("-------------------------NAIVE BAYES------------------------")
# nb =MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
# X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, random_state=1)
# nb.fit(X1_train, y1_train)
# predicted = nb.predict(X1_test)
# predicted_probas = nb.predict_proba(X1_test)
#
# matrix=pd.DataFrame(
#     confusion_matrix(y1_test, predicted),
#     columns=['Predicted 1', 'Predicted 2','Predicted 3','Predicted 4'],
#     index=['True 1', 'True 2','True 3','True 4']
# )
#
# print("-------------------------STATISTICS-------------------------")
# print("confusion_matrix:")
# print(matrix)
# print("------------------------------------------------------------")
# # accuracy
# accuracy=accuracy_score(y1_test, predicted)
# print("accuracy is: %s" %(accuracy))
# print("------------------------------------------------------------")
# # TP
# tp = np.diag(matrix)
# print("TP is: %s" %(tp))
# print("------------------------------------------------------------")
# # FP
# fp=matrix.sum(axis=0)-np.diag(matrix)
# print("FP:")
# print(fp)
# print("------------------------------------------------------------")
# # FN
# fn = matrix.sum(axis=1) - np.diag(matrix)
# print("FN:")
# print(fn)
# print("------------------------------------------------------------")
# # ROC
# print("ROC:")
# # This is the ROC curve
# skplt.metrics.plot_roc_curve(y1_test, predicted_probas)
# plt.show()
# print("see diagram")
# print("------------------------------------------------------------")
# print()


# print("-------------------------KNN------------------------")
# print()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_train, y_train)
# #
# # TEST THE ALGORITHEM AND SHOW STATISTICS
# y_predict = knn.predict(X_test)
#
# #CONFUSION MATRIX
# matrix=pd.DataFrame(
#     confusion_matrix(y_test, y_predict),
#     columns=['Predicted 1', 'Predicted 2','Predicted 3','Predicted 4'],
#     index=['True 1', 'True 2','True 3','True 4']
# )
# print("-------------------------STATISTICS-------------------------")
# print("confusion_matrix:")
# print(matrix)
# print("------------------------------------------------------------")
# # accuracy
# accuracy=accuracy_score(y_test, y_predict)
# print("accuracy is: %s" %(accuracy))
# print("------------------------------------------------------------")
# # TP
# tp = np.diag(matrix)
# print("TP is: %s" %(tp))
# print("------------------------------------------------------------")
# # FP
# fp=matrix.sum(axis=0)-np.diag(matrix)
# print("FP:")
# print(fp)
# print("------------------------------------------------------------")
# # FN
# fn = matrix.sum(axis=1) - np.diag(matrix)
# print("FN:")
# print(fn)
# print("------------------------------------------------------------")
# # ROC
# print("ROC:")
# # This is the ROC curve
# y_predict2 = knn.predict_proba(X_test)
# skplt.metrics.plot_roc_curve(y_test, y_predict2)
# plt.show()
# print("see diagram")
# print("------------------------------------------------------------")
# print()

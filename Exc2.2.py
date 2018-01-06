import pandas as pd
import scipy
from pandas import DataFrame
from sklearn import tree, preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB

# LOAD THE DATA FRAME
df = pd.read_csv('Classification.csv')
df=df.head(1000) # TODO  delete this

# GET ONLY THE FEATURES I NEED
features = list(df.columns[1:6])  # TODO should be changed by ortal final csv
# DROP THE ROWS WITH MISSING VALUES
df = df.dropna()

# INSERT THE VALUES INTO VARIABLES
X = df[features]
y = df["Discount Code"]

# CONVERT THE FEATUERS TO NUMERIC: to translte back: le.transform(THE FITCHER NAME)
translate1 = lambda row: wde.transform([row])[0]
wde = preprocessing.LabelEncoder()
wde.fit(X['WeekDay'])

translate2 = lambda row: hne.transform([row])[0]
hne = preprocessing.LabelEncoder()
hne.fit(X['Hotel Name'])

translate3 = lambda row: cde.transform([row])[0]
cde = preprocessing.LabelEncoder()
cde.fit(X['Checkin Date'])

translate4 = lambda row: sde.transform([row])[0]
sde = preprocessing.LabelEncoder()
sde.fit(X['Snapshot Date'])

X['WeekDay'] = X['WeekDay'].apply(translate1)
X['Hotel Name'] = X['Hotel Name'].apply(translate2)
X['Checkin Date'] = X['Checkin Date'].apply(translate3)
X['Snapshot Date'] = X['Snapshot Date'].apply(translate4)

# DECISION TREE CLASSIFIER
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
# # FP
fp=matrix.sum(axis=0)-np.diag(matrix)
print("FP:")
print(fp)
print("------------------------------------------------------------")
# # FN
fn = matrix.sum(axis=1) - np.diag(matrix)
print("FN:")
print(fn)
print("------------------------------------------------------------")
# # ROC
print("ROC:")


print("------------------------------------------------------------")




# NAIVE BAYES CLASSIFIER
# print(X)
# print(y)

# clf = GaussianNB()
# print(clf.fit(X, y))
# print(clf.predict([[10,46,274,21,4]]))

# dlf = BernoulliNB()
# dlf.fit(X, y)
# BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True)
# print(dlf.predict([[0,29,253,10,1]]))

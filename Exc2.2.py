import pandas as pd
import scipy
from pandas import DataFrame
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction import DictVectorizer


#LOAD THE DATA FRAME
df = pd.read_csv('hotels_data_Changed.csv')

#GET ONLY THE FEATURES I NEED
features = list(df.columns[:3])

# DROP THE ROWS WITH MISSING VALUES
df = df.dropna()

# DECISION TREE CLASSIFIER
X = df[features]
y = df["Discount Code"]

print(X)



#
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# model = tree.DecisionTreeClassifier()
#
# print(model)
#
#
# # FIT OUR MODEL USING OUR TRAINING DATA
# model.fit(X_train, y_train)
#
#



# y_predict = model.predict(X_test)
#
# accuracy_score(y_test, y_predict)
#
# pd.DataFrame(
#     confusion_matrix(y_test, y_predict),
#     columns=['Predicted Not Survival', 'Predicted Survival'],
#     index=['True Not Survival', 'True Survival']
# )
#
#








#
# ##naive bayes
#
# iris = tree.load_iris()
# gnb = GaussianNB()
# y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
# print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))


import scipy
from pandas import DataFrame
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

# df = DataFrame
# features = list(df.columns[:4])

##tree

# y = df["Target"]
# X = df[features]
# dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
# dt.fit(X, y)

##naive bayes

iris = datasets.load_iris()
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))

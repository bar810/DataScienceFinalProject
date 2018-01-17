from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession

sc = SparkSession.builder.appName("FinalProject").master("local[*]").getOrCreate()

lines = sc.read.text('text.txt').rdd
parsed = lines.map(lambda l: MLUtils._parse_libsvm_line(l))

parsed.cache()
numFeatures = parsed.map(lambda x: -1 if x[1].size == 0 else x[1][-1]).reduce(max) + 1
data = parsed.map(lambda x: LabeledPoint(x[0], Vectors.sparse(numFeatures, x[1], x[2])))

# Load and parse the data file into an RDD of LabeledPoint.
#data = MLUtils.loadLibSVMFile(sc, 'text.txt')
# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                    impurity='variance', maxDepth=5, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testMSE = labelsAndPredictions.map(lambda lp: (lp[0] - lp[1]) * (lp[0] - lp[1])).sum() /\
    float(testData.count())
print('Test Mean Squared Error = ' + str(testMSE))
print('Learned regression tree model:')
print(model.toDebugString())

# Save and load model
model.save(sc, "target/tmp/myDecisionTreeRegressionModel")
sameModel = DecisionTreeModel.load(sc, "target/tmp/myDecisionTreeRegressionModel")
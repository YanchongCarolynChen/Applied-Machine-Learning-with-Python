name = 'My name is Mike'
print(name[0])
print(name[-4])
print(name[11:14])
print(name[11:15])
print(name[5:])
print(name[:5])

# learn how to use xgboost
# tutorial is from https://blog.csdn.net/u011630575/article/details/79418138
import xgboost as xgb
from sklearn.metrics import accuracy_score

working_path = ""
dtrain = xgb.DMatrix(working_path + 'agaricus.txt.train')
dtest = xgb.DMatrix(working_path + 'agaricus.txt.test')

dtrain.num_col()
dtrain.num_row()
dtest.num_row()

param = {'max_depth' : 2, 'eta' : 1, 'silent' : 0, 'objective' : 'binary:logistic'}
print(param)

num_round = 2
import time
start_time = time.clock()
bst = xgb.train(param, dtrain, num_round)
end_time = time.clock()
print(end_time - start_time)

train_preds = bst.predict(dtrain)
train_predictions = [round(value) for value in train_preds]
y_train = dtrain.get_label()
train_accuracy = accuracy_score(y_train, train_predictions)
print("train accuracy: %.2f%%" % (train_accuracy * 100.0))

preds = bst.predict(dtest)
predictions = [round(value) for value in preds]

y_test = dtest.get_label()
test_accuracy = accuracy_score(y_test, predictions)
print("test accuracy %.2f%%" % (test_accuracy * 100.0))

# visualize result
from matplotlib import pyplot
import graphviz
xgb.plot_tree(bst, num_trees = 0, rankdir = 'LR')
pyplot.show()


import numpy as np
import twigy
import pandas as pd

# read and prepare training data
titanic_data = pd.read_csv("./test/testdata/titanic_data.csv")
X = titanic_data.drop(columns=['Survived']).to_numpy()
y = np.reshape(titanic_data[['Survived']].to_numpy(), (-1))

# initialize, train and print decision tree classifier
tree_clf = twigy.DecisionTreeClassifier()
tree_clf.build_tree(X, y)
tree_clf.print_tree()

# predict class probabilities on training data
preds = tree_clf.predict_classes(X)

# initialize, train and print random forest classifier
forest_clf = twigy.RandomForestClassifier()
forest_clf.build_forest(X, y)

# predict class probabilities on training data
preds = forest_clf.predict_classes(X)

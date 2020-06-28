import pandas as pd
import pytest
import twigy


@pytest.fixture
def titanic_training_data():
    titanic_data = pd.read_csv("/test/testdata/titanic_data.csv")
    X = titanic_data.drop(columns=['Survived'])
    y = titanic_data[['Survived']]
    return X.to_numpy(), y.to_numpy()


def test_result(titanic_training_data):
    tree_clf = twigy.DecisionTreeClassifier(max_depth=5)
    tree_clf.build_tree(titanic_training_data[0], titanic_training_data[1])

    assert len(tree_clf.tree.nodes) == 53
    assert tree_clf.tree.nodes[0].n_samples == 891
    assert tree_clf.tree.nodes[0].split_feature == 5
    assert tree_clf.tree.nodes[0].threshold == 1.5
    assert tree_clf.tree.nodes[0].left_child_id == 1
    assert tree_clf.tree.nodes[0].right_child_id == 26
    assert tree_clf.tree.nodes[0].impurity == pytest.approx(0.473013)
    assert tree_clf.tree.nodes[0].value == [549, 342]

    assert tree_clf.tree.nodes[22].n_samples == 32
    assert tree_clf.tree.nodes[22].impurity == pytest.approx(0.0)
    assert tree_clf.tree.nodes[22].value == [32, 0]

    assert tree_clf.tree.nodes[49].n_samples == 21
    assert tree_clf.tree.nodes[49].split_feature == 2
    assert tree_clf.tree.nodes[49].threshold == 0.5
    assert tree_clf.tree.nodes[49].left_child_id == 50
    assert tree_clf.tree.nodes[49].right_child_id == 51
    assert tree_clf.tree.nodes[49].impurity == pytest.approx(0.17233559)
    assert tree_clf.tree.nodes[49].value == [19, 2]


def test_neg_max_depth(titanic_training_data):
    tree_clf = twigy.DecisionTreeClassifier(max_depth=-5)
    tree_clf.build_tree(titanic_training_data[0], titanic_training_data[1])
    assert len(tree_clf.tree.nodes) == 1


def test_max_depth(titanic_training_data):
    tree_clf = twigy.DecisionTreeClassifier(max_depth=1)
    tree_clf.build_tree(titanic_training_data[0], titanic_training_data[1])
    assert len(tree_clf.tree.nodes) == 3


def test_max_depth(titanic_training_data):
    tree_clf = twigy.DecisionTreeClassifier(max_depth=5, max_features=1)
    tree_clf.build_tree(titanic_training_data[0], titanic_training_data[1])
    assert tree_clf.tree.nodes[0].split_feature == 1
    assert tree_clf.tree.nodes[1].split_feature == 7
    assert tree_clf.tree.nodes[2].split_feature == 6
    assert tree_clf.tree.nodes[3].split_feature == 2
    assert tree_clf.tree.nodes[4].split_feature == 7


def test_high_min_impurity_split(titanic_training_data):
    tree_clf = twigy.DecisionTreeClassifier(max_depth=5, min_impurity_split=100)
    tree_clf.build_tree(titanic_training_data[0], titanic_training_data[1])
    assert len(tree_clf.tree.nodes) == 1


def test_normal_min_impurity_split(titanic_training_data):
    tree_clf = twigy.DecisionTreeClassifier(max_depth=5, min_impurity_split=0.1)
    tree_clf.build_tree(titanic_training_data[0], titanic_training_data[1])
    assert len(tree_clf.tree.nodes) == 47

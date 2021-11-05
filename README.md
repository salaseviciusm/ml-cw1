# Usage
```python
from main import read_data
from main import split_dataset
from tree import decision_tree_learning, predict, prune_tree
# load in the dataset from filepath into dataset
dataset = read_data("filepath.txt")
# get training, validation and test numpy datasets with 90% split
training, test = split_dataset(dataset, 90)
training, validation = split_dataset(training, 90)
# store the decision tree trained on the dataset
dtree = decision_tree_learning(dataset=training)
# load testcase as numpy array here, however you'd like as a numpy 1 row with every attribute
tc = read_data("single_testcase.txt") 
# predict using the tree
predict(tree=dtree, attributes=tc)
# prune a given tree
prune_tree(node=dtree, validation_set=validation)
```
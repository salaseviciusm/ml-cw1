import numpy as np


def room_numbers(dataset):
    """
    Finds unique labels from data.
    :param dataset: 'numpy.ndarray' of full dataset
    :return: 'numpy.ndarray' 1D array of unique labels
    """
    return np.unique(dataset[:, -1])


def decision_tree_learning(dataset, depth=0):
    """
    Generates a decision tree from a training set, by maximising information gain at each split
    :param dataset: 'numpy.ndarray' of the training set
    :param depth: 'int' of the current depth of the decision tree
    :return: 'dict' as a nested dictionary of the decision tree
    """
    if dataset.size == 0:
        return None, depth
    if len(np.unique(dataset[:, -1])) == 1:
        return dict({
            "a_value": np.unique(dataset[:, -1])[0],
            "is_leaf": True
        }), depth
    else:
        column_num, value = find_split(dataset)
        l_branch, l_depth = decision_tree_learning(
            dataset[dataset[:, column_num] < value, :], depth + 1)
        r_branch, r_depth = decision_tree_learning(
            dataset[dataset[:, column_num] >= value, :], depth + 1)
        node = {
            "attribute": column_num,
            "a_value": value,
            "left": l_branch,
            "right": r_branch,
            "is_leaf": False
        }
        return node, max(l_depth, r_depth)


def find_split(dataset):
    """
    Find split iterates over every possible interval, i.e. where the attributes differ, and calculates the
    information gain at each split. The optimal split is the split which partitions the data set to provide the
    largest information gain.
    :param dataset: 'numpy.ndarray' of the dataset being operated on
    :return: 'tuple' of the (feature, split_value)
    """
    labels = dataset[:, -1]
    split = None
    split_attribute = None
    n_cols = dataset.shape[1]
    max_gain = 0
    for attr_idx in range(n_cols - 1):
        attribute = dataset[:, attr_idx]
        sort_index = np.argsort(attribute)
        sort_labels, sort_attr = labels[sort_index], attribute[sort_index]
        for i in range(1, len(sort_labels)):
            if sort_attr[i] != sort_attr[i - 1]:
                left = sort_labels[:i]
                right = sort_labels[i:]
                curr_gain = gain(sort_labels, left, right)
                if curr_gain > max_gain:
                    max_gain = curr_gain
                    split = (sort_attr[i] + sort_attr[i - 1]) / 2
                    split_attribute = attr_idx
    return split_attribute, split


def entropy(dataset):
    """
    Calculates the entropy of a given dataset.
    .. math:: -\sum_{k=1}^{K}p_{k}\cdot \log_2 p_{k}
    :param dataset: 'numpy.ndarray' of the dataset being operated on
    :return: 'float' of the entropy of a dataset given by label frequencies
    """
    _, count_arr = np.unique(dataset, return_counts=True)
    n_samples = len(dataset)
    h = 0
    for freq in count_arr:
        p_k = freq / n_samples
        if p_k:
            h -= p_k * np.log2(p_k)
    return h


def remainder(left, right):
    """
    Returns the remainder term according to the formula below.
    .. math:: \frac{\left| S_{left} \right|}{\left| S_{left} \right| + \left| S_{right} \right|}H(S_{left})
            + \frac{\left| S_{right} \right|}{\left| S_{left} \right| + \left| S_{right} \right|}H(S_{right})
    :param left: 'numpy.ndarray' of the left set, i.e. x < n
    :param right: 'numpy.ndarray' of the right set, i.e. x >= n
    :return: 'float' of the remainder term
    """
    l_size = len(left)
    r_size = len(right)
    return (l_size * entropy(left) + r_size * entropy(right)) / (l_size + r_size)


def gain(dataset, left, right):
    """
    Returns the overall information gain according to a given partition
    .. math:: H(S_{all}) - Remainder(S_{left}, S_{right})
    :param dataset: 'numpy.ndarray' of the operating full dataset
    :param left: 'numpy.ndarray' of the left partitioned dataset
    :param right: 'numpy.ndarray' of the right partitioned dataset
    :return: 'float' of the overall information gain
    """
    return entropy(dataset) - remainder(left, right)


def evaluate(test_db, trained_tree):
    correct = 0
    samples, _ = test_db.shape
    for row in test_db:
        label = row[-1]
        attributes = row[:-1]
        if predict(tree=trained_tree, attributes=attributes) == label:
            correct += 1
    return correct / samples


def prune_tree(validation_set, node, majority_attribute=-1.0):
    """
    Prune tree attempts to replace a terminal node with both the left and the right leaves and calculates the
    validation accuracy to determine acceptable prunes.
    :param majority_attribute: maintains the previous majority attribute of the validation set while not empty
    :param validation_set: 'numpy.ndarray' of the operational validation set
    :param node: 'dict' of the current active tree/node, as every tree's branch is its own tree
    :return: 'dict' of the pruned tree
    """
    if len(validation_set):
        labels = validation_set[:, -1].astype(int)
        majority_attribute = np.bincount(labels).argmax()
    if node["is_leaf"]:
        return node

    if not node["left"]["is_leaf"]:
        node["left"] = prune_tree(
            validation_set=validation_set[validation_set[:,
                                          node["attribute"]] < node["a_value"], :],
            node=node["left"],
            majority_attribute=majority_attribute
        )

    if not node["right"]["is_leaf"]:
        node["right"] = prune_tree(
            validation_set=validation_set[validation_set[:,
                                          node["attribute"]] >= node["a_value"], :],
            node=node["right"],
            majority_attribute=majority_attribute
        )

    if node["left"]["is_leaf"] and node["right"]["is_leaf"]:
        old_node = dict(node)
        if not len(validation_set):
            return dict({
                "a_value": majority_attribute,
                "is_leaf": True
            })
        prev = (evaluate(validation_set, old_node), old_node)
        left = (evaluate(validation_set, old_node["left"]), old_node["left"])
        right = (evaluate(validation_set,
                          old_node["right"]), old_node["right"])
        max_node = max([left, right, prev], key=lambda t: t[0])[1]
        node = max_node
    return node


def predict(tree, attributes):
    """
    Predicts a label following a given tree and an numpy array
    :param tree: 'dict' of nested dictionaries holding the decision tree
    :param attributes: 'numpy.ndarray' of a single row of attributes
    :return: 'float' of the predicted label
    """
    node = tree
    while not node["is_leaf"]:
        if attributes[node["attribute"]] < node["a_value"]:
            node = node["left"]
        else:
            node = node["right"]
    return node["a_value"]
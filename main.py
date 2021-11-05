import numpy as np

from helpers import get_max_depth, get_avg_depth
from visualizer import visualize_tree

np.random.seed(3)


def read_data(file_path):
    """
    Reads the CSV or space separated file into a 2D numpy array.
    :param file_path: 'str' of relative or absolute file_path
    :return: 'numpy.ndarray' of dataset in 2D array
    """
    return np.loadtxt(file_path)


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


def split_dataset(data, training_perc):
    size = len(data)
    training_size = int(size * training_perc / 100)
    training, test = data[:training_size, :], data[training_size:, :]
    return training, test


def split_dataset_10_fold(data, index):
    """
    Returns tuple of training_set, testing_set, where the testing set
    is the index-th fold in the dataset
    """
    assert 1 <= index <= 10
    size = len(data)
    fold_size = size // 10
    if index == 1:
        return data[fold_size:, ], data[:fold_size, ]
    elif index == 10:
        return data[:fold_size * 9, ], data[fold_size * 9:, ]
    else:
        test = data[(index - 1) * fold_size:index * fold_size, ]
        training = data[:(index - 1) * fold_size, ]
        training = np.append(training, data[index * fold_size:, ], axis=0)
        return training, test


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


def evaluate(test_db, trained_tree):
    correct = 0
    samples, _ = test_db.shape
    for row in test_db:
        label = row[-1]
        attributes = row[:-1]
        if predict(tree=trained_tree, attributes=attributes) == label:
            correct += 1
    return correct / samples


"""
---- CROSS VALIDATION CLASSIFICATION METRICS ---- 
"""


def generate_confusion_matrix(testing_set, tree):
    labels = room_numbers(testing_set)
    n_labels = len(labels)
    confusion_matrix = np.zeros((4, 4))
    n_cols = testing_set.shape[1]
    for i in range(len(testing_set)):
        row = testing_set[i][:n_cols - 1]
        predicted_label = predict(tree, row)
        actual_label = testing_set[i][-1]
        confusion_matrix[int(predicted_label - 1)][int(actual_label - 1)] += 1

    return confusion_matrix


def get_overall_accuracy(confusion_matrix):
    correct = 0
    total = 0

    for r in range(len(confusion_matrix)):
        for c in range(len(confusion_matrix[0])):
            if c == r:
                correct += confusion_matrix[r][c]
            total += confusion_matrix[r][c]
    return correct / total


def get_tp_tn_fp_fn_vals(confusion_matrix, positive_label):
    tp = 0  # True Positive
    tn = 0  # True Negative
    fp = 0  # False Positive
    fn = 0  # False Negative
    positive_label -= 1
    for r in range(len(confusion_matrix)):
        for c in range(len(confusion_matrix[0])):
            if c == positive_label and r == positive_label:
                tp += confusion_matrix[r][c]
            elif c == positive_label and r != positive_label:
                fp += confusion_matrix[r][c]
            elif c == positive_label and r != positive_label:
                fn += confusion_matrix[r][c]
            elif c == r:
                tn += confusion_matrix[r][c]

    return tp, tn, fp, fn


def get_accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def get_recall(tp, fp):
    return tp / (tp + fp)


def get_precision(tp, fn):
    return tp / (tp + fn)


def get_f1(precision, recall):
    return (2 * precision * recall) / (precision + recall)


def get_accuracy_precision_recall_matrix(confusion_matrix, labels):
    res_matrix = []
    print("---- EVALUATION MATRIX ---- ")
    for i in labels:
        (tp, tn, fp, fn) = get_tp_tn_fp_fn_vals(confusion_matrix, i)
        accuracy = get_accuracy(tp, tn, fp, fn)
        recall = get_recall(tp, fp)
        precision = get_precision(tp, fn)
        f1 = get_f1(precision, recall)
        print("for Label :" + str(i) +
              ", Accuracy :" + str(accuracy) +
              ", Recall :" + str(recall) +
              ", Precision :" + str(precision) +
              ", f1: " + str(f1))
        res_matrix.append([i, accuracy, recall, precision, f1])
    return res_matrix


def nested_ten_fold_validation(data):
    labels = room_numbers(data)
    number_of_labels = len(room_numbers(data))
    summed_confusion_matrix = np.zeros((4, 4))
    for index in range(1, 11):
        training, testing = split_dataset_10_fold(data, index)
        (tree, depth) = decision_tree_learning(training)

        for validation_index in range(1, 11):
            training, validation = split_dataset_10_fold(training, index)
            prune_tree(validation_set=validation, node=tree)
            confusion_matrix = generate_confusion_matrix(testing, tree)
            summed_confusion_matrix = summed_confusion_matrix + confusion_matrix

    print(summed_confusion_matrix)
    summed_confusion_matrix = summed_confusion_matrix
    overall_accuracy = get_overall_accuracy(summed_confusion_matrix)
    accuracy_precision_recall_f1_per_label = \
        get_accuracy_precision_recall_matrix(summed_confusion_matrix, labels)

    return (overall_accuracy, accuracy_precision_recall_f1_per_label)


def ten_fold_validation(data):
    labels = room_numbers(data)
    number_of_labels = len(room_numbers(data))
    summed_confusion_matrix = np.zeros((4, 4))
    for index in range(1, 11):
        training, testing = split_dataset_10_fold(data, index)
        # training, validation = split_dataset(training, 90)
        (tree, depth) = decision_tree_learning(training)
        #  prune_tree(validation_set=validation, node=tree)
        confusion_matrix = generate_confusion_matrix(testing, tree)
        summed_confusion_matrix = summed_confusion_matrix + confusion_matrix
    print(summed_confusion_matrix)
    summed_confusion_matrix = summed_confusion_matrix
    overall_accuracy = get_overall_accuracy(summed_confusion_matrix)
    accuracy_precision_recall_f1_per_label = \
        get_accuracy_precision_recall_matrix(summed_confusion_matrix, labels)

    return overall_accuracy, accuracy_precision_recall_f1_per_label


"""
-------- CROSS VALIDATION METRICS ENDS HERE -------
"""


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


x = read_data("wifi_db/clean_dataset.txt")
# x = read_data("wifi_db/noisy_dataset.txt")

np.random.shuffle(x)

new_x, testing = split_dataset(x, 90)
training, validation = split_dataset(new_x, 90)

y, depth = decision_tree_learning(training)
# visualize_tree(y, depth, "foo.png")
print(evaluate(test_db=testing, trained_tree=y))
print("avg depth = " + str(get_avg_depth(y)))
print("max depth = " + str(get_max_depth(y)))
prune_tree(validation_set=validation, node=y)
visualize_tree(y, depth, "foo1.png")
print(evaluate(test_db=testing, trained_tree=y))
accuracy, res_matrix = nested_ten_fold_validation(x)
print(accuracy)
print(res_matrix)
print("avg depth = " + str(get_avg_depth(y)))
print("max depth = " + str(get_max_depth(y)))

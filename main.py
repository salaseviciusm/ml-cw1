import numpy as np

from helpers import get_depth
from visualizer import visualize_tree

np.random.seed(0)

def read_data(file_path):
    return np.loadtxt(file_path)


def room_numbers(dataset):
    return np.unique(dataset[:, -1])


def decision_tree_learning(dataset, depth=0):
    if dataset.size == 0:
        return None, depth
    if len(np.unique(dataset[:, -1])) == 1:
        return dict({
            "a_value": np.unique(dataset[:, -1])[0],
            "leaf": True,
            "left": None,
            "right": None
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
            "leaf": False
        }
        return node, max(l_depth, r_depth)


def find_split(dataset):
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
    _, count_arr = np.unique(dataset, return_counts=True)
    n_samples = len(dataset)
    h = 0
    for freq in count_arr:
        p_k = freq / n_samples
        h -= p_k * np.log2(p_k)
    return h


def remainder(left, right):
    l_size = len(left)
    r_size = len(right)
    return (l_size * entropy(left) + r_size * entropy(right)) / (l_size + r_size)


def gain(dataset, left, right):
    return entropy(dataset) - remainder(left, right)


def split_dataset(data, training_perc, shuffle=True):
    if shuffle:
        np.random.shuffle(data)
    training, test = data[:training_perc, :], data[training_perc:, :]
    return training, test


def split_dataset_10_fold(data, index):
    # Returns tuple of training_set, testing_set, where the testing set
    # is the index-th fold in the dataset
    assert 1 <= index <= 10

    size = len(data)
    fold_size = size / 10
    if index == 1:
        return data[fold_size:, ], data[:fold_size, ]
    elif index == 10:
        return data[:fold_size * 9, ], data[fold_size * 9:, ]
    else:
        test = data[(index - 1) * fold_size:index * fold_size, ]
        training = data[:(index - 1) * fold_size, ]
        training = np.append(training, data[index * fold_size:, ], axis=0)
        return training, test


def generate_label(tree, attributes):
    # attributes = n x 1 numpy array
    node = tree
    while not node["leaf"]:
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
        if generate_label(tree=trained_tree, attributes=attributes) == label:
            correct += 1
    return correct / samples


"""
---- CROSS VALIDATION CLASSIFICATION METRICS ---- 
"""


def generate_confusion_matrix(testing_set, tree):
    labels = room_numbers(testing_set)
    confusion_matrix = np.zeros((labels, labels))
    n_cols = testing_set.shape[1]
    for i in range(len(testing_set)):
        row = testing_set[i][:n_cols - 1]
        predicted_label = generate_label(tree, row)
        actual_label = testing_set[i][-1]
        confusion_matrix[predicted_label - 1][actual_label - 1] += 1

    return confusion_matrix


def get_tp_tn_fp_fn_vals(confusion_matrix, positive_label):
    tp = 0  # True Positive
    tn = 0  # True Negative
    fp = 0  # False Positive
    fn = 0  # False Negative
    positive_label -= 1
    for row in confusion_matrix:
        for col in row:
            if col == positive_label and row == positive_label:
                tp += 1
            elif col == positive_label and row != positive_label:
                fp += 1
            elif row == positive_label and row != positive_label:
                fn += 1
            elif col == row:
                tn += 1

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
    accuracy = 0
    print("---- EVALUATION MATRIX ---- ")
    for i in labels:
        (tp, tn, fp, fn) = get_tp_tn_fp_fn_vals(confusion_matrix, i)
        accuracy = get_accuracy(tp, tn, fp, fn)
        recall = get_recall(tp, tn, fp)
        precision = get_precision(tp, tn, fn)
        f1 = get_f1(precision, recall)
        print("for Label :" + str(i) +
              ", Recall :" + str(recall) +
              ", Precision :" + str(precision) +
              ", f1: " + str(f1))
        res_matrix.append([i, accuracy, recall, precision, f1])
    # Returns (accuracy, [label, accuracy, recall, percision, f1])
    return (accuracy, res_matrix)


def ten_fold_validation(data):
    number_of_labels = room_numbers(data)
    average_confusion_matrix = np.zeros((number_of_labels, number_of_labels))
    for index in range(1, 11):
        training, testing = split_dataset_10_fold(data, index)
        (tree, depth) = decision_tree_learning(training)
        average_confusion_matrix = average_confusion_matrix + \
                                   generate_confusion_matrix(testing, tree)  #  might be broken

    average_confusion_matrix = average_confusion_matrix / 10

    return get_accuracy_precision_recall_matrix(average_confusion_matrix, number_of_labels)


"""
-------- CROSS VALIDATION METRICS ENDS HERE -------
"""


def prune_tree(validation_set, node):
    if not node["left"]["leaf"]:
        node["left"] = prune_tree(
            validation_set=validation_set[validation_set[:, node["attribute"]] < node["a_value"], :],
            node=node["left"]
        )

    if not node["right"]["leaf"]:
        node["right"] = prune_tree(
            validation_set=validation_set[validation_set[:, node["attribute"]] >= node["a_value"], :],
            node=node["right"]
        )

    if node["left"]["leaf"] and node["right"]["leaf"]:
        old_node = dict(node)
        prev = (evaluate(validation_set, old_node), old_node)
        left = (evaluate(validation_set, old_node["left"]), old_node["left"])
        right = (evaluate(validation_set, old_node["right"]), old_node["right"])
        max_node = max([prev, left, right], key=lambda t: t[0])[1]
        node = max_node
    return node


# x = read_data("wifi_db/clean_dataset.txt")
x = read_data("wifi_db/noisy_dataset.txt")
training, testing = split_dataset(x, 90)

y, depth = decision_tree_learning(training)
visualize_tree(y, depth, "foo.png")
print(evaluate(test_db=testing, trained_tree=y))
# print(get_depth(y))
prune_tree(validation_set=testing, node=y)
visualize_tree(y, depth, "foo1.png")
print(evaluate(test_db=testing, trained_tree=y))
# print(get_depth(y))
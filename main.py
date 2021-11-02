import numpy as np
import pprint

# np.random.seed(1)


def read_data(file_path):
    return np.loadtxt(file_path)


def room_numbers(dataset):
    return np.unique(dataset[:, -1])


def decision_tree_learning(dataset, depth=0):
    if dataset.size == 0:
        return None, depth
    if len(np.unique(dataset[:, -1])) == 1:
        return dict({
            "value": np.unique(dataset[:, -1])[0],
            "leaf": True,
            "left": None,
            "right": None
        }), depth
    else:
        column_num, value = find_split(dataset)
        l_branch, l_depth = decision_tree_learning(dataset[dataset[:, column_num] < value, :], depth + 1)
        r_branch, r_depth = decision_tree_learning(dataset[dataset[:, column_num] >= value, :], depth + 1)
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
    for attr_idx in range(n_cols - 1):
        attribute = dataset[:, attr_idx]
        sort_index = np.argsort(attribute)
        sort_labels, sort_attr = labels[sort_index], attribute[sort_index]
        curr_label = sort_labels[0]
        max_gain = 0
        for i in range(1, len(sort_labels)):
            if sort_attr[i] != sort_attr[i - 1]:
                curr_label = sort_labels[i]
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


x = read_data("wifi_db/clean_dataset.txt")
y = decision_tree_learning(x)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(y)

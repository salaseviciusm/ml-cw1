import math
import numpy as np

data = np.loadtxt("wifi_db/clean_dataset.txt")

num_of_labels = len(np.unique(data[:, -1]))
label_col = 7


def preprocess(dataset):
    dataset_deleted = np.delete(dataset, 0, -1)
    return dataset


def decision_tree_learning(dataset, depth):
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
        print(column_num, value)
        l_branch, l_depth = decision_tree_learning(dataset[dataset[:, column_num] < value, :], depth + 1)
        r_branch, r_depth = decision_tree_learning(dataset[dataset[:, column_num] >= value, :], depth + 1)
        node = {
            "attribute": column_num,
            "value": value,
            "left": l_branch,
            "right": r_branch,
            "leaf": False
        }
        return node, max(l_depth, r_depth)


# [[-59. -54. -58. -59. -63. -81. -80.   1.]
#  [-60. -58. -57. -59. -66. -82. -83.   1.]
#  [-60. -57. -58. -58. -50. -83. -88.   4.]]

def find_split(dataset):
    print(dataset.shape)
    best_col_num = 0
    best_attr_index = 0
    best_attr_gain = 0
    for col in range(len(dataset[0]) - 1):
        attribute = np.sort(dataset[:, (col, label_col)], 0)

        maximum_information_gain = 0
        maximum_split_index = 0
        for split_index in range(1, len(dataset)):
            if attribute[split_index - 1][0] == attribute[split_index][0]:
                # skips if split not possible i.e. attribute values are identical
                continue

            data_l = attribute[:split_index]
            data_r = attribute[split_index:]

            info_gain = gain(dataset, data_l, data_r)
            if info_gain > maximum_information_gain:
                maximum_information_gain = info_gain
                maximum_split_index = split_index

        if maximum_information_gain > best_attr_gain:
            best_attr_gain = maximum_information_gain
            best_attr_index = maximum_split_index
            best_col_num = col

    print(best_attr_gain)
    return best_col_num, (dataset[best_attr_index][best_col_num] + dataset[best_attr_index - 1][best_col_num]) / 2


def gain(dataset, leftset, rightset):
    return H(dataset) - remainder(leftset, rightset)


def H(dataset):
    assert dataset.shape[1] >= 2, dataset
    num_rows = dataset.shape[0]
    freq = {}
    for row in dataset:
        if int(row[-1]) not in freq:
            freq[int(row[-1])] = 0
        freq[int(row[-1])] += 1
    entropy = 0
    for elem in freq:
        if elem != 0:
            p_k = elem / num_rows
            entropy -= p_k * math.log2(p_k)
    return entropy


def remainder(left_set, right_set):
    left_size = left_set.shape[0]
    right_size = right_set.shape[0]
    total_size = left_size + right_size
    return (left_size * H(left_set) / total_size) + (right_size * H(right_set) / total_size)


# H_total = H(data)  #  TODO:// inside gain function replace it with h_total
# active_data = data
# res = find_split(active_data)
# entropy = res[2]
# print(res)
# while entropy:
#     active_data = active_data[active_data[:, res[0]] > res[1], :]
#     res = find_split(active_data)
#     entropy = res[2]
#     print(res)

final = decision_tree_learning(data, 0)
print(final)

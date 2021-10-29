import math
import numpy as np

data = np.loadtxt("wifi_db/clean_dataset.txt")

num_of_labels = len(np.unique(data[:, -1]))
label_col = 7


def preprocess(dataset):
    dataset_deleted = np.delete(dataset, 0, -1)
    return dataset


def decision_tree_learning(dataset, depth):
    split = find_split(dataset)


def find_split(dataset):
    best_col_num = 0
    best_attr_index = 0
    best_attr_gain = 0

    for col in range(0, len(dataset[0]) - 1):
        attribute = np.sort(dataset[:, (col, label_col)], 0)

        maximum_information_gain = 0
        maximum_split_index = 0
        for split_index in range(1, len(dataset)):

            if attribute[split_index-1][0] == attribute[split_index][0]:
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

    return best_col_num, (dataset[best_attr_index][best_col_num] + dataset[best_attr_index - 1][best_col_num]) / 2


def gain(dataset, leftset, rightset):
    return H(dataset) - remainder(leftset, rightset)


def H(dataset):
    assert dataset.shape[1] >= 2, dataset
    num_rows = dataset.shape[0]
    freq = [0 for _ in range(num_of_labels)]
    for row in dataset:
        freq[int(row[-1]) - 1] += 1
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


H_total = H(data)  #  TODO:// inside gain function replace it with h_total
print(find_split(data))

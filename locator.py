import math
import numpy as np
import sys


def read_dataset(path):
    data = []
    labels = []
    label_freqs = {}
    for line in open(path):
        if line.strip() != "":
            row = line.strip().replace("\t", " ").split(" ")
            data.append(list(map(float, row[:-1])))

            label = int(row[-1])
            labels.append([label])

            if label not in label_freqs:
                label_freqs[label] = 0
            label_freqs[label] += 1

    data = np.array(data)
    labels = np.array(labels)

    data = np.append(data, labels, axis=1)

    return data, label_freqs


data, label_freqs = read_dataset("wifi_db/clean_dataset.txt")

print(data)


def decision_tree_learning(dataset, depth):
    split = find_split(dataset)


def find_split(dataset, freqs):

    best_col_num = 0
    best_attr_index = 0
    best_attr_gain = 0

    np.set_printoptions(threshold=sys.maxsize)

    for col in range(0, len(dataset[0])-1):
        attribute = dataset[:, (col, 7)]
        attribute = attribute[np.argsort(attribute[:, 0])]
        print(attribute)

        maximum_information_gain = 0
        maximum_split_index = 0

        split_index = 0
        leftset_freqs = {elem: 0 for elem in freqs}
        rightset_freqs = dict(freqs)
        while split_index < len(dataset):
            prev_val = None
            split_val = None

            while split_index < len(attribute):
                curr_val = attribute[split_index][0]

                leftset_freqs[attribute[split_index][1]] += 1
                rightset_freqs[attribute[split_index][1]] -= 1

                split_index += 1
                if prev_val is not None and curr_val != prev_val:
                    split_val = (curr_val + prev_val) / 2
                    break
                prev_val = curr_val

            if split_val is None:
                break

            data_l = attribute[attribute[:, 0] < split_val]
            data_r = attribute[attribute[:, 0] >= split_val]

            print(leftset_freqs)
            print(rightset_freqs)
            print(split_val)
            print()

            info_gain = gain(dataset, data_l, data_r,
                             leftset_freqs, rightset_freqs)
            if info_gain > maximum_information_gain:
                maximum_information_gain = info_gain
                maximum_split_index = split_index

        if maximum_information_gain > best_attr_gain:
            best_attr_gain = maximum_information_gain
            best_attr_index = maximum_split_index
            best_col_num = col

    # TODO: do we need to get average value between best_attr_index and best_attr_index+1 ?
    print(info_gain)
    return best_col_num, dataset[best_attr_index][best_col_num]


def gain(dataset, leftset, rightset, leftset_freqs, rightset_freqs):
    freqs = {}
    for elem in leftset_freqs:
        freqs[elem] = leftset_freqs[elem]

    for elem in rightset_freqs:
        if elem not in freqs:
            freqs[elem] = rightset_freqs[elem]
        else:
            freqs[elem] += rightset_freqs[elem]

    return H(dataset, freqs) - remainder(leftset, rightset, leftset_freqs, rightset_freqs)


def H(dataset, freqs):
    assert dataset.shape[1] >= 2, dataset
    num_rows = dataset.shape[0]
    entropy = 0
    for elem in freqs:
        if freqs[elem] != 0:
            p_k = freqs[elem]/num_rows
            entropy -= p_k * math.log2(p_k)
    return entropy


def remainder(left_set, right_set, leftset_freqs, rightset_freqs):
    left_size = left_set.shape[0]
    right_size = right_set.shape[0]
    total_size = left_size + right_size
    return (left_size * H(left_set, leftset_freqs) / total_size) + (right_size * H(right_set, rightset_freqs) / total_size)


#  TODO:// inside gain function replace it with h_total
H_total = H(data, label_freqs)
print(find_split(data, label_freqs))

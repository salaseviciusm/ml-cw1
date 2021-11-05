import numpy as np

from tree import room_numbers, decision_tree_learning, prune_tree, predict


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


def get_recall(tp, fn):
    return tp / (tp + fn)


def get_precision(tp, fp):
    return tp / (tp + fp)


def get_f1(precision, recall):
    return (2 * precision * recall) / (precision + recall)


def get_accuracy_precision_recall_matrix(confusion_matrix, labels):
    res_matrix = []
    print("---- EVALUATION MATRIX ---- ")
    for i in labels:
        (tp, tn, fp, fn) = get_tp_tn_fp_fn_vals(confusion_matrix, i)
        accuracy = get_accuracy(tp, tn, fp, fn)
        recall = get_recall(tp, fn)
        precision = get_precision(tp, fp)
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
    summed_confusion_matrix = np.zeros((4, 4))
    for index in range(1, 11):
        training, testing = split_dataset_10_fold(data, index)
        # training, validation = split_dataset(training, 90)
        (tree, depth) = decision_tree_learning(training)
        confusion_matrix = generate_confusion_matrix(testing, tree)
        summed_confusion_matrix = summed_confusion_matrix + confusion_matrix
    print(summed_confusion_matrix)
    summed_confusion_matrix = summed_confusion_matrix
    overall_accuracy = get_overall_accuracy(summed_confusion_matrix)
    accuracy_precision_recall_f1_per_label = \
        get_accuracy_precision_recall_matrix(summed_confusion_matrix, labels)

    return overall_accuracy, accuracy_precision_recall_f1_per_label
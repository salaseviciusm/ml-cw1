import numpy as np

from evaluation import nested_ten_fold_validation
from helpers import get_max_depth, get_avg_depth
from tree import decision_tree_learning, evaluate, prune_tree

np.random.seed(3)


def read_data(file_path):
    """
    Reads the CSV or space separated file into a 2D numpy array.
    :param file_path: 'str' of relative or absolute file_path
    :return: 'numpy.ndarray' of dataset in 2D array
    """
    return np.loadtxt(file_path)


def split_dataset(data, training_perc):
    size = len(data)
    training_size = int(size * training_perc / 100)
    training, test = data[:training_size, :], data[training_size:, :]
    return training, test

# x = read_data("wifi_db/clean_dataset.txt")
x = read_data("wifi_db/noisy_dataset.txt")

np.random.shuffle(x)

new_x, testing = split_dataset(x, 90)
training, validation = split_dataset(new_x, 90)

y, depth = decision_tree_learning(training)
# visualize_tree(y, depth, "foo.png")
print(evaluate(test_db=testing, trained_tree=y))
print("avg depth = " + str(get_avg_depth(y)))
print("max depth = " + str(get_max_depth(y)))
prune_tree(validation_set=validation, node=y)
# visualize_tree(y, depth, "foo1.png")
print(evaluate(test_db=testing, trained_tree=y))
accuracy, res_matrix = nested_ten_fold_validation(x)
print(accuracy)
print(res_matrix)
print("avg depth = " + str(get_avg_depth(y)))
print("max depth = " + str(get_max_depth(y)))

import matplotlib.pyplot as pp

tree = {"boxstyle": "square"}
leaf = {"boxstyle": "circle"}
font = {"size": 40}
figure_size = (320, 90)
dots_per_inch = 60


def visualize_tree(tree_h, depth, filename):
    pp.figure(num=None, figsize=figure_size, dpi=dots_per_inch)
    make_tree_picture(tree_h, depth, (0, 0)).savefig(filename)
    pp.close()


def make_tree_picture(node, depth, position):
    if node is None:
        return

    x = position[0]
    y = position[1]

    if node["is_leaf"]:
        pp.text(x, y, str(node["a_value"]), horizontalalignment="center", verticalalignment="center", bbox=leaf,
                fontdict=font)
        return

    height = y - depth
    left = x - (2 ** depth)
    right = x + (2 ** depth)
    rule = "x" + str(node["attribute"]) + " > " + str(node["a_value"])

    pp.plot([left, x, right], [height, y, height])
    pp.text(x, y, rule, horizontalalignment="center",
            verticalalignment="center", bbox=tree, fontdict=font)

    make_tree_picture(node["left"], depth - 1, (left, height))
    make_tree_picture(node["right"], depth - 1, (right, height))

    return pp

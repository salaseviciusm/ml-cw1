def get_max_depth(tree):
    if tree["is_leaf"]:
        return 0
    return max(get_max_depth(tree["left"]), get_max_depth(tree["right"])) + 1


def get_total_depth(tree, depth=0):
    if tree["is_leaf"]:
        return 0
    return depth + get_total_depth(tree=tree["left"], depth=depth + 1) + get_total_depth(tree=tree["right"],
                                                                                         depth=depth + 1)


def get_total_nodes(tree):
    if tree["is_leaf"]:
        return 0
    return 1 + get_total_nodes(tree["left"]) + get_total_nodes(tree["right"])


def get_avg_depth(tree):
    return get_total_depth(tree) / get_total_nodes(tree)

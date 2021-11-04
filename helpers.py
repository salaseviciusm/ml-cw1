def get_depth(tree):
    if tree["leaf"]:
        return 1
    return max(get_depth(tree["left"]), get_depth(tree["right"])) + 1

"""Implements the Hummingbird (https://github.com/microsoft/hummingbird)."""
from typing import Any, Dict, Tuple

import numpy
from sklearn.tree import _tree

from ..common.debugging.custom_assert import assert_true


def create_hummingbird_tensor_a(tree_, features, internal_nodes) -> numpy.ndarray:
    """Create Hummingbird tensor A.

    Args:
        tree_: A sklearn tree.
        features: A list of features.
        internal_nodes: A list of internal nodes.

    Returns:
        numpy.array: A numpy array of shape (len(internal_nodes), len(features)).
    """
    a = numpy.zeros((len(features), len(internal_nodes)), dtype=numpy.int64)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i, j] = tree_.feature[internal_nodes[j]] == features[i]

    return a


def create_hummingbird_tensor_b(tree_, internal_nodes, is_integer_tree=False) -> numpy.ndarray:
    """Create Hummingbird tensor B.

    Args:
        tree_: A sklearn tree.
        internal_nodes: A list of internal nodes.
        is_integer_tree: A boolean indicating whether the tree is an integer tree.

    Returns:
        numpy.array: A numpy array of shape (len(internal_nodes), len(tree_.value)).
    """
    b = numpy.array([tree_.threshold[int_node] for int_node in internal_nodes])

    return b.astype(numpy.int64) if is_integer_tree else b


def create_subtree_nodes_set_per_node(
    all_nodes, leaf_nodes, is_left_child_of: dict, is_right_child_of: dict
) -> Tuple[Dict, Dict]:
    """Create subtrees nodes set for each node in the tree.

    Args:
        all_nodes: A list of all nodes.
        leaf_nodes: A list of leaf nodes.
        is_left_child_of: A dictionary mapping each internal node to its left child.
        is_right_child_of: A dictionary mapping each internal node to its right child.

    Returns:
        A tuple of two sets:
            - left_subtree_nodes_per_node:  A dictionary mapping each internal
                                            node to its left subtree nodes.
            - right_subtree_nodes_per_node: A dictionary mapping each internal
                                            node to its right subtree nodes.
    """
    left_subtree_nodes_per_node: Dict[Any, set] = {node: set() for node in all_nodes}
    right_subtree_nodes_per_node: Dict[Any, set] = {node: set() for node in all_nodes}

    current_nodes = {node: None for node in leaf_nodes}
    while current_nodes:
        next_nodes: dict = {}
        for node in current_nodes:
            parent_as_left_child = is_left_child_of.get(node, None)
            if parent_as_left_child is not None:
                left_subtree = left_subtree_nodes_per_node[parent_as_left_child]
                left_subtree.add(node)
                left_subtree.update(left_subtree_nodes_per_node[node])
                left_subtree.update(right_subtree_nodes_per_node[node])
                next_nodes.update({parent_as_left_child: None})

            parent_as_right_child = is_right_child_of.get(node, None)
            if parent_as_right_child is not None:
                right_subtree = right_subtree_nodes_per_node[parent_as_right_child]
                right_subtree.add(node)
                right_subtree.update(left_subtree_nodes_per_node[node])
                right_subtree.update(right_subtree_nodes_per_node[node])
                next_nodes.update({parent_as_right_child: None})

        current_nodes = next_nodes

    return left_subtree_nodes_per_node, right_subtree_nodes_per_node


def create_hummingbird_tensor_c(
    all_nodes, internal_nodes, leaf_nodes, is_left_child_of: dict, is_right_child_of: dict
) -> numpy.ndarray:
    """Create Hummingbird tensor C.

    Args:
        all_nodes: A list of all nodes.
        internal_nodes: A list of internal nodes.
        leaf_nodes: A list of leaf nodes.
        is_left_child_of: A dictionary mapping each internal node to its left child.
        is_right_child_of: A dictionary mapping each internal node to its right child.

    Returns:
        numpy.array: A numpy array of shape (len(internal_nodes), len(leaf_nodes)).
    """
    left_subtree_nodes_per_node, right_subtree_nodes_per_node = create_subtree_nodes_set_per_node(
        all_nodes, leaf_nodes, is_left_child_of, is_right_child_of
    )

    c = numpy.zeros((len(internal_nodes), len(leaf_nodes)), dtype=numpy.int64)

    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if leaf_nodes[j] in right_subtree_nodes_per_node[internal_nodes[i]]:
                c[i, j] = -1
            elif leaf_nodes[j] in left_subtree_nodes_per_node[internal_nodes[i]]:
                c[i, j] = 1

    return c


def create_hummingbird_tensor_d(leaf_nodes, is_left_child_of, is_right_child_of) -> numpy.ndarray:
    """Create Hummingbird tensor D.

    Args:
        leaf_nodes: A list of leaf nodes.
        is_left_child_of: A dictionary mapping each internal node to its left child.
        is_right_child_of: A dictionary mapping each internal node to its right child.

    Returns:
        numpy.array: A numpy array of shape (len(leaf_nodes), len(leaf_nodes)).
    """
    d = numpy.zeros((len(leaf_nodes)), dtype=numpy.int64)
    for k in range(d.shape[0]):
        current_node = leaf_nodes[k]
        num_left_children = 0
        while True:
            if (parent_as_left_child := is_left_child_of.get(current_node, None)) is not None:
                num_left_children += 1
                current_node = parent_as_left_child
            elif (parent_as_right_child := is_right_child_of.get(current_node, None)) is not None:
                current_node = parent_as_right_child
            else:
                break  # pragma: no cover (pytest does not seem to cover the break line)
        d[k] = num_left_children

    return d


def create_hummingbird_tensor_e(tree_, leaf_nodes, classes) -> numpy.ndarray:
    """Create Hummingbird tensor E.

    Args:
        tree_: A tree object.
        leaf_nodes: A list of leaf nodes.
        classes: A list of classes.

    Returns:
        A numpy array of shape (len(leaf_nodes), len(classes)).
    """
    e = numpy.zeros((len(leaf_nodes), len(classes)), dtype=numpy.int64)
    for i in range(e.shape[0]):
        leaf_node = leaf_nodes[i]
        for j in range(e.shape[1]):
            value = None
            # Make sure tree_.n_outputs == 1
            # In some cases, sklearn uses value = tree_.value[leaf_node].T[0]
            # but we don't seem to need this for now.
            assert_true(tree_.n_outputs == 1, "tree_.n_outputs != 1")
            value = tree_.value[leaf_node][0]
            class_name = numpy.argmax(value)
            e[i, j] = class_name == j

    return e


def tree_to_numpy(tree, num_features, classes):
    """Convert an sklearn tree to its Hummingbird tensor equivalent.

    Args:
        tree: The sklearn tree.
        num_features: The number of features in the dataset.
        classes: The classes in the dataset.

    Returns:
        function:   A function that takes a input and returns the tree prediction
                    with tensors operations.
    """
    tree_ = tree.tree_

    number_of_nodes = tree_.node_count
    all_nodes = list(range(number_of_nodes))
    internal_nodes = [
        node_idx
        for node_idx, feature in enumerate(tree_.feature)
        if feature != _tree.TREE_UNDEFINED  # pylint: disable=c-extension-no-member
    ]
    leaf_nodes = [
        node_idx
        for node_idx, feature in enumerate(tree_.feature)
        if feature == _tree.TREE_UNDEFINED  # pylint: disable=c-extension-no-member
    ]

    features = list(range(num_features))

    a = create_hummingbird_tensor_a(tree_, features, internal_nodes)

    b = create_hummingbird_tensor_b(tree_, internal_nodes, is_integer_tree=True)

    is_left_child_of = {
        left_child: parent
        for parent, left_child in enumerate(tree_.children_left)
        if left_child != _tree.TREE_UNDEFINED  # pylint: disable=c-extension-no-member
    }
    is_right_child_of = {
        right_child: parent
        for parent, right_child in enumerate(tree_.children_right)
        if right_child != _tree.TREE_UNDEFINED  # pylint: disable=c-extension-no-member
    }

    c = create_hummingbird_tensor_c(
        all_nodes, internal_nodes, leaf_nodes, is_left_child_of, is_right_child_of
    )

    d = create_hummingbird_tensor_d(leaf_nodes, is_left_child_of, is_right_child_of)

    e = create_hummingbird_tensor_e(tree_, leaf_nodes, classes)

    def tree_predict(inputs):
        t = inputs @ a
        t = t <= b
        t = t @ c
        t = t == d
        r = t @ e
        return r

    return tree_predict

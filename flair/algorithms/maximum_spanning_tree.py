from typing import List, Set, Tuple, Dict
import numpy

from flair.utils.checks import ConfigurationError

def decode_mst(energy: numpy.ndarray,
               length: int,
               has_labels: bool = True) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Note: Counter to typical intuition, this function decodes the _maximum_
    spanning tree.
    Decode the optimal MST tree with the Chu-Liu-Edmonds algorithm for
    maximum spanning arboresences on graphs.
    Parameters
    ----------
    energy : ``numpy.ndarray``, required.
        A tensor with shape (num_labels, timesteps, timesteps)
        containing the energy of each edge. If has_labels is ``False``,
        the tensor should have shape (timesteps, timesteps) instead.
    length : ``int``, required.
        The length of this sequence, as the energy may have come
        from a padded batch.
    has_labels : ``bool``, optional, (default = True)
        Whether the graph has labels or not.
    """
    if has_labels and energy.ndim != 3:
        raise ConfigurationError("The dimension of the energy array is not equal to 3.")
    elif not has_labels and energy.ndim != 2:
        raise ConfigurationError("The dimension of the energy array is not equal to 2.")
    input_shape = energy.shape
    max_length = input_shape[-1]

    # Our energy matrix might have been batched -
    # here we clip it to contain only non padded tokens.
    if has_labels:
        energy = energy[:, :length, :length]
        # get best label for each edge.
        label_id_matrix = energy.argmax(axis=0)
        energy = energy.max(axis=0)
    else:
        energy = energy[:length, :length]
        label_id_matrix = None
    # get original score matrix
    original_score_matrix = energy
    # initialize score matrix to original score matrix
    score_matrix = numpy.array(original_score_matrix, copy=True)

    old_input = numpy.zeros([length, length], dtype=numpy.int32)
    old_output = numpy.zeros([length, length], dtype=numpy.int32)
    current_nodes = [True for _ in range(length)]
    representatives: List[Set[int]] = []

    for node1 in range(length):
        original_score_matrix[node1, node1] = 0.0
        score_matrix[node1, node1] = 0.0
        representatives.append({node1})

        for node2 in range(node1 + 1, length):
            old_input[node1, node2] = node1
            old_output[node1, node2] = node2

            old_input[node2, node1] = node2
            old_output[node2, node1] = node1

    final_edges: Dict[int, int] = {}

    # The main algorithm operates inplace.
    chu_liu_edmonds(length, score_matrix, current_nodes,
                    final_edges, old_input, old_output, representatives)

    heads = numpy.zeros([max_length], numpy.int32)
    if has_labels:
        head_type = numpy.ones([max_length], numpy.int32)
    else:
        head_type = None

    for child, parent in final_edges.items():
        heads[child] = parent
        if has_labels:
            head_type[child] = label_id_matrix[parent, child]

    return heads, head_type


def chu_liu_edmonds(length: int,
                    score_matrix: numpy.ndarray,
                    current_nodes: List[bool],
                    final_edges: Dict[int, int],
                    old_input: numpy.ndarray,
                    old_output: numpy.ndarray,
                    representatives: List[Set[int]]):
    """
    Applies the chu-liu-edmonds algorithm recursively
    to a graph with edge weights defined by score_matrix.
    Note that this function operates in place, so variables
    will be modified.
    Parameters
    ----------
    length : ``int``, required.
        The number of nodes.
    score_matrix : ``numpy.ndarray``, required.
        The score matrix representing the scores for pairs
        of nodes.
    current_nodes : ``List[bool]``, required.
        The nodes which are representatives in the graph.
        A representative at it's most basic represents a node,
        but as the algorithm progresses, individual nodes will
        represent collapsed cycles in the graph.
    final_edges: ``Dict[int, int]``, required.
        An empty dictionary which will be populated with the
        nodes which are connected in the maximum spanning tree.
    old_input: ``numpy.ndarray``, required.
        a map from an edge to its head node.
        Key: The edge is a tuple, and elements in a tuple
        could be a node or a representative of a cycle.
    old_output: ``numpy.ndarray``, required.
    representatives : ``List[Set[int]]``, required.
        A list containing the nodes that a particular node
        is representing at this iteration in the graph.
    Returns
    -------
    Nothing - all variables are modified in place.
    """
    # Set the initial graph to be the greedy best one.
    # Node '0' is always the root node.
    parents = [-1]
    for node1 in range(1, length):
        # Init the parent of each node to be the root node.
        parents.append(0)
        if current_nodes[node1]:
            # If the node is a representative,
            # find the max outgoing edge to other non-root representative,
            # and update its parent.
            max_score = score_matrix[0, node1]
            for node2 in range(1, length):
                if node2 == node1 or not current_nodes[node2]:
                    continue

                new_score = score_matrix[node2, node1]
                if new_score > max_score:
                    max_score = new_score
                    parents[node1] = node2

    # Check if this solution has a cycle.
    has_cycle, cycle = _find_cycle(parents, length, current_nodes)
    # If there are no cycles, find all edges and return.
    if not has_cycle:
        final_edges[0] = -1
        for node in range(1, length):
            if not current_nodes[node]:
                continue

            parent = old_input[parents[node], node]
            child = old_output[parents[node], node]
            final_edges[child] = parent
        return

    # Otherwise, we have a cycle so we need to remove an edge.
    # From here until the recursive call is the contraction stage of the algorithm.
    cycle_weight = 0.0
    # Find the weight of the cycle.
    index = 0
    for node in cycle:
        index += 1
        cycle_weight += score_matrix[parents[node], node]

    # For each node in the graph, find the maximum weight incoming
    # and outgoing edge into the cycle.
    cycle_representative = cycle[0]
    for node in range(length):
        # Nodes not in the cycle.
        if not current_nodes[node] or node in cycle:
            continue

        in_edge_weight = float("-inf")
        in_edge = -1
        out_edge_weight = float("-inf")
        out_edge = -1

        for node_in_cycle in cycle:
            if score_matrix[node_in_cycle, node] > in_edge_weight:
                in_edge_weight = score_matrix[node_in_cycle, node]
                in_edge = node_in_cycle

            # Add the new edge score to the cycle weight
            # and subtract the edge we're considering removing.
            score = (cycle_weight +
                     score_matrix[node, node_in_cycle] -
                     score_matrix[parents[node_in_cycle], node_in_cycle])

            if score > out_edge_weight:
                out_edge_weight = score
                out_edge = node_in_cycle

        score_matrix[cycle_representative, node] = in_edge_weight
        old_input[cycle_representative, node] = old_input[in_edge, node]
        old_output[cycle_representative, node] = old_output[in_edge, node]

        score_matrix[node, cycle_representative] = out_edge_weight
        old_output[node, cycle_representative] = old_output[node, out_edge]
        old_input[node, cycle_representative] = old_input[node, out_edge]

    # For the next recursive iteration, we want to consider the cycle as a
    # single node. Here we collapse the cycle into the first node in the
    # cycle (first node is arbitrary), set all the other nodes not be
    # considered in the next iteration. We also keep track of which
    # representatives we are considering this iteration because we need
    # them below to check if we're done.
    considered_representatives: List[Set[int]] = []
    for i, node_in_cycle in enumerate(cycle):
        considered_representatives.append(set())
        if i > 0:
            # We need to consider at least one
            # node in the cycle, arbitrarily choose
            # the first.
            current_nodes[node_in_cycle] = False

        for node in representatives[node_in_cycle]:
            considered_representatives[i].add(node)
            if i > 0:
                representatives[cycle_representative].add(node)

    chu_liu_edmonds(length, score_matrix, current_nodes, final_edges, old_input, old_output, representatives)

    # Expansion stage.
    # check each node in cycle, if one of its representatives
    # is a key in the final_edges, it is the one we need.
    # The node we are looking for is the node which is the child
    # of the incoming edge to the cycle.
    found = False
    key_node = -1
    for i, node in enumerate(cycle):
        for cycle_rep in considered_representatives[i]:
            if cycle_rep in final_edges:
                key_node = node
                found = True
                break
        if found:
            break

    # break the cycle.
    previous = parents[key_node]
    while previous != key_node:
        child = old_output[parents[previous], previous]
        parent = old_input[parents[previous], previous]
        final_edges[child] = parent
        previous = parents[previous]


def _find_cycle(parents: List[int],
                length: int,
                current_nodes: List[bool]) -> Tuple[bool, List[int]]:
    """
    :return:
        has_cycle: whether the graph has at least a cycle.
        cycle: a list of nodes which form a cycle in the graph.
    """

    # 'added' means that the node has been visited.
    added = [False for _ in range(length)]
    added[0] = True
    cycle = set()
    has_cycle = False
    for i in range(1, length):
        if has_cycle:
            break
        # don't redo nodes we've already
        # visited or aren't considering.
        if added[i] or not current_nodes[i]:
            continue
        # Initialize a new possible cycle.
        this_cycle = set()
        this_cycle.add(i)
        added[i] = True
        has_cycle = True
        next_node = i
        while parents[next_node] not in this_cycle:
            next_node = parents[next_node]
            # If we see a node we've already processed,
            # we can stop, because the node we are
            # processing would have been in that cycle.
            # Note that in the first pass of the for loop,
            # every node except that the root has been assigned
            # a head, if there's no cycle, the while loop
            # will finally arrive at the root
            if added[next_node]:
                has_cycle = False
                break
            added[next_node] = True
            this_cycle.add(next_node)

        if has_cycle:
            original = next_node
            cycle.add(original)
            next_node = parents[original]
            while next_node != original:
                cycle.add(next_node)
                next_node = parents[next_node]
            break

    return has_cycle, list(cycle)


def decode_mst_with_coreference(
        energy: numpy.ndarray,
        coreference: List[int],
        length: int,
        has_labels: bool = True) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Note: Counter to typical intuition, this function decodes the _maximum_
    spanning tree.
    Decode the optimal MST tree with the Chu-Liu-Edmonds algorithm for
    maximum spanning arboresences on graphs.
    Parameters
    ----------
    energy : ``numpy.ndarray``, required.
        A tensor with shape (num_labels, timesteps, timesteps)
        containing the energy of each edge. If has_labels is ``False``,
        the tensor should have shape (timesteps, timesteps) instead.
    coreference: ``List[int]``, required.
        A list which maps a node to its first precedent.
    length : ``int``, required.
        The length of this sequence, as the energy may have come
        from a padded batch.
    has_labels : ``bool``, optional, (default = True)
        Whether the graph has labels or not.
    """
    if has_labels and energy.ndim != 3:
        raise ConfigurationError("The dimension of the energy array is not equal to 3.")
    elif not has_labels and energy.ndim != 2:
        raise ConfigurationError("The dimension of the energy array is not equal to 2.")
    input_shape = energy.shape
    max_length = input_shape[-1]

    # Our energy matrix might have been batched -
    # here we clip it to contain only non padded tokens.
    if has_labels:
        energy = energy[:, :length, :length]
        # get best label for each edge.
        label_id_matrix = energy.argmax(axis=0)
        energy = energy.max(axis=0)
    else:
        energy = energy[:length, :length]
        label_id_matrix = None
    # get original score matrix
    original_score_matrix = energy
    # initialize score matrix to original score matrix
    score_matrix = numpy.array(original_score_matrix, copy=True)

    old_input = numpy.zeros([length, length], dtype=numpy.int32)
    old_output = numpy.zeros([length, length], dtype=numpy.int32)
    current_nodes = [True for _ in range(length)]
    representatives: List[Set[int]] = []

    for node1 in range(length):
        original_score_matrix[node1, node1] = 0.0
        score_matrix[node1, node1] = 0.0
        representatives.append({node1})

        for node2 in range(node1 + 1, length):
            old_input[node1, node2] = node1
            old_output[node1, node2] = node2

            old_input[node2, node1] = node2
            old_output[node2, node1] = node1

    final_edges: Dict[int, int] = {}

    # The main algorithm operates inplace.
    adapted_chu_liu_edmonds(
        length, score_matrix, coreference, current_nodes,
        final_edges, old_input, old_output, representatives)

    # Modify edges which are invalid according to coreference.
    _validate(final_edges, length, original_score_matrix, coreference)

    heads = numpy.zeros([max_length], numpy.int32)
    if has_labels:
        head_type = numpy.ones([max_length], numpy.int32)
    else:
        head_type = None

    for child, parent in final_edges.items():
        heads[child] = parent
        if has_labels:
            head_type[child] = label_id_matrix[parent, child]

    return heads, head_type


def adapted_chu_liu_edmonds(length: int,
                            score_matrix: numpy.ndarray,
                            coreference: List[int],
                            current_nodes: List[bool],
                            final_edges: Dict[int, int],
                            old_input: numpy.ndarray,
                            old_output: numpy.ndarray,
                            representatives: List[Set[int]]):
    """
    Applies the chu-liu-edmonds algorithm recursively
    to a graph with edge weights defined by score_matrix.
    Note that this function operates in place, so variables
    will be modified.
    Parameters
    ----------
    length : ``int``, required.
        The number of nodes.
    score_matrix : ``numpy.ndarray``, required.
        The score matrix representing the scores for pairs
        of nodes.
    coreference: ``List[int]``, required.
        A list which maps a node to its first precedent.
    current_nodes : ``List[bool]``, required.
        The nodes which are representatives in the graph.
        A representative at it's most basic represents a node,
        but as the algorithm progresses, individual nodes will
        represent collapsed cycles in the graph.
    final_edges: ``Dict[int, int]``, required.
        An empty dictionary which will be populated with the
        nodes which are connected in the maximum spanning tree.
    old_input: ``numpy.ndarray``, required.
        a map from an edge to its head node.
        Key: The edge is a tuple, and elements in a tuple
        could be a node or a representative of a cycle.
    old_output: ``numpy.ndarray``, required.
    representatives : ``List[Set[int]]``, required.
        A list containing the nodes that a particular node
        is representing at this iteration in the graph.
    Returns
    -------
    Nothing - all variables are modified in place.
    """
    # Set the initial graph to be the greedy best one.
    # Node '0' is always the root node.
    parents = [-1]
    for node1 in range(1, length):
        # Init the parent of each node to be the root node.
        parents.append(0)
        if current_nodes[node1]:
            # If the node is a representative,
            # find the max outgoing edge to other non-root representative,
            # and update its parent.
            max_score = score_matrix[0, node1]
            for node2 in range(1, length):
                if node2 == node1 or not current_nodes[node2]:
                    continue

                # Exclude edges formed by two coreferred nodes
                _parent = old_input[node1, node2]
                _child = old_output[node1, node2]
                if coreference[_parent] == coreference[_child]:
                    continue

                new_score = score_matrix[node2, node1]
                if new_score > max_score:
                    max_score = new_score
                    parents[node1] = node2

    # Check if this solution has a cycle.
    has_cycle, cycle = _find_cycle(parents, length, current_nodes)
    # If there are no cycles, find all edges and return.
    if not has_cycle:
        final_edges[0] = -1
        for node in range(1, length):
            if not current_nodes[node]:
                continue

            parent = old_input[parents[node], node]
            child = old_output[parents[node], node]
            final_edges[child] = parent
        return

    # Otherwise, we have a cycle so we need to remove an edge.
    # From here until the recursive call is the contraction stage of the algorithm.
    cycle_weight = 0.0
    # Find the weight of the cycle.
    index = 0
    for node in cycle:
        index += 1
        cycle_weight += score_matrix[parents[node], node]

    # For each node in the graph, find the maximum weight incoming
    # and outgoing edge into the cycle.
    cycle_representative = cycle[0]
    for node in range(length):
        # Nodes not in the cycle.
        if not current_nodes[node] or node in cycle:
            continue

        in_edge_weight = float("-inf")
        in_edge = -1
        out_edge_weight = float("-inf")
        out_edge = -1

        for node_in_cycle in cycle:
            # Exclude edges formed by two coreferred nodes.
            _parent = old_input[node_in_cycle, node]
            _child = old_output[node_in_cycle, node]
            if coreference[_parent] != coreference[_child]:
                if score_matrix[node_in_cycle, node] > in_edge_weight:
                    in_edge_weight = score_matrix[node_in_cycle, node]
                    in_edge = node_in_cycle

            # Exclude edges formed by two coreferred nodes.
            _parent = old_input[node, node_in_cycle]
            _child = old_output[node, node_in_cycle]
            if coreference[_parent] != coreference[_child]:
                # Add the new edge score to the cycle weight
                # and subtract the edge we're considering removing.
                score = (cycle_weight +
                        score_matrix[node, node_in_cycle] -
                        score_matrix[parents[node_in_cycle], node_in_cycle])

                if score > out_edge_weight:
                    out_edge_weight = score
                    out_edge = node_in_cycle

        score_matrix[cycle_representative, node] = in_edge_weight
        old_input[cycle_representative, node] = old_input[in_edge, node]
        old_output[cycle_representative, node] = old_output[in_edge, node]

        score_matrix[node, cycle_representative] = out_edge_weight
        old_output[node, cycle_representative] = old_output[node, out_edge]
        old_input[node, cycle_representative] = old_input[node, out_edge]

    # For the next recursive iteration, we want to consider the cycle as a
    # single node. Here we collapse the cycle into the first node in the
    # cycle (first node is arbitrary), set all the other nodes not be
    # considered in the next iteration. We also keep track of which
    # representatives we are considering this iteration because we need
    # them below to check if we're done.
    considered_representatives: List[Set[int]] = []
    for i, node_in_cycle in enumerate(cycle):
        considered_representatives.append(set())
        if i > 0:
            # We need to consider at least one
            # node in the cycle, arbitrarily choose
            # the first.
            current_nodes[node_in_cycle] = False

        for node in representatives[node_in_cycle]:
            considered_representatives[i].add(node)
            if i > 0:
                representatives[cycle_representative].add(node)

    adapted_chu_liu_edmonds(length, score_matrix, coreference, current_nodes, final_edges, old_input, old_output, representatives)

    # Expansion stage.
    # check each node in cycle, if one of its representatives
    # is a key in the final_edges, it is the one we need.
    # The node we are looking for is the node which is the child
    # of the incoming edge to the cycle.
    found = False
    key_node = -1
    for i, node in enumerate(cycle):
        for cycle_rep in considered_representatives[i]:
            if cycle_rep in final_edges:
                key_node = node
                found = True
                break
        if found:
            break

    # break the cycle.
    previous = parents[key_node]
    while previous != key_node:
        child = old_output[parents[previous], previous]
        parent = old_input[parents[previous], previous]
        final_edges[child] = parent
        previous = parents[previous]


def _validate(final_edges, length, original_score_matrix, coreference):
    # Count how many edges have been modified by this function.
    modified = 0

    # Make a constant used by _find_cycle.
    current_nodes = [True for _ in range(length)]

    # Group nodes by coreference.
    group_by_precedent = {}
    for node, precedent in enumerate(coreference):
        if precedent not in group_by_precedent:
            group_by_precedent[precedent] = []
        group_by_precedent[precedent].append(node)

    # Validate parents of nodes in each group.
    for group in group_by_precedent.values():
        # Skip if only one node in the group.
        if len(group) == 1:
            continue
        # Group conflicting nodes by parent.
        conflicts_by_parent = {}
        for child in group:
            parent = final_edges[child]
            if parent not in conflicts_by_parent:
                conflicts_by_parent[parent] = []
            conflicts_by_parent[parent].append(child)

        # Keep the parents which have already been taken.
        reserved_parents = set(conflicts_by_parent.keys())
        for parent, conflicts in conflicts_by_parent.items():
            # Skip if no conflict.
            if len(conflicts) == 1:
                continue
            # Find the node that has the maximum edge with the parent.
            winner = max(conflicts, key=lambda _child: original_score_matrix[parent, _child])
            # Modify other nodes' parents.
            for child in conflicts:
                # Skip the winner.
                if child == winner:
                    continue
                # Sort its candidate parents by score.
                parent_scores = original_score_matrix[:, child]
                for _parent in numpy.argsort(parent_scores)[::-1]:
                    # Skip its current parent and the reserved parents.
                    if _parent == parent or _parent in reserved_parents:
                        continue
                    # Check if there's any cycle if we use this parent.
                    parents = final_edges.copy()
                    parents[child] = _parent
                    has_cycle, _ = _find_cycle(parents, length, current_nodes)
                    if has_cycle:
                        continue
                    # Add it to the reserved parents.
                    reserved_parents.add(_parent)
                    # Update its parent.
                    final_edges[child] = _parent
                    # Update the counter.
                    modified += 1
                    break
                # else:
                #     print('* Could not find another parent. Use the old one.')
    # if modified > 0:
    #     print('* Validate')
    return modified

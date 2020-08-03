import json
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# start_index is the first unused index for new nodes
def convert_structure(structure: Union[list, dict, int, str, bool],
                      start_index: int = 1, our_index: int = 0) -> \
        Tuple[int, List[int], List[int],
              Dict[int, Dict[str, Union[str, int, bool]]]]:
    if isinstance(structure, list):
        src: List[int] = []
        dest: List[int] = []
        labels: Dict[int, Dict[str, Union[str, int, bool]]] = \
            defaultdict(lambda: {})
        for index, child in enumerate(structure):
            src.append(our_index)
            dest.append(start_index)
            labels[start_index]["index"] = index
            start_index, node_src, node_dest, node_labels = \
                convert_structure(child, start_index + 1, start_index)
            src += node_src
            dest += node_dest
            for node_index, n_labels in node_labels.items():
                labels[node_index].update(n_labels)
        return start_index, src, dest, labels
    elif isinstance(structure, dict):
        src = []
        dest = []
        labels = defaultdict(lambda: {})
        for key, child in structure.items():
            src.append(our_index)
            dest.append(start_index)
            labels[start_index]["key"] = key
            start_index, node_src, node_dest, node_labels = \
                convert_structure(child, start_index + 1, start_index)
            src += node_src
            dest += node_dest
            for node_index, n_labels in node_labels.items():
                labels[node_index].update(n_labels)
        return start_index, src, dest, labels
    else:
        return start_index, [], [], {our_index: {"value": structure}}


def convert_json(file_name: str):
    with open(file_name, 'r') as in_file:
        data = json.load(in_file)
    _, src, dest, labels = convert_structure(data)
    src = np.array(src)
    dest = np.array(dest)
    graph = dgl.DGLGraph()
    node_count = max(*src, *dest) + 1
    graph.add_nodes(node_count)
    graph.add_edges(src, dest)
    nx_graph = graph.to_networkx()
    pos = nx.planar_layout(nx_graph)
    nx.draw(nx_graph, pos, with_labels=True, node_color=[[.7, .7, .7]])
    plt.show()
    plt.savefig("test.png")


if __name__ == "__main__":
    convert_json(sys.argv[1])

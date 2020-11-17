import json
from argparse import ArgumentParser


class Node:
    def __init__(self, id):
        self.id = id
        self.parents = []
        self.children = []


def analyze_cag_edges(iden_by_ids, nodes_by_id):
    edges_from = {}
    for id in iden_by_ids:
        node = nodes_by_id[id]
        iden = iden_by_ids[id]
        if iden not in edges_from:
            edges_from[iden] = {}

        qu = []
        qu.append((node, 0))
        while qu:
            n, level = qu.pop(0)
            for p in n.parents:
                if p.id in iden_by_ids:
                    from_iden, from_level = iden_by_ids[p.id], level
                    if (from_iden not in edges_from[iden]) or \
                            (from_iden in edges_from[iden] and from_level < edges_from[iden][from_iden]):
                        edges_from[iden][from_iden] = from_level
                qu.append((p, level + 1))
    return edges_from


def analyze_cag(input):
    constants = set()
    variables = set()
    iden_by_ids = {}
    nodes_by_id = {}
    defs_by_idens = {}

    with open(input, "rt") as f:
        model = json.load(f)

        parameters = []
        inputs = []
        model_variables = []
        for metadata in model["metadata"]:
            if metadata["type"] == "model-identifiers":
                parameters.extend(metadata["attributes"][0]["parameters"])
                inputs.extend(metadata["attributes"][0]["inputs"])
                model_variables.extend(metadata["attributes"][0]["model_variables"])

        for variable in model["variables"]:
            if variable["uid"] in model_variables:
                if "metadata" in variable:
                    defn = variable["metadata"][0]["attributes"][0]["text_definition"]
                    iden = variable["metadata"][0]["attributes"][0]["text_identifier"]
                    variables.add(iden)
                    iden_by_ids[variable["uid"]] = iden
                    defs_by_idens[iden] = defn

            nodes_by_id[variable["uid"]] = Node(variable["uid"])

        for variable in model["variables"]:
            if variable["uid"] in inputs:
                if "metadata" in variable:
                    defn = variable["metadata"][0]["attributes"][0]["text_definition"]
                    iden = variable["metadata"][0]["attributes"][0]["text_identifier"]
                    if iden not in variables:
                        constants.add(iden)
                        defs_by_idens[iden] = defn
                    iden_by_ids[variable["uid"]] = iden

            if variable["uid"] in parameters:
                if "metadata" in variable:
                    defn = variable["metadata"][0]["attributes"][0]["text_definition"]
                    iden = variable["metadata"][0]["attributes"][0]["text_identifier"]
                    if iden not in variables:
                        constants.add(iden)
                        defs_by_idens[iden] = defn
                    iden_by_ids[variable["uid"]] = iden

        for edge in model["edges"]:
            from_id, to_id = edge
            from_node = nodes_by_id[from_id]
            to_node = nodes_by_id[to_id]
            from_node.children.append(to_node)
            to_node.parents.append(from_node)

    edges_from = analyze_cag_edges(iden_by_ids, nodes_by_id)

    return variables, constants, edges_from, defs_by_idens


def create_nodes(variables, constants, defs_by_idens):
    nodes = []
    node_id_by_iden = {}
    node_id = 0

    for variable in variables:
        node_id += 1
        nodes.append({
            "id": str(node_id),
            "concept": variable,
            "label": variable,
            "type": "variable",
            "metadata": {
                "description": defs_by_idens[variable]
            }
        })
        node_id_by_iden[variable] = node_id

    for constant in constants:
        node_id += 1
        nodes.append({
            "id": str(node_id),
            "concept": constant,
            "label": constant,
            "type": "constant",
            "metadata": {
                "description": defs_by_idens[constant]
            }
        })
        node_id_by_iden[constant] = node_id

    return nodes, node_id_by_iden


def create_edges(edges_from, node_id_by_iden):
    edges = []
    max_weight = 0

    for edge_from in edges_from.values():
        for weight in edge_from.values():
            if weight > max_weight:
                max_weight = weight

    edge_id = 0

    for node in edges_from:
        target = node_id_by_iden[node]
        for node_from, weight in edges_from[node].items():
            source = node_id_by_iden[node_from]
            edge_id += 1
            edges.append({
                "id": str(edge_id),
                "source": str(source),
                "target": str(target),
                "metadata": {
                    "weight": (max_weight - weight) / max_weight
                }
            })

    return edges


def create_vs(variables, constants, edges_from, defs_by_idens):
    nodes, node_id_by_iden = create_nodes(variables, constants, defs_by_idens)
    edges = create_edges(edges_from, node_id_by_iden)
    return nodes, edges


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', required=True,
                        help='Input CAG .json file')
    parser.add_argument('--output', required=True,
                        help='The location of the resulting output file')
    args = parser.parse_args()

    variables, constants, edges_from, defs_by_idens = analyze_cag(args.input)

    nodes, edges = create_vs(variables, constants, edges_from, defs_by_idens)

    with open(args.output, 'w') as f:
        json.dump({
            "nodes": nodes,
            "edges": edges
        }, f)

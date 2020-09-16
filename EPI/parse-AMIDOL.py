import json
import re
from argparse import ArgumentParser


def get_nodes_to_class_name(vdsol):
    nodes_to_class_name = {}
    with open(vdsol) as f:
        data = json.load(f)
        nodes = data["journal"][0]["nodes"]  # assume there is only a single "journal"
        for node in nodes:
            nodes_to_class_name[node["label"]] = node["props"]["className"]
    return nodes_to_class_name


def get_keys_to_vals(data):
    keys_to_vals = {}
    for key in data:
        keys_to_vals[key] = data[key]
    return keys_to_vals


def get_initial_value_names(states):
    names = set()
    for state in states:
        names.add(state["initial_value"])
    return names


def get_nodes(states_to_vals, constants_to_vals, expressions_to_vals):
    nodes_by_id = {}
    node_ids_by_node_label = {}
    node_id = 0
    initial_value_names = get_initial_value_names(states_to_vals.values())

    for state_label, state in states_to_vals.items():
        node_id += 1
        node = {
            "id": str(node_id),
            "concept": state_label,
            "label": state_label,
            "type": "variable",
            "metadata": {
                "initial_value": constants_to_vals[state["initial_value"]],
                "description": state["metadata"]["description"]
            }
        }
        nodes_by_id[node_id] = node
        node_ids_by_node_label[state_label] = node_id

    for constant_label, constant_val in constants_to_vals.items():
        if constant_label in initial_value_names:
            continue  # we don't want to make the state initial value constants explicit nodes in the graph
        node_id += 1
        node = {
            "id": str(node_id),
            "concept": constant_label,
            "label": constant_label,
            "type": "constant",
            "metadata": {
                "value": str(constant_val)
            }
        }
        nodes_by_id[node_id] = node
        node_ids_by_node_label[constant_label] = node_id

    for expression_label, expression_val in expressions_to_vals.items():
        node_id += 1
        node = {
            "id": str(node_id),
            "concept": expression_label,
            "label": expression_label,
            "type": "variable",
            "metadata": {
                "expression": expression_val
            }
        }
        nodes_by_id[node_id] = node
        node_ids_by_node_label[expression_label] = node_id

    return nodes_by_id, node_ids_by_node_label


def parse_vars(expression):
    return set(re.findall("[A-Za-z_]+", expression))


def get_edges_for_expressions(edge_id, expressions_to_vals, nodes_by_id, node_ids_by_node_label):
    edges = []
    for expression_label, expression in expressions_to_vals.items():
        node_id = node_ids_by_node_label[expression_label]
        vars = parse_vars(expression)
        if len(vars) == 0:
            # this expression is really a constant; update the node type accordingly
            nodes_by_id[node_id]["type"] = "constant"  # TODO side-effect programming!
        else:
            for var in vars:
                if var not in node_ids_by_node_label:
                    #  sometimes var can be "AIR_time" or "and"
                    continue
                edge_id += 1
                edges.append({
                    "id": str(edge_id),
                    "source": str(node_ids_by_node_label[var]),
                    "target": str(node_id)
                })
    return edges, edge_id


def get_edges_from_events(edge_id, events_to_vals, nodes_by_id, node_ids_by_node_label):
    edges = []
    established_edges = set()
    for event in events_to_vals.values():
        rate = event["rate"]
        vars = parse_vars(rate)
        for transition in event["output_predicate"]["transition_function"]:
            node_id = node_ids_by_node_label[transition]

            # update the expression of the state nodes  TODO side-effect programming!
            sign = event["output_predicate"]["transition_function"][transition]
            node = nodes_by_id[node_id]
            expr = sign + " * (" + rate + ")" if sign == "-1.0" else rate
            if "expression" in node["metadata"]:
                node["metadata"]["expression"] = "(" + node["metadata"]["expression"] + ") + " + expr
            else:
                node["metadata"]["expression"] = expr

            # all the nodes given in vars point to nodes given by transitions (unless that edge was always established)
            for var in vars:
                if var not in node_ids_by_node_label:
                    #  sometimes var can be "AIR_time" or "and"
                    continue
                source_id = node_ids_by_node_label[var]
                target_id = node_id
                if (source_id, target_id) in established_edges:
                    continue
                edge_id += 1
                established_edges.add((source_id, target_id))
                edges.append({
                    "id": str(edge_id),
                    "source": str(source_id),
                    "target": str(target_id)
                })

    return edges, edge_id


def get_groups(nodes_to_class_name, node_ids_by_node_label):
    groups = []
    class_names_to_node_ids = {}

    for node_label, class_name in nodes_to_class_name.items():
        if node_label not in node_ids_by_node_label:
            continue
        if class_name not in class_names_to_node_ids:
            class_names_to_node_ids[class_name] = []
        class_names_to_node_ids[class_name].append(str(node_ids_by_node_label[node_label]))

    for class_name, node_ids in class_names_to_node_ids.items():
        groups.append({
            "id": class_name,
            "members": node_ids
        })

    return groups


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ir', required=True,
                        help='Input AMIDOL IR .json file')
    parser.add_argument('--vdsol', required=True,
                        help='Input AMIDOL VDSOL .json file')
    parser.add_argument('--output', required=True,
                        help='The location of the resulting output file')
    args = parser.parse_args()

    nodes_to_class_name = get_nodes_to_class_name(args.vdsol)

    nodes = []
    edges = []
    groups = []

    with open(args.ir) as f:
        data = json.load(f)

        constants_to_vals = get_keys_to_vals(data["constants"])
        expressions_to_vals = get_keys_to_vals(data["expressions"])
        states_to_vals = get_keys_to_vals(data["states"])
        events_to_vals = get_keys_to_vals(data["events"])

        nodes_by_id, node_ids_by_node_label = get_nodes(states_to_vals, constants_to_vals, expressions_to_vals)

        edge_id = 0

        expression_edges, edge_id = get_edges_for_expressions(edge_id, expressions_to_vals,
                                                              nodes_by_id, node_ids_by_node_label)
        edges.extend(expression_edges)

        event_edges, edge_id = get_edges_from_events(edge_id, events_to_vals, nodes_by_id, node_ids_by_node_label)
        edges.extend(event_edges)

        nodes.extend(nodes_by_id.values())

        groups.extend(get_groups(nodes_to_class_name, node_ids_by_node_label))

    with open(args.output, 'w') as f:
        json.dump({
            "nodes": nodes,
            "edges": edges,
            "groups": groups
        }, f)


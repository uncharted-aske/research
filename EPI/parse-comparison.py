import argparse
import json


def add_node(common, name, all_nodes, grfn_id_by_name, linked_to_nodes):
    node_id = common[name]
    for n in common:
        if n is not name:
            if node_id not in linked_to_nodes:
                linked_to_nodes[node_id] = []
            linked_to_nodes[node_id].append({"grfn-id":  grfn_id_by_name[n], "node-id": common[n]})
    # TODO validate that the node is in the GrFN
    all_nodes.add(node_id)


def add_edge(common, name, all_edges, grfn_id_by_name, linked_to_edges):
    src_id = common[name]["src"]
    dst_id = common[name]["dst"]
    for n in common:
        if n is not name:
            if (src_id, dst_id) not in linked_to_edges:
                linked_to_edges[(src_id, dst_id)] = []
            linked_to_edges[(src_id, dst_id)].append({
                "grfn-id": grfn_id_by_name[n],
                "source": common[n]["src"],
                "target": common[n]["dst"]
            })
    # TODO validate that the edge is in the GrFN
    all_edges.add((src_id, dst_id))


def populate_comparison_hmi(comparison_hmi, comparison_az, name, grfn, grfn_id_by_name):
    subgraph = {"id": grfn["uid"], "name": grfn_name, "nodes": [], "edges": []}

    all_nodes = set()
    linked_to_nodes = {}
    [add_node(common, name, all_nodes, grfn_id_by_name, linked_to_nodes) for common in comparison_az["common_model_input_nodes"]]
    [add_node(common, name, all_nodes, grfn_id_by_name, linked_to_nodes) for common in comparison_az["common_model_output_nodes"]]
    [add_node(common, name, all_nodes, grfn_id_by_name, linked_to_nodes) for common in comparison_az["common_model_variable_nodes"]]
    for node_id in all_nodes:
        subgraph["nodes"].append({
            "id": node_id,
            "linked-to": linked_to_nodes[node_id]
        })

    all_edges = set()
    linked_to_edges = {}
    [add_edge(common, name, all_edges, grfn_id_by_name, linked_to_edges) for common in comparison_az["common_edges"]]
    for src_id, dst_id in all_edges:
        subgraph["edges"].append({
            "source": src_id,
            "target": dst_id,
            "linked-to": linked_to_edges[(src_id, dst_id)]
        })
    
    comparison_hmi["subgraphs"].append(subgraph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', required=True,
                        help='Input comparison .json file')
    parser.add_argument('--grfn', required=True, action='append', nargs="?",
                        help='Input GrFN .json file')
    parser.add_argument('--output', required=True,
                        help='The location of the resulting output .json file')
    args = parser.parse_args()

    with open(args.comparison, "r") as f:
        comparison_az = json.load(f)

    grfn_by_id = {}
    for grfn_file in args.grfn:
        with open(grfn_file, "r") as f:
            grfn = json.load(f)
            grfn_by_id[grfn["uid"]] = grfn

    # verify that all the GrFNs specified in the Arizona comparison have been provided
    grfn_id_by_name = {}
    for grfn_name in comparison_az["grfn_ids"]:
        grfn_id = comparison_az["grfn_ids"][grfn_name]
        grfn_id_by_name[grfn_name] = grfn_id
        if grfn_id not in grfn_by_id:
            raise Exception("A GrFN file with uid %s has not been provided" % grfn_id)

    comparison_hmi = {"subgraphs": []}

    # populate comparison_hmi
    for grfn_name in grfn_id_by_name:
        grfn_id = grfn_id_by_name[grfn_name]
        populate_comparison_hmi(comparison_hmi, comparison_az, grfn_name, grfn_by_id[grfn_id], grfn_id_by_name)

    with open(args.output, "w") as f:
        json.dump(comparison_hmi, f)

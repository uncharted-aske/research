import json
from argparse import ArgumentParser


def create_or_update_edge(edge_id, source_id, target_id, established_edges):
    if (source_id, target_id) in established_edges:
        established_edges[(source_id, target_id)]["metadata"]["multiplicity"] += 1
    else:
        edge_id += 1
        established_edges[(source_id, target_id)] = {
            "id": str(edge_id),
            "source": str(source_id),
            "target": str(target_id),
            "metadata": {
                "multiplicity": 1
            }
        }
    return edge_id


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', required=True,
                        help='Input GTRI .json file')
    parser.add_argument('--output', required=True,
                        help='The location of the resulting output file')
    args = parser.parse_args()

    nodes = []
    edges = []

    with open(args.input) as f:
        data = json.load(f)

        node_id = 0

        for t in data["T"]:
            node_id += 1
            nodes.append({
               "id": str(node_id),
                "concept": t["tname"],
                "label": t["tname"],
                "type": "transition",
                "metadata": {
                    "rate": t["rate"]
                }
            })

        for s in data["S"]:
            node_id += 1
            nodes.append({
                "id": str(node_id),
                "concept": s["sname"],
                "label": s["sname"],
                "type": "state",
                "metadata": {
                    "concentration": s["concentration"]
                }
            })

        edge_id = 0

        established_edges = {}

        for i in data["I"]:
            source_id = len(data["T"]) + int(i["is"])
            target_id = i["it"]
            edge_id = create_or_update_edge(edge_id, source_id, target_id, established_edges)

        for o in data["O"]:
            source_id = o["ot"]
            target_id = len(data["T"]) + int(o["os"])
            edge_id = create_or_update_edge(edge_id, source_id, target_id, established_edges)

        edges.extend(established_edges.values())

    with open(args.output, 'w') as f:
        json.dump({
            "nodes": nodes,
            "edges": edges
        }, f)
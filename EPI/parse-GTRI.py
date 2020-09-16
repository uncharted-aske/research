import json
from argparse import ArgumentParser

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

        for i in data["I"]:
            edge_id += 1
            edges.append({
                "id": str(edge_id),
                "source": str(len(data["T"]) + int(i["is"])),
                "target": str(i["it"])
            })

        for o in data["O"]:
            edge_id += 1
            edges.append({
                "id": str(edge_id),
                "source": str(o["ot"]),
                "target": str(len(data["T"]) + int(o["os"])),
            })

    with open(args.output, 'w') as f:
        json.dump({
            "nodes": nodes,
            "edges": edges
        }, f)
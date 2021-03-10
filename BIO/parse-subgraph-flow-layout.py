import json
from argparse import ArgumentParser

def formatGraph(nodes, edges): 
    formattedNodes = []
    formattedEdges = []
   
    for index in range(1, len(nodes)):       
        formattedNodes.append({ 'id': str(nodes[index]['id']), 'label':nodes[index]['name'], 'nodeType': 'ontological grounding', 'metadata': { 'db_refs': nodes[index]['db_refs']}, 
        })

    for index in range(1, len(edges) -1):
        metadata = edges[index]
        formattedEdges.append({ 'id': str(edges[index]['id']), 'source': str(edges[index]['source_id']), 'target':str(edges[index]['target_id']), 'edgeType': edges[index]['type'], 'metadata': metadata  })
    
    return  { 'nodes': formattedNodes,'edges': formattedEdges }

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--nodes', required=True,
                        help='Nodes .json file')
    parser.add_argument('--edges', required=True,
                        help='Edges .json file')
    parser.add_argument('--output', required=True,
                        help='Output .json file')
    args = parser.parse_args()

    with open(args.nodes) as f_nodes, open(args.edges) as f_edges:
        nodes = json.load(f_nodes)
        edges = json.load(f_edges)
        graph = formatGraph(nodes, edges) 

    with open(args.output, 'w') as f:
        json.dump(graph, f)
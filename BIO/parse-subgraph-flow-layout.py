import json
from argparse import ArgumentParser

nodes_neighborhood_file = 'nodes_neighborhood.json'
edges_neighborhood_file = 'edges_neighborhood.json'

def formatGraph(nodes, edges, evidences, nodes_neighborhood, edges_neighborhood): 
    formattedNodes = []
    formattedEdges = []
    evidences_hash = {}
    nodes_neighborhood_hash = {}

    for evidence in evidences:
        for statement_id in evidence['statement_ids']:
            evidences_hash[statement_id] = evidence['text']

    for node in nodes_neighborhood:
        nodes_neighborhood_hash[node['id']] = node['name']

    for index in range(1, len(nodes)):
        node_id = nodes[index]['id']
        # Find incoming neighbor edges
        incoming_neighbors = [{'id': str(edge['id']), 'source': str(edge['source_id']), 'source_label': nodes_neighborhood_hash[edge['source_id']], 'target':str(edge['target_id']), 'target_label': nodes_neighborhood_hash[edge['target_id']], 'edgeType': edge['type'], 'metadata': edge  } for edge in edges_neighborhood if edge['target_id'] == node_id]
        outgoing_neighbors = [{'id': str(edge['id']), 'source': str(edge['source_id']), 'source_label': nodes_neighborhood_hash[edge['source_id']], 'target':str(edge['target_id']), 'target_label': nodes_neighborhood_hash[edge['target_id']], 'edgeType': edge['type'], 'metadata': edge  } for edge in edges_neighborhood if edge['source_id'] == node_id]

        formattedNodes.append({ 'id': str(nodes[index]['id']), 'label':nodes[index]['name'], 'nodeType': 'ontological grounding', 'metadata': { 'db_refs': nodes[index]['db_refs'], 'incoming_neighbors': incoming_neighbors, 'outgoing_neighbors': outgoing_neighbors }})

    for index in range(1, len(edges) -1):
        statement_id = edges[index]['statement_id']
        evidence = evidences_hash[statement_id]
        metadata = edges[index]
        metadata['evidence'] = evidence
        formattedEdges.append({ 'id': str(edges[index]['id']), 'source': str(edges[index]['source_id']), 'target':str(edges[index]['target_id']), 'edgeType': edges[index]['type'], 'metadata': metadata  })
    
    return  { 'nodes': formattedNodes,'edges': formattedEdges }

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--nodes', required=True,
                        help='Nodes .json file')
    parser.add_argument('--edges', required=True,
                        help='Edges .json file')
    parser.add_argument('--evidences', required=True,
                        help='Evidences .json file')
    parser.add_argument('--output', required=True,
                        help='Output .json file')
    args = parser.parse_args()

    with open(args.nodes) as f_nodes, open(args.edges) as f_edges, open(args.evidences) as f_evidences, open(nodes_neighborhood_file) as f_n_neighborhood, open(edges_neighborhood_file) as f_e_neighborhood:
        nodes = json.load(f_nodes)
        edges = json.load(f_edges)
        nodes_neighborhood = json.load(f_n_neighborhood)
        edges_neighborhood = json.load(f_e_neighborhood)
        evidences = json.load(f_evidences)
        graph = formatGraph(nodes, edges, evidences, nodes_neighborhood, edges_neighborhood) 

    with open(args.output, 'w') as f:
        json.dump(graph, f)





         

import json
from argparse import ArgumentParser


def formatGraph(data): 
    modelMetadata = {}
    nodesDict = {}
    formattedNodes = []
    formattedEdges = []

    variables = data['variables']
    for variable in variables:
        metadata = variable['metadata'][0] if 'metadata' in variable else {}
        variable['nodeType'] = 'variable'
        variable['metadata'] = metadata
        nodesDict[variable['uid']] = variable
    
    edges = data['edges']
    for i in range(len(edges)):
        formattedEdges.append({ 'id': i, 'source': edges[i][0], 'target': edges[i][1]})
    
    subgraphs = data['subgraphs']
    for subgraph in subgraphs:
        # Get parent name
        parent_name = subgraph['scope']
        if (parent_name == '@global'):
            parent_name = 'root'

        # Get container name
        splitted_id = subgraph['basename'].split('.')

        #Get container metadata
        metadata = subgraph['metadata'][0] if 'metadata' in subgraph else {}

        node = { 'id': subgraph['basename'], 'concept': splitted_id[(len(splitted_id) - 1)], 'nodeType': subgraph['type'], 'label': splitted_id[(len(splitted_id) - 1)], 'parent': parent_name, 'metadata': metadata }
        formattedNodes.append(node)
        for n in subgraph['nodes']:
            found = n in nodesDict
            if (found): 
                found_node = nodesDict[n]
                # Variables
                if (found_node['nodeType'] == 'variable'):
                    splitted_identifier = found_node['identifier'].split('::')
                    node = { 'id': n, 'concept': splitted_identifier[len(splitted_identifier)-2], 'label': splitted_identifier[len(splitted_identifier)-2], 'nodeType': 'variable', 'type': found_node['type'], 'parent': subgraph['basename'], 'metadata': found_node['metadata']}
                    formattedNodes.append(node)
                else: 
                    raise Exception('Unrecognized node type')
            else:
                raise Exception(n + ' Node missing')

        # Append root for visualization purposes so we don't have multiple roots
        formattedNodes.append({'concept': 'root', 'parent': '', 'id': 'root'}) 
    
    metadata = data['metadata'] if 'metadata' in data else {}
    modelMetadata = metadata
    
    return  { 'metadata': modelMetadata, 'nodes': formattedNodes,'edges': formattedEdges }

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', required=True,
                        help='Input GrFN .json file')
    parser.add_argument('--output', required=True,
                        help='The location of the resulting output file')
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
        graph = formatGraph(data)

    with open(args.output, 'w') as f:
        json.dump(graph, f)





         

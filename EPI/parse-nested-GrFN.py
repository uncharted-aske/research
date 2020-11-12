import json
from argparse import ArgumentParser


def formatGraph(data): 
    modelMetadata = {}
    nodesDict = {}
    nodes = []
    edges = []

    for key,value in data.items():
        if key == 'variables':
            for item in value: 
                if 'metadata' in item:
                    metadata = item['metadata'][0]
                else:
                    metadata = {}
                item['nodeType'] = 'variable'
                item['metadata'] = metadata
                nodesDict[item['uid']] = item
        if key == 'functions':
            for item in value: 
                if 'metadata' in item:
                    metadata = item['metadata'][0]
                else:
                    metadata = {} 
                item['nodeType'] = 'function'
                item['metadata'] = metadata
                nodesDict[item['uid']] = item
        if key == 'hyper_edges':
            for item in value: 
                if len(item['inputs']) > 0:
                    for i in item['inputs']: # Inputs array is sometimes empty
                        first_edge = { 'source': i, 'target': item['function']}
                        edges.append(first_edge)
                
                for o in item['outputs']:
                    second_edge = { 'source': item['function'], 'target': o }
                    edges.append(second_edge)
        if key == 'subgraphs':
            for item in value: 
                # Get parent name
                parent_name = item['scope']
                if (parent_name == '@global'):
                    parent_name = 'root'

                # Get container name
                splitted_id = item['basename'].split('.')

                #Get container metadata
                if 'metadata' in item:
                    metadata = item['metadata'][0]

                node = { 'id': item['basename'], 'concept': splitted_id[(len(splitted_id) - 1)], 'nodeType': item['type'], 'label': splitted_id[(len(splitted_id) - 1)], 'parent': parent_name, 'metadata': metadata }
                nodes.append(node)
                for n in item['nodes']:
                    found = nodesDict[n]
                    # Variables
                    if (found['nodeType'] == 'variable'):
                        splitted_identifier = found['identifier'].split('::')
                        node = { 'id': n, 'concept': splitted_identifier[len(splitted_identifier)-2], 'label': splitted_identifier[len(splitted_identifier)-2], 'nodeType': 'variable', 'type': found['type'], 'parent': item['basename'], 'metadata': found['metadata']}
                    # Functions
                    else:
                        node = { 'id': n, 'concept': found['type'], 'label': found['type'], 'type': found['type'], 'nodeType': 'function', 'parent': item['basename'], 'metadata': found['metadata']}
                    
                    nodes.append(node)
       
            # Append root for visualization purposes so we don't have multiple roots
            nodes.append({'concept': 'root', 'parent': '', 'id': 'root'}) 
        
        if key == 'metadata':
            modelMetadata = value[0]
            variableTypes = value[0]['attributes']
            variableTypesDict = {}
            # Distinguish variable type
            for item in variableTypes:
                for i in item['inputs']:
                    variableTypesDict[i] = 'input'
                for i in item['outputs']:
                    variableTypesDict[i] = 'output'
                for i in item['parameters']:
                    variableTypesDict[i] = 'parameter'
                for i in item['model_variables']:
                    variableTypesDict[i] = 'model_variable'
                for i in item['initial_conditions']:
                    variableTypesDict[i] = 'initial_condition'
                for i in item['internal_variables']:
                    variableTypesDict[i] = 'internal_variable'
            for node in nodes:
                if 'nodeType' in node:
                    if (node['nodeType'] == 'variable'):
                        if (node['id'] in variableTypesDict):
                            node['varType'] = variableTypesDict[node['id']]

        
    for i in range(len(edges)):
        edges[i]['id'] = i 

    return {
        'metadata': modelMetadata,
        'nodes': nodes,
        'edges': edges
    }

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





         

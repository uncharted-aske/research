import json
from argparse import ArgumentParser


def formatGraph(data): 
    modelMetadata = {}
    nodesDict = {}
    nodes = []
    edges = []

    # FIXME: We might want to only check for fields we are interested in instead of just going over all items. 
    for key,value in data.items():
        if key == 'variables':
            for variable in value: 
                metadata = variable['metadata'][0] if 'metadata' in variable else {}

                variable['nodeType'] = 'variable'
                variable['metadata'] = metadata
                nodesDict[variable['uid']] = variable
        if key == 'functions':
            for function in value: 
                metadata = function['metadata'][0] if 'metadata' in function else {}
 
                function['nodeType'] = 'function'
                function['metadata'] = metadata
                nodesDict[function['uid']] = function
        if key == 'hyper_edges':
            for edge in value: 
                [edges.append({ 'source': i, 'target': edge['function']}) for i in edge['inputs']]
                [edges.append({ 'source': edge['function'], 'target': o }) for o in edge['outputs']]
            for i in range(len(edges)):
                edges[i]['id'] = i

        if key == 'subgraphs':
            for subgraph in value: 
                # Get parent name
                parent_name = subgraph['scope']
                if (parent_name == '@global'):
                    parent_name = 'root'

                # Get container name
                splitted_id = subgraph['basename'].split('.')

                #Get container metadata
                metadata = subgraph['metadata'][0] if 'metadata' in subgraph else {}

                node = { 'id': subgraph['basename'], 'concept': splitted_id[(len(splitted_id) - 1)], 'nodeType': subgraph['type'], 'label': splitted_id[(len(splitted_id) - 1)], 'parent': parent_name, 'metadata': metadata }
                nodes.append(node)
                for n in subgraph['nodes']:
                    found = n in nodesDict
                    if (found): 
                        found_node = nodesDict[n]
                        # Variables
                        if (found_node['nodeType'] == 'variable'):
                            splitted_identifier = found_node['identifier'].split('::')
                            node = { 'id': n, 'concept': splitted_identifier[len(splitted_identifier)-2], 'label': splitted_identifier[len(splitted_identifier)-2], 'nodeType': 'variable', 'type': found_node['type'], 'parent': subgraph['basename'], 'metadata': found_node['metadata']}
                        # Functions
                        elif found_node['nodeType'] == 'function':
                            node = { 'id': n, 'concept': found_node['type'], 'label': found_node['type'], 'type': found_node['type'], 'nodeType': 'function', 'parent': subgraph['basename'], 'metadata': found_node['metadata']}
                        else: 
                            raise Exception('Unrecognized node type')

                        nodes.append(node)
                    else:
                        raise Exception(n + ' Node missing')

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
                if 'nodeType' in node and node['nodeType'] == 'variable' and node['id'] in variableTypesDict:
                    node['varType'] = variableTypesDict[node['id']]
    
    return  { 'metadata': modelMetadata, 'nodes': nodes,'edges': edges }

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





         
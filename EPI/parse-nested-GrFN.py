import json
from argparse import ArgumentParser


def formatGraph(data): 
    modelMetadata = {}
    nodesDict = {}
    nodes = []
    edges = []

    variables = data['variables']
    for variable in variables:
        metadata = variable['metadata'][0] if 'metadata' in variable else {}
        variable['nodeType'] = 'variable'
        variable['dataType'] = variable['type']
        variable['metadata'] = metadata
        nodesDict[variable['uid']] = variable
   
    functions = data['functions']
    for function in functions:
        metadata = function['metadata'][0] if 'metadata' in function else {}
        function['nodeType'] = 'function'
        function['dataType'] = function['type']
        function['metadata'] = metadata
        nodesDict[function['uid']] = function
    
    hyperEdges = data['hyper_edges']
    for edge in hyperEdges: 
        [edges.append({ 'source': i, 'target': edge['function']}) for i in edge['inputs']]
        [edges.append({ 'source': edge['function'], 'target': o }) for o in edge['outputs']]

    subgraphs = data['subgraphs']
    for subgraph in subgraphs:
        #Get container metadata
        metadata = subgraph['metadata'][0] if 'metadata' in subgraph else {}

        #Get parent id
        parent_id = 'root' if subgraph['parent'] is None else subgraph['parent'] 

        node = { 'id': subgraph['uid'], 'concept': subgraph['basename'], 'nodeType': 'container', 'label': subgraph['basename'], 'parent': parent_id, 'metadata': metadata }
        nodes.append(node)
        for n in subgraph['nodes']:
            found = n in nodesDict
            if (found): 
                found_node = nodesDict[n]
                # Variables
                if (found_node['nodeType'] == 'variable'):
                    splitted_identifier = found_node['identifier'].split('::')
                    node = { 'id': n, 'concept': splitted_identifier[len(splitted_identifier)-2], 'label': splitted_identifier[len(splitted_identifier)-2], 'nodeType': 'variable', 'dataType': found_node['type'], 'parent': subgraph['uid'], 'metadata': found_node['metadata']}
                # Functions
                elif found_node['nodeType'] == 'function':
                    node = { 'id': n, 'concept': found_node['type'], 'label': found_node['type'], 'dataType': found_node['type'], 'nodeType': 'function', 'parent': subgraph['uid'], 'metadata': found_node['metadata']}
                else: 
                    raise Exception('Unrecognized node type')

                nodes.append(node)
            else:
                raise Exception(n + ' Node missing')

    metadata = data['metadata'] if 'metadata' in data else {}
    modelMetadata = metadata
    
    if modelMetadata:
        variableTypes = modelMetadata[0]['attributes']
        variableTypesDict = {}
        for varType in variableTypes:
            for i in varType['inputs']:
                variableTypesDict[i] = ['input']
            for i in varType['outputs']:
                variableTypesDict[i] = ['output']
            
            # Distinguish variable type (inputs and outputs can also be classified as model variables, params, initial conditions or internal variables)
            for i in varType['parameters']:
                found = i in variableTypesDict
                if (found):
                    variableTypesDict[i].append('parameter')
                else:
                    variableTypesDict[i] = ['parameter']
            for i in varType['model_variables']:
                found = i in variableTypesDict
                if (found):
                    variableTypesDict[i].append('model_variable')
                else:
                    variableTypesDict[i] = ['model_variable']
            for i in varType['initial_conditions']:
                found = i in variableTypesDict
                if (found):
                    variableTypesDict[i].append('initial_condition')
                else:
                    variableTypesDict[i] = 'initial_condition'
            for i in varType['internal_variables']:
                found = i in variableTypesDict
                if (found):
                    variableTypesDict[i].append('internal_variable')
                else:
                    variableTypesDict[i] = 'internal_variable'
        for node in nodes:
            if 'nodeType' in node and node['nodeType'] == 'variable' and node['id'] in variableTypesDict:
                node['nodeSubType'] = variableTypesDict[node['id']]
            else:
                node['nodeSubType'] = []
    
    # Append root for visualization purposes so we don't have multiple roots
    nodes.append({'concept': 'root', 'parent': None, 'id': 'root'}) 
    
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





         

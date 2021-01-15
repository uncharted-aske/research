import json
from argparse import ArgumentParser


def populate_types_dict(types, type_name, types_dict):
    for t in types:
        if t in types_dict:
            types_dict[t].append(type_name)
        else:
            types_dict[t] = [type_name]


def format_graph(data):
    nodes_dict = {}
    nodes = []
    edges = []

    for variable in data['variables']:
        metadata = variable['metadata'][0] if 'metadata' in variable else {}
        variable['nodeType'] = 'variable'
        variable['dataType'] = variable['type']
        variable['metadata'] = metadata
        nodes_dict[variable['uid']] = variable
   
    for function in data['functions']:
        metadata = function['metadata'][0] if 'metadata' in function else {}
        function['nodeType'] = 'function'
        function['dataType'] = function['type']
        function['metadata'] = metadata
        nodes_dict[function['uid']] = function
    
    for edge in data['hyper_edges']:
        [edges.append({'source': i, 'target': edge['function']}) for i in edge['inputs']]
        [edges.append({'source': edge['function'], 'target': o}) for o in edge['outputs']]

    for subgraph in data['subgraphs']:
        # Get container metadata
        metadata = subgraph['metadata'][0] if 'metadata' in subgraph else {}

        # Get parent id
        parent_id = 'root' if subgraph['parent'] is None else subgraph['parent'] 

        nodes.append({
            'id': subgraph['uid'],
            'concept': subgraph['basename'],
            'nodeType': 'container',
            'label': subgraph['basename'],
            'parent': parent_id,
            'metadata': metadata
        })
        for n in subgraph['nodes']:
            if n in nodes_dict:
                found_node = nodes_dict[n]
                # Variables
                if found_node['nodeType'] == 'variable':
                    splitted_identifier = found_node['identifier'].split('::')
                    nodes.append({
                        'id': n,
                        'concept': splitted_identifier[len(splitted_identifier)-2],
                        'label': splitted_identifier[len(splitted_identifier)-2],
                        'nodeType': 'variable',
                        'dataType': found_node['type'],
                        'parent': subgraph['uid'],
                        'metadata': found_node['metadata']
                    })
                # Functions
                elif found_node['nodeType'] == 'function':
                    nodes.append({
                        'id': n,
                        'concept': found_node['type'],
                        'label': found_node['type'],
                        'dataType': found_node['type'],
                        'nodeType': 'function',
                        'parent': subgraph['uid'],
                        'metadata': found_node['metadata']
                    })
                else: 
                    raise Exception('Unrecognized node type: %s' % found_node['nodeType'])
            else:
                raise Exception('Node missing: %s' % n)

    metadata = data['metadata'] if 'metadata' in data else {}

    if metadata:
        variable_types = metadata[0]['attributes']
        variable_types_dict = {}
        for var_type in variable_types:
            for i in var_type['inputs']:
                variable_types_dict[i] = ['input']
            for i in var_type['outputs']:
                variable_types_dict[i] = ['output']

            # Distinguish variable type (inputs and outputs can also be classified as model variables, params,
            #   initial conditions or internal variables)
            populate_types_dict(var_type['parameters'], 'parameter', variable_types_dict)
            populate_types_dict(var_type['model_variables'], 'model_variable', variable_types_dict)
            populate_types_dict(var_type['initial_conditions'], 'initial_condition', variable_types_dict)
            populate_types_dict(var_type['internal_variables'], 'internal_variable', variable_types_dict)

        for node in nodes:
            if 'nodeType' in node and node['nodeType'] == 'variable' and node['id'] in variable_types_dict:
                node['nodeSubType'] = variable_types_dict[node['id']]
            else:
                node['nodeSubType'] = []

    # Append root for visualization purposes so we don't have multiple roots
    nodes.append({'concept': 'root', 'parent': None, 'id': 'root'})
    
    return {'metadata': metadata, 'nodes': nodes,'edges': edges}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', required=True,
                        help='Input GrFN .json file')
    parser.add_argument('--output', required=True,
                        help='The location of the resulting output file')
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
        graph = format_graph(data)

    with open(args.output, 'w') as f:
        json.dump(graph, f)

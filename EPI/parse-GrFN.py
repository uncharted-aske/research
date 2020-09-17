import json
from argparse import ArgumentParser

def formatGraph(data): 
    nodes = []
    edges = []
    groups = []

    for key,value in data.items():
        if key == 'variables':
            for item in value:  
                node = { 'id': item['uid'], 'concept': item['identifier'], 'label': item['identifier'], 'type': 'variable' } # FIXME: We might want to clean up the node labels
                nodes.append(node)
        if key == 'functions':
            for item in value:  
                node = { 'id': item['uid'], 'concept': item['type'], 'label': item['type'],  'type': 'function'  }
                nodes.append(node)
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
                group = { 'id': item['basename'], 'members': item['nodes'] }
                groups.append(group) 


    return {
        'groups': groups,
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





         

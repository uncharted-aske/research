import json
from argparse import ArgumentParser

def formatGraph(data): 
    nodes = []
    edges = []
    groups = []
    modelMetadata = {}

    for key,value in data.items():
        if key == 'metadata':
            modelMetadata = { 'name': value[1]['attributes'][0]['model_name'], 'description': value[1]['attributes'][0]['model_description'], 'authors': value[2]['attributes'][0]['authors'], 'sources': value[1]['provenance']['sources'][0] }
        if key == 'variables':
            for item in value: 
                splitted_identifier = item['identifier'].split('::')
                if 'metadata' in item:
                    metadata = { 'type': item['type'], 'text_identifier':item['metadata'][0]['attributes'][0]['text_identifier'], 'text_definition': item['metadata'][0]['attributes'][0]['text_definition'], 'knowledge': item['metadata'][0]['provenance']['sources'][0]['document_source'] }
                else:
                    metadata = {}
                node = { 'id': item['uid'], 'concept': item['identifier'], 'label': splitted_identifier[len(splitted_identifier)-2], 'type': 'variable', 'metadata': metadata } 
                nodes.append(node)
        if key == 'functions':
            for item in value: 
                if 'metadata' in item:
                    metadata = { 'type': item['type'], 'eqn_source':item['metadata'][0]['attributes'][0]['eqn_source'], 'knowledge': item['metadata'][0]['provenance']['sources'][0]['document_source'] }
                else:
                    metadata = {} 
                node = { 'id': item['uid'], 'concept': item['type'], 'label': item['type'],  'type': 'function', 'metadata': metadata  }
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
                if 'metadata' in item:
                    metadata = item['metadata']
                else:
                    metadata = {}
                group = { 'id': item['basename'], 'members': item['nodes'], 'metadata': metadata }
                groups.append(group) 

        
    for i in range(len(edges)):
        edges[i]['id'] = i 

    return {
        'metadata': modelMetadata,
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





         

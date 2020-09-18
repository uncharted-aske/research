import json
from argparse import ArgumentParser

def formatGraph(data): 
    nodes = []
    edges = []

    for key,value in data.items():
        if key == 'variables':
            for item in value:  
                splitted_identifier = item['identifier'].split('::')
                node = { 'id': item['uid'], 'concept': item['identifier'], 'label': splitted_identifier[len(splitted_identifier)-2], 'metadata': splitted_identifier[len(splitted_identifier)-1]  }
                nodes.append(node)
        if key == 'edges':
            for item in value: 
                edges.append({'source': item[0], 'target': item[1]})
                


    return {
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





         

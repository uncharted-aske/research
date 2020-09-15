import sys
import json

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
                group = { 'id': item['scope'], 'members': item['nodes'] }
                groups.append(group) 


    return {
        'groups': groups,
        'nodes': nodes,
        'edges': edges
    }



def main(argv):
   inputfile = argv
   outputfile = inputfile.replace('.json', '')
   outputfile = outputfile + '_formatted.json'

   with open(inputfile) as f:
    data = json.load(f)

    num_edges = 0
    num_nodes = 0

 
    graph = formatGraph(data)
    print('Number of nodes', len(graph['nodes']))
    print('Number of edges', len(graph['edges']))

    with open(outputfile, 'w') as f:
        json.dump(graph, f)
 

if __name__ == "__main__":
   main(sys.argv[1])




         

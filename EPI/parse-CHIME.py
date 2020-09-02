import json

def formatGraph(data): 
    nodes = []
    edges = []
    groups = []

    for key,value in data.items():
        if key == 'variables':
            for item in value:  
                label =  item['identifier'].replace('CHIME-SIR::', '')
                node = { 'id': item['uid'], 'concept': label, 'label': label, 'type': 'variable' }
                nodes.append(node)
        if key == 'functions':
            for item in value:  
                node = { 'id': item['uid'], 'concept': item['type'], 'label': item['type'],  'type': 'function'  }
                nodes.append(node)
        if key == 'hyper_edges':
            for item in value: 
                if len(item['inputs']) > 0:
                    for i in item['inputs']:
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


with open('models/CHIME-SIR--GrFN.json') as f:
    data = json.load(f)

    num_edges = 0
    num_nodes = 0

 
    graph = formatGraph(data)
    aggregatedGraph = formatAggregatedGraph(graph)
    print('Number of nodes', len(graph['nodes']))
    print('Number of edges', len(graph['edges']))

    with open('models/formatted-CHIME-SIR--GrFN.json', 'w') as f:
        json.dump(graph, f)




         

interface NodeSpec {
    id: string;
    concept: string;
    role: string[];
    label: string;
    nodeType: string;
    dataType: string;
    parent: string | null;
    nodeSubType: string[];
    metadata: any;
}

interface EdgeSpec {
    source: string;
    target: string;
}

interface GraphSpec {
    nodes: NodeSpec[];
    edges: EdgeSpec[];
    metadata: any[] | null;
}
/// <reference path="../types/graph.d.ts" />

import GrometElm = GroMEt.GrometElm;
import Box = GroMEt.Box;
import HasContents = GroMEt.HasContents;
import Port = GroMEt.Port;
import Wire = GroMEt.Wire;
import Expression = GroMEt.Expression;
import Expr = GroMEt.Expr;
import Literal = GroMEt.Literal;
import BoxCall = GroMEt.BoxCall;
import PortCall = GroMEt.PortCall;
import Relation = GroMEt.Relation;
import Junction = GroMEt.Junction;
import Gromet = GroMEt.Gromet;
import ModelInterface = GroMEt.ModelInterface;
import {GroMEtMap} from './GroMEtMap.ts';
import Loop = GroMEt.Loop;
import Conditional = GroMEt.Conditional;

export class GroMEt2Graph extends GroMEtMap {
    private idStack: number[] = [];
    private roleMap: Map<string, Set<string>> = new Map();
    private varMetaMap: Map<string, any[]> = new Map();
    private idMap: Map<string, number> = new Map();
    private uniqueID: number = 0;

    public static parseGromet(gromet: Gromet): GraphSpec {
        const inst = new GroMEt2Graph(gromet);
        return inst.toGraph();
    }

    private constructor(gromet: Gromet) {
        super(gromet);
        this.processMetadata();
        this.processVariables();
    }

    private processMetadataReferences(ids: string[], setName: string): void {
        if (!this.roleMap.has(setName)) {
            this.roleMap.set(setName, new Set());
        }
        const set = this.roleMap.get(setName) as Set<string>;

        for (const id of ids) {
            // look in the variables first (because reasons) and then junctions
            if (this.vars && this.vars.has(id)) {
                const varObj = this.vars.get(id);
                if (varObj) {
                    set.add(varObj.states[0]);
                }
            } else if (this.junctions && this.junctions.has(id)) {
                set.add(id);
            }
        }
    }

    private processMetadata(): void {
        if (this.gromet.metadata) {
            for (const datum of this.gromet.metadata) {
                if (datum.metadata_type === 'ModelInterface') {
                    const meta = datum as ModelInterface;
                    this.processMetadataReferences(meta.variables, 'variables');
                    this.processMetadataReferences(meta.parameters, 'parameters');
                    this.processMetadataReferences(meta.initial_conditions, 'initial_conditions');
                }
            }
        }
    }

    private processVariables(): void {
        if (this.vars) {
            for (const variable of this.vars.values()) {
                for (const stateID of variable.states) {
                    const varMetaArr = this.varMetaMap.get(stateID);
                    if (varMetaArr) {
                        varMetaArr.push(variable.metadata);
                    } else {
                        this.varMetaMap.set(stateID, [variable.metadata]);
                    }
                }
            }
        }
    }

    private getElementRoles(id: string): string[] {
        const result = [];

        if (this.roleMap.get('variables')?.has(id)) {
            result.push('variable');
        }

        if (this.roleMap.get('parameters')?.has(id)) {
            result.push('parameter');
        }

        if (this.roleMap.get('initial_conditions')?.has(id)) {
            result.push('initial_condition');
        }

        return result;
    }

    private getVariableMetadata(id: string): any[] {
        return this.varMetaMap.get(id) || [];
    }

    private toGraph(): GraphSpec {
        const graph: GraphSpec = {
            nodes: [],
            edges: [],
            metadata: this.gromet.metadata,
        };

        if (!this.gromet.root) {
            throw 'GroMEts without a root box are not supported!';
        }

        this.parseElement(this.getElement(this.gromet.root, this.boxes), null, graph);

        return graph;
    }

    private getElement<T>(id: string, map: Map<string, T> | null): T {
        if (!map) {
            throw `Trying to get element [${id}] from a GroMEt without elements of that type defined`;
        }
        const element = map.get(id);
        if (!element) {
            throw `Could not find element ${id}`;
        }
        return element;
    }

    private getID(...IDs: Array<string | number>): string {
        return this.getStackID(this.idStack, ...IDs);
    }

    private getStackID(stack: number[], ...IDs: Array<string | number>): string {
        return `${stack.join('::')}${IDs.length ? `::${IDs.join('::')}`: ''}`;
    }

    private registerUID(uid: string): number {
        if (!this.idMap.has(uid)) {
            this.idMap.set(uid, this.idMap.size);
            // console.log(`${uid} => ${this.idMap.get(uid)}`);
        }
        return this.idMap.get(uid) as number;
    }

    private createFauxBox(name: string, syntax: string): Box {
        return {
            name,
            syntax,
            uid: `_GEN_FAUX_ID_${this.uniqueID++}_`,
            ports: null,
            type: null,
            metadata: null,
        }
    }

    private parseBox(box: Box, parent: string | null, graph: GraphSpec): void {
        const node: NodeSpec = {
            id: this.getID(),
            grometID: box.uid,
            concept: box.syntax,
            role: this.getElementRoles(box.uid),
            label: box.name || box.uid,
            nodeType: 'Box',
            dataType: box.type || box.syntax,
            parent: parent,
            nodeSubType: [ box.syntax ],
            metadata: [box.metadata, ...this.getVariableMetadata(box.uid)].filter(v => Boolean(v)),
        };
        graph.nodes.push(node);

        // parse ports
        if (box.ports) {
            for (const portID of box.ports) {
                this.parseElement(this.getElement(portID, this.ports), node.id, graph);
            }
        }
    }

    private parseHasContents(hc: HasContents, parent: string | null, graph: GraphSpec): void {
        const id = this.getID();
        if (hc.wires) {
            for (const wireID of hc.wires) {
                this.parseElement(this.getElement(wireID, this.wires), id, graph);
            }
        }

        if (hc.boxes) {
            for (const boxID of hc.boxes) {
                this.parseElement(this.getElement(boxID, this.boxes), id, graph);
            }
        }

        if (hc.junctions) {
            for (const junctionID of hc.junctions) {
                this.parseElement(this.getElement(junctionID, this.junctions), id, graph);
            }
        }
    }

    private parseFunction(fn: GroMEt.Function, parent: string | null, graph: GraphSpec): void {
        this.parseBox(fn, parent, graph);
        this.parseHasContents(fn, parent, graph);
    }

    private parsePort(port: Port, parent: string | null, graph: GraphSpec): void {
        const node: NodeSpec = {
            id: this.getID(),
            grometID: port.uid,
            concept: port.syntax,
            role: this.getElementRoles(port.uid),
            label: port.name || port.uid,
            nodeType: 'Port',
            dataType: port.value_type || port.type || port.syntax,
            parent: parent,
            nodeSubType: [ port.type || port.syntax ],
            metadata: [port.metadata, ...this.getVariableMetadata(port.uid)].filter(v => Boolean(v)),
        };
        graph.nodes.push(node);
    }

    private getPortCallID(nodeID: string): string {
        const graphID = this.registerUID(nodeID);
        // if the node is a port, check the parent ID
        try {
            const port = this.getElement(nodeID, this.ports);
            const parentID = this.registerUID(port.box);

            const idStack = [...this.idStack];
            while (idStack.length) {
                if (idStack.pop() === parentID) {
                    return this.getStackID(idStack, parentID, graphID);
                }
            }
        } catch {}
        return this.getID(graphID);
    }

    private parsePortCall(port: PortCall, parent: string | null, graph: GraphSpec): void {
        this.parsePort(port, parent, graph);
        // PortCall edges should be ignored
        // graph.edges.push({
        //     source: this.getID(),
        //     target: this.getPortCallID(port.call),
        // });
    }

    private getWireNodeID(nodeID: string): string {
        const graphID = this.registerUID(nodeID);
        // if the node is a port, check the parent ID
        try {
            const port = this.getElement(nodeID, this.ports);
            const parentID = this.registerUID(port.box);

            if (this.idStack[this.idStack.length - 1] !== parentID) {
                return this.getID(parentID, graphID);
            }
        } catch {}
        return this.getID(graphID);
    }

    private parseWire(wire: Wire, parent: string | null, graph: GraphSpec): void {
        if (!wire.src || !wire.tgt) {
            throw `Wires with unspecified source or target are not supported: (${wire.uid}) ${wire.src} => ${wire.tgt}`;
        }

        const edge: EdgeSpec = {
            source: this.getWireNodeID(wire.src),
            target: this.getWireNodeID(wire.tgt),
        };
        graph.edges.push(edge);
    }

    private parseExpression(exp: Expression, parent: string | null, graph: GraphSpec): void {
        this.parseBox(exp, parent, graph);

        if (!exp.ports) {
            throw `Expression types must have at least an output port! [${exp.uid}]`;
        }

        let outPort = null;
        for (const portID of exp.ports) {
            const port = this.getElement(portID, this.ports);
            if (port.type && port.type.indexOf('PortOutput') !== -1) {
                if (!outPort) {
                    outPort = port;
                } else {
                    throw `Multiple port outputs found for expression [${exp.uid}]`;
                }
            }
        }

        if (!outPort) {
            throw `Expressions must contain at least one output port! [${exp.uid}]`;
        }

        const outID = this.registerUID(outPort.uid);
        this.parseElement(exp.tree, this.getID(), graph, this.getID(outID));
    }

    private parseExpr(exp: Expr, parent: string | null, graph: GraphSpec, out: string): void {
        switch (exp.call.syntax) {
            case 'RefOp': {
                const id = `_GEN_EXPR_ID_${this.uniqueID++}_`;
                const graphID = this.registerUID(id);
                const node: NodeSpec = {
                    id: this.getID(graphID),
                    grometID: null,
                    concept: exp.call.syntax,
                    role: [], // Expr does not have a uid
                    label: exp.call.name,
                    nodeType: 'Expr',
                    dataType: exp.call.syntax,
                    parent: parent,
                    nodeSubType: [exp.syntax],
                    metadata: [], // no metadata or var metadata for Expr? // Future Dario, look into this :)
                }
                graph.nodes.push(node);

                if (exp.args) {
                    for (const arg of exp.args) {
                        if (typeof arg === 'string') {
                            graph.edges.push({
                                source: this.getID(this.registerUID(arg)),
                                target: node.id,
                            });
                        } else {
                            this.parseElement(arg as Expr, parent, graph, node.id);
                        }
                    }
                }
                graph.edges.push({
                    source: node.id,
                    target: out,
                });
            }
                break;

            default:
                throw `Unknown Expr syntax [${exp.call.syntax}]`;
        }
    }

    private parseLiteral(literal: Literal, parent: string | null, graph: GraphSpec, out: string): void {
        const id = `_GEN_LITERAL_ID_${this.uniqueID++}_`;
        const graphID = this.registerUID(id);
        const node: NodeSpec = {
            id: this.getID(graphID),
            grometID: literal.uid,
            concept: literal.syntax,
            role: literal.uid ? this.getElementRoles(literal.uid) : [],
            label: literal.name || literal.value.val.toString(),
            nodeType: 'Literal',
            dataType: literal.type || literal.syntax,
            parent: parent,
            nodeSubType: [literal.syntax],
            metadata: [literal.metadata, ...(literal.uid ? this.getVariableMetadata(literal.uid) : [])].filter(v => Boolean(v)),
        }
        graph.nodes.push(node);
        graph.edges.push({
            source: node.id,
            target: out,
        });
    }

    private parseBoxCall(call: BoxCall, parent: string | null, graph: GraphSpec): void {
        this.parseBox(call, parent, graph);

        if (call.ports) {
            for (const portCallID of call.ports) {
                const portCall = this.getElement(portCallID, this.ports) as PortCall;
                const port = this.getElement(portCall.call, this.ports);

                if (port.type && port.type.indexOf('PortOutput') !== -1) {
                    graph.edges.push({
                        source: this.getID(this.registerUID(call.call), this.registerUID(port.uid)),
                        target: this.getID(this.registerUID(portCallID)),
                    });
                } else {
                    graph.edges.push({
                        source: this.getID(this.registerUID(portCallID)),
                        target: this.getID(this.registerUID(call.call), this.registerUID(port.uid)),
                    });
                }
            }
        }

        const box = this.getElement(call.call, this.boxes);
        this.parseElement(box, this.getID(), graph);
    }

    private parseRelation(relation: Relation, parent: string | null, graph: GraphSpec): void {
        this.parseFunction(relation, parent, graph);
    }

    private parseJunction(junction: Junction, parent: string | null, graph: GraphSpec): void {
        const node: NodeSpec = {
            id: this.getID(),
            grometID: junction.uid,
            concept: junction.syntax,
            role: this.getElementRoles(junction.uid),
            label: junction.name || junction.uid,
            nodeType: 'Junction',
            dataType: junction.value_type || junction.type || junction.syntax,
            parent: parent,
            nodeSubType: [ junction.type || junction.syntax ],
            metadata: [junction.metadata, ...this.getVariableMetadata(junction.uid)].filter(v => Boolean(v)),
        };
        graph.nodes.push(node);
    }

    private parseLoop(loop: Loop, parent: string | null, graph: GraphSpec): void {
        this.parseBox(loop, parent, graph);
        this.parseHasContents(loop, parent, graph);
        if (loop.exit_condition) {
            const box = this.getElement(loop.exit_condition, this.boxes);
            this.parseElement(box, this.getID(), graph);
        }
    }

    private parseConditional(conditional: Conditional, parent: string | null, graph: GraphSpec): void {
        this.parseBox(conditional, parent, graph);

        for (let i = 0, n = conditional.branches.length; i < n; ++i) {
            const branch = conditional.branches[i];
            const faux = this.createFauxBox(`branch ${i}`, 'Function') as GroMEt.Function;
            faux.wires = null;
            faux.junctions = null;
            faux.boxes = [];

            if (branch[0]) {
                faux.boxes.push(branch[0]);
            }
            faux.boxes.push(branch[1]);

            this.parseElement(faux, this.getID(), graph);
        }
    }

    private _parseElement(element: GrometElm, parent: string | null, graph: GraphSpec, out?: string): void {
        switch (element.syntax) {
            case 'Function':
                this.parseFunction(element as GroMEt.Function, parent, graph);
                break;

            case 'Port':
                this.parsePort(element as Port, parent, graph);
                break;

            case 'PortCall':
                this.parsePortCall(element as PortCall, parent, graph);
                break;

            case 'Wire':
                this.parseWire(element as Wire, parent, graph);
                break;

            case 'Predicate':
            case 'Expression':
                this.parseExpression(element as Expression, parent, graph);
                break;

            case 'Expr':
                this.parseExpr(element as Expr, parent, graph, out as string);
                break;

            case 'Literal':
                this.parseLiteral(element as Literal, parent, graph, out as string);
                break;

            case 'BoxCall':
                this.parseBoxCall(element as BoxCall, parent, graph);
                break;

            case 'Relation':
                this.parseRelation(element as Relation, parent, graph);
                break;

            case 'Junction':
                this.parseJunction(element as Junction, parent, graph);
                break;

            case 'Loop':
                this.parseLoop(element as Loop, parent, graph);
                break;

            case 'Conditional':
                this.parseConditional(element as Conditional, parent, graph);
                break;

            default:
                throw `Unsupported element syntax [${element.syntax}]`;
        }
    }

    private parseElement(element: GrometElm, parent: string | null, graph: GraphSpec, out?: string): void {
        const elWithUID = element as unknown as { uid: string };
        if ('uid' in elWithUID) {
            const graphID = this.registerUID(elWithUID.uid);

            switch (element.syntax) {
                // these are all the element types that don't need to be pushed to the id stack
                case 'Wire':
                    this._parseElement(element, parent, graph, out);
                    break;

                default:
                    this.idStack.push(graphID);
                    this._parseElement(element, parent, graph, out);
                    this.idStack.pop();
                    break;
            }
        } else {
            this._parseElement(element, parent, graph, out);
        }
    }
}
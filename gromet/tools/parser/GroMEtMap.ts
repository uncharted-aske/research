/// <reference path="../types/GroMEt.d.ts" />

import Gromet = GroMEt.Gromet;

export type ElementMap<T> = Map<GroMEt.UidType, T>;
export type TypeMap = ElementMap<GroMEt.TypeDeclaration>;
export type PortMap = ElementMap<GroMEt.Port>;
export type WireMap = ElementMap<GroMEt.Wire>;
export type BoxMap = ElementMap<GroMEt.Box>;
export type JunctionMap = ElementMap<GroMEt.Junction>;
export type VarMap = ElementMap<GroMEt.Variable>;

export class GroMEtMap {
    protected gromet: GroMEt.Gromet;
    protected types: TypeMap | null;
    protected ports: PortMap | null;
    protected wires: WireMap | null;
    protected junctions: JunctionMap | null;
    protected boxes: BoxMap;
    protected vars: VarMap | null;

    constructor(gromet: Gromet) {
        this.gromet = gromet;
        this.types = gromet.types ? this.parseMap(gromet.types, 'name') : null;
        this.ports = gromet.ports ? this.parseMap(gromet.ports, 'uid') : null;
        this.wires = gromet.wires ? this.parseMap(gromet.wires, 'uid') : null;
        this.junctions = gromet.junctions ? this.parseMap(gromet.junctions, 'uid') : null;
        this.boxes = this.parseMap(gromet.boxes, 'uid');
        this.vars = gromet.variables ? this.parseMap(gromet.variables, 'uid') : null;
    }

    private parseMap<T>(arr: T[], key: keyof T): ElementMap<T> {
        const map = new Map();
        for (const entry of arr) {
            map.set(entry[key], entry);
        }
        return map;
    }
}

import { GroMEt2Graph } from './parser/GroMEt2Graph.ts';

const grometText = await Deno.readTextFile(Deno.args[0] ?? './gromet.json');
const graph = GroMEt2Graph.parseGromet(JSON.parse(grometText));
Deno.writeTextFile(Deno.args[1] ?? "./graph.json", JSON.stringify(graph, null, 2));

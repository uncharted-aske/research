import {loadFileJSON} from './JSONL.js';
import {GroMEt2Graph} from '../parser/GroMEt2Graph.ts';

async function main(inputFile: string, outputFile: string): Promise<void> {
    const gromet: GroMEt.Gromet = await loadFileJSON(inputFile);
    const graph = GroMEt2Graph.parseGromet(gromet);
    // console.log(JSON.stringify(graph));

    console.log('Writing graph JSON file...');
    await Deno.writeTextFile(outputFile, JSON.stringify(graph));
}

main(...Deno.args as [string, string]);
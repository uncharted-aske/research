import {DataFile} from 'https://cdn.skypack.dev/@dekkai/data-source';

interface ParsingState {
    content: string;
    writeIndent: number;
    readIndent: string;
    state: ((line: string, state: ParsingState) => void)[];
}

const rxMultilineComment = /^(\s*)(['|"]{3})(.*)/;
const rxInlineComment = /(.*)#\s*(.+)$/;
const rxStandaloneComment = /^\s*#(.+)$/;
const rxClassDeclaration = /^(\s*)class ([^(]+)(?:\((.+)\))?:/;
const rxIndentCheck = /^(\s*).+$/;
const rxBlankLine = /^\s*$/;
const rxTypeDeclaration = /^.+=\s*NewType\s*\(\s*["|'](\S+)["|']\s*,\s*(.+)\s*\)/;
const rxUnionType = /^\s*Union\[(.+)]\s*$/;
const rxListType = /^\s*List\[(.+)]\s*$/;
const rxTupleType = /^\s*Tuple\[(.+)]\s*$/;
const rxClassProperty = /^\s*(\S+)\s*:\s*([^=#]+)/;
// const rxSplitList = /(?!\[[^\]]+),(?![^[]+])/;

export async function parseLines(inputPath: string, cb: (s: string) => void) {
    const file = await DataFile.fromLocalSource(inputPath);

    // load 16MB chunks
    const sizeOf16MB = 16 * 1024 * 1024;
    const byteLength = await file.byteLength;
    const decoder = new TextDecoder();
    const lineBreak = '\n'.charCodeAt(0);

    for(let offset = 0; offset <= byteLength; offset += sizeOf16MB) {
        const chunkEnd = Math.min(offset + sizeOf16MB, byteLength);
        const chunk = await file.loadData(offset, chunkEnd) as ArrayBuffer;
        const view = new DataView(chunk);
        let start = 0;
        let count = 0;
        for (let i = 0, n = chunk.byteLength; i < n; ++i) {
            if (view.getUint8(i) === lineBreak || offset + i === byteLength) {
                const statementBuffer = new Uint8Array(chunk, start, i - start);
                start = i + 1;
                ++count;

                const str = decoder.decode(statementBuffer);
                await cb(str);
            }
        }

        if (start < chunk.byteLength) {
            offset -= chunk.byteLength - start;
        }

        console.log(`${chunkEnd} / ${byteLength} - ${((chunkEnd/byteLength) * 100).toFixed(2)}%`);
    }
}

function splitList(list: string): string[] {
    // doing this in a regex would be a nightmare
    const result = [];
    let start = 0;
    let depth = 0;
    for (let i = 0, n = list.length; i < n; ++i) {
        if (list[i] === '[') {
            ++depth;
        } else if (list[i] === ']') {
            --depth;
        } else if (list[i] === ',' && depth == 0) {
            result.push(list.substr(start, i - start));
            start = i + 1;
        }
    }
    result.push(list.substr(start));
    return result;
}

function convertType(pyType: string): string {
    const union = rxUnionType.exec(pyType);
    if (union) {
        return splitList(union[1]).map(t => convertType(t)).join(' | ');
    }

    const tuple = rxTupleType.exec(pyType);
    if (tuple) {
        return `[${splitList(tuple[1]).map(t => convertType(t)).join(', ')}]`;
    }

    const list = rxListType.exec(pyType);
    if (list) {
        if (list[1].indexOf(',') === -1) {
            return `${list[1].trim()}[]`;
        } else {
            const elements = splitList(list[1]);
            return `Array<${elements.map(t => convertType(t)).join(', ')}>`;
        }
    }

    const clean = pyType.trim().replace(/['"]?([^'"]+)['"]?/g, '$1');
    switch (clean) {
        case 'str':
            return 'string';

        case 'object':
            return 'any';

        case 'None':
            return 'null';

        default:
            return clean;
    }
}

function addInlineComment(text: string, line: string): string {
    const comment = rxInlineComment.exec(line);
    if (comment && comment[2]) {
        return `${text} // ${comment[2]}`;
    }
    return text;
}

function processMultilineComment(line: string, state: ParsingState): boolean {
    const comment = rxMultilineComment.exec(line);
    if (comment) {
        // add the start of the multiline comment
        state.content += makeLine('/**', state.writeIndent);
        // if there is content besides the comment delimiter add it to the next line
        if (comment[3]) {
            state.content += makeLine(` * ${comment[3]}`, state.writeIndent);
        }
        // set the state to multiline comment
        state.state.push(stateMultilineComment);
        return true;
    }
    return false;
}

function processStandaloneComment(line: string, state: ParsingState): boolean {
    // check for standalone comments
    const comment = rxStandaloneComment.exec(line);
    if (comment) {
        state.content += makeLine(`// ${comment[1]}`, state.writeIndent);
        return true;
    }
    return false;
}

function processClassProperty(line: string, state: ParsingState): boolean {
    const property = rxClassProperty.exec(line);
    if (property) {
        const content = addInlineComment(`${property[1]}: ${convertType(property[2])};`, line);
        state.content += makeLine(content, state.writeIndent);
    }
    return false;
}

function processClassDeclaration(line: string, state: ParsingState): boolean {
    // check for class declarations
    const declaration = rxClassDeclaration.exec(line);
    if (declaration) {
        const content = `interface ${declaration[2]} ${declaration[3] && declaration[3] !== 'object' ? `extends ${declaration[3]} ` : ''}{`;
        state.content += makeLine(addInlineComment(content, line), state.writeIndent++);
        state.state.push(stateClassDeclaration);
        return true;
    }
    return false;
}

function processTypeDeclaration(line: string, state: ParsingState): boolean {
    // check for type declarations
    const declaration = rxTypeDeclaration.exec(line);
    if (declaration) {
        const content = addInlineComment(`type ${declaration[1]} = ${convertType(declaration[2])};`, line);
        state.content += makeLine(content, state.writeIndent);
        return true;
    }
    return false;
}

function stateMultilineComment(line: string, state: ParsingState): void {
    if (rxBlankLine.exec(line)) {
        state.content += makeLine(' *', state.writeIndent);
    }

    const commentEnd = rxMultilineComment.exec(line);
    if (commentEnd) {
        // throw if there is text after the comment delimiter
        if (commentEnd[3]) {
            throw `Found text after multiline comment closing delimiter: [${commentEnd[3]}]`;
        }
        state.content += makeLine(' */', state.writeIndent);
        state.state.pop();
        return;
    }

    state.content += makeLine(` * ${line.substr(state.readIndent.length)}`, state.writeIndent);
}

function stateClassDeclaration(line: string, state: ParsingState): void {
    const indentCheck = rxIndentCheck.exec(line);
    if (indentCheck) {
        if (indentCheck[1].length > state.readIndent.length) {
            state.readIndent = indentCheck[1];
        } else if (indentCheck[1].length < state.readIndent.length) {
            state.readIndent = indentCheck[1];
            state.content += makeLine('}\n', --state.writeIndent);
            state.state.pop();
            runState(state, line);
            return;
        }
    }
    if (processMultilineComment(line, state)) return;
    if (processStandaloneComment(line, state)) return;
    if (processClassProperty(line, state)) return;
}

function stateUnknown(line: string, state: ParsingState): void {
    if (rxBlankLine.exec(line)) {
        state.content += makeLine('', state.writeIndent);
        return;
    }

    if (processMultilineComment(line, state)) return;
    if (processStandaloneComment(line, state)) return;
    if (processClassDeclaration(line, state)) return;
    if (processTypeDeclaration(line, state)) return;
}

function makeIndent(indent: number): string {
    return '    '.repeat(indent);
}

function makeLine(text: string, indent: number): string {
    return `${makeIndent(indent)}${text}\n`;
}

function runState(state: ParsingState, line: string): void {
    state.state[state.state.length - 1](line, state);
}

async function main(inputFile: string, outputFile: string): Promise<void> {
    const state: ParsingState = {
        content: '',
        writeIndent: 0,
        readIndent: '',
        state: [stateUnknown],
    };

    // add the namespace
    state.content += makeLine('declare namespace GroMEt {', state.writeIndent++);

    await parseLines(inputFile, line => {
        // maintain spacing
        runState(state, line);
    });

    // close the namespace
    state.content += makeLine('}', --state.writeIndent);

    console.log('Writing type definition file...');
    await Deno.writeTextFile(outputFile, state.content);
}

main(...Deno.args as [string, string]);


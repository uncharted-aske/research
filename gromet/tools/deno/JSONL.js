import {DataFile} from 'https://cdn.skypack.dev/@dekkai/data-source';
import LosslessJSON from 'https://cdn.skypack.dev/lossless-json';

export async function loadFileJSON(inputPath) {
    const file = await DataFile.fromLocalSource(inputPath);
    const data = await file.loadData();
    const decoder = new TextDecoder();
    const str = decoder.decode(data);
    return JSON.parse(str);
}

export async function loadFileLosslessJSON(inputPath) {
    const file = await DataFile.fromLocalSource(inputPath);
    const data = await file.loadData();
    const decoder = new TextDecoder();
    const str = decoder.decode(data);
    return LosslessJSON.parse(str);
}

export async function loadFileJSONL(file, idKey = 'id', cb = null) {
    const map = new Map();
    let lineNumber = 0;
    await parseJSONL(file, json => {
        if (lineNumber++) {
            map.set(json[idKey], cb ? cb(json) : json);
        }
    });
    return map;
}

export async function _parseJSONL(lossless, inputPath, cb) {
    const file = await DataFile.fromLocalSource(inputPath);

    // load 16MB chunks
    const sizeOf16MB = 16 * 1024 * 1024;
    const byteLength = await file.byteLength;
    const decoder = new TextDecoder();
    const lineBreak = '\n'.charCodeAt(0);
    const JSONParser = lossless ? LosslessJSON : JSON;

    for(let offset = 0; offset <= byteLength; offset += sizeOf16MB) {
        const chunkEnd = Math.min(offset + sizeOf16MB, byteLength);
        const chunk = await file.loadData(offset, chunkEnd);
        const view = new DataView(chunk);
        let start = 0;
        let count = 0;
        for (let i = 0, n = chunk.byteLength; i < n; ++i) {
            if (view.getUint8(i) === lineBreak || offset + i === byteLength) {
                const statementBuffer = new Uint8Array(chunk, start, i - start);
                start = i + 1;
                ++count;

                const str = decoder.decode(statementBuffer);
                const json = JSONParser.parse(str);

                await cb(json);
            }
        }

        if (start < chunk.byteLength) {
            offset -= chunk.byteLength - start;
        }

        console.log(`${chunkEnd} / ${byteLength} - ${((chunkEnd/byteLength) * 100).toFixed(2)}%`);
    }
}

export async function parseJSONL(inputPath, cb) {
    await _parseJSONL(false, inputPath, cb);
}

export async function parseLosslessJSONL(inputPath, cb) {
    await _parseJSONL(true, inputPath, cb);
}

export function makeJSONL(entries) {
    let result = '';
    for (const entry of entries) {
        result += `${JSON.stringify(entry)}\n`;
    }
    return result;
}

export async function openWriteJSONL(path) {
    const handle = await Deno.open(path, { write: true, create: true, truncate: true });
    return {
        length: 0,
        handle,
    };
}

const textEncoder = new TextEncoder();
export async function writeEntryJSONL(file, entry) {
    await Deno.write(file.handle.rid, textEncoder.encode(`${JSON.stringify(entry)}\n`));
    ++file.length;
}

export async function writeEntriesJSONL(file, entries) {
    const promises = [];
    for (let i = 0, n = entries.length; i < n; ++i) {
        promises.push(writeEntryJSONL(file, entries[i]));
    }
    await Promise.all(promises);
}

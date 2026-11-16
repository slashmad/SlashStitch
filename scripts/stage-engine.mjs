import { copyFileSync, chmodSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import {createHash} from "node:crypto";
import { join } from "node:path";

const exe = process.platform === "win32" ? "slashstitch-worker.exe" : "slashstitch-worker";
const candidates = [join("engine", "target", "release", exe)];
const source = candidates.find(p=>existsSync(p));
if (!source) {
  throw new Error(`Missing Rust worker binary. Run "npm run build:engine" first. Checked: ${candidates.join(", ")}`);
}

const bytes=readFileSync(source);
const metadataPath=join('engine','target','release','engine-build.json');
if(!existsSync(metadataPath))throw new Error('Missing verified engine-build.json. Run npm run build:engine.');
const metadata=JSON.parse(readFileSync(metadataPath,'utf8'));
if(createHash('sha256').update(bytes).digest('hex')!==metadata.sha256)throw new Error('Worker differs from verified release build');
if(process.platform==='linux'&&(metadata.glibc_max!=='2.28'||bytes.subarray(0,4).toString('hex')!=='7f454c46'))throw new Error('Linux worker must be the verified GLIBC 2.28 ELF build');
if(process.platform==='win32'&&bytes.subarray(0,2).toString()!=='MZ')throw new Error('Expected a Windows worker');
const outDir = join("build", "bin");
mkdirSync(outDir, { recursive: true });
const target = join(outDir, exe);
copyFileSync(source, target);
copyFileSync(metadataPath,join(outDir,"engine-build.json"));
if (process.platform !== "win32") chmodSync(target, 0o755);
console.log(`Staged ${source} -> ${target}`);

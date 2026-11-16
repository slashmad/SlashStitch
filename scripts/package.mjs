// Each installer is built on its own OS with its verified native Rust worker.
import {spawnSync} from 'node:child_process';
import {existsSync,readFileSync} from 'node:fs';
import {join,resolve} from 'node:path';
const platform=process.argv.slice(2).find(a=>!a.startsWith('--'))||process.platform;
const directoryOnly=process.argv.includes('--dir');
if(!['linux','win32','darwin'].includes(platform)||platform!==process.platform)throw new Error(`Build ${platform} packages on ${platform}; cross-packaging a host worker is refused.`);
function run(command,args,env=process.env){const r=spawnSync(command,args,{stdio:'inherit',env});if(r.error)throw r.error;if(r.status!==0)process.exit(r.status??1);}
for(const script of ['build-engine.mjs','stage-engine.mjs','build-notices.mjs','build-main.mjs','build-renderer.mjs'])run(process.execPath,[join('scripts',script)]);
// Exercise the actual optimized/staged binary, not only Cargo's debug test worker.
const worker=resolve('build','bin',platform==='win32'?'slashstitch-worker.exe':'slashstitch-worker');
run(process.execPath,['--test','tests/worker.test.mjs'],{...process.env,SLASHSTITCH_TEST_WORKER:worker});
// Reuse npm ci's already downloaded, version-checked native Electron distribution.
// This avoids a second download and makes offline packaging possible after setup.
const electronDist=resolve('node_modules/electron/dist');
const installed=readFileSync(join(electronDist,'version'),'utf8').trim().replace(/^v/,'');
const expected=JSON.parse(readFileSync('package.json','utf8')).devDependencies.electron;
if(installed!==expected)throw new Error(`Electron version mismatch: installed ${installed}, expected ${expected}`);
const binary=platform==='win32'?'electron.exe':platform==='darwin'?'Electron.app/Contents/MacOS/Electron':'electron';
if(!existsSync(join(electronDist,binary)))throw new Error('The installed Electron distribution is for a different OS');
const cli=resolve('node_modules/electron-builder/out/cli/cli.js');
run(process.execPath,[cli,...(directoryOnly?['--dir']:[platform==='win32'?'--win':platform==='darwin'?'--mac':'--linux']),`--${process.arch}`,'--publish','never',`--config.electronDist=${electronDist}`],{...process.env,SLASHSTITCH_ARCH:process.arch});

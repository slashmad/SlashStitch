import {spawnSync} from 'node:child_process';
import {existsSync,copyFileSync,mkdirSync,writeFileSync} from 'node:fs';
import {delimiter,join,resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
const local=resolve('.build-tools');
const hostInfo=spawnSync('rustc',['-vV'],{encoding:'utf8'});
const rustTarget=hostInfo.stdout?.match(/^host: (.+)$/m)?.[1];
if(!rustTarget)throw new Error('Cannot identify the native Rust toolchain');
const rustArch=rustTarget.startsWith('aarch64-')?'arm64':rustTarget.startsWith('x86_64-')?'x64':null;
if(rustArch!==process.arch)throw new Error(`Rust toolchain ${rustTarget} does not match Node ${process.arch}; use native tools for packaging`);
const target=process.platform==='linux'?`${process.arch==='arm64'?'aarch64':'x86_64'}-unknown-linux-gnu`:null;
const args=['build','--manifest-path','engine/Cargo.toml','--release','--locked','--bin','slashstitch-worker'];
const env={...process.env};
let command='cargo';
if(target){
 const zigbuild=join(local,'bin','cargo-zigbuild');
 const pythonZig=spawnSync(join(local,'python','bin','python'),['-c','import pathlib,ziglang;print(pathlib.Path(ziglang.__file__).parent)'],{encoding:'utf8'});
 env.PATH=[join(local,'bin'),...(pythonZig.status===0?[pythonZig.stdout.trim()]:[]),env.PATH].join(delimiter);
 env.CARGO_ZIGBUILD_CACHE_DIR=resolve('.build-tools/zigbuild-cache');
 env.ZIG_GLOBAL_CACHE_DIR=resolve('.build-tools/zig-global');
 env.ZIG_LOCAL_CACHE_DIR=resolve('.build-tools/zig-local');
 command=existsSync(zigbuild)?zigbuild:'cargo-zigbuild';
 args.push('--target',`${target}.2.28`);
}
const built=spawnSync(command,args,{env,stdio:'inherit'});
if(built.error)throw new Error(`${built.error.message}. Linux release builds require cargo-zigbuild and Zig (see README).`);
if(built.status!==0)process.exit(built.status??1);
const exe=process.platform==='win32'?'slashstitch-worker.exe':'slashstitch-worker';
const source=join('engine','target',...(target?[target]:[]),'release',exe);
const out=join('engine','target','release');mkdirSync(out,{recursive:true});
if(target)copyFileSync(source,join(out,exe));
writeFileSync(join(out,'engine-build.json'),JSON.stringify({target:target??rustTarget,platform:process.platform,arch:process.arch,glibc_max:target?'2.28':null,sha256:createHash('sha256').update(readFileSync(source)).digest('hex')},null,2)+'\n');
if(target){
 const info=spawnSync('objdump',['-T',source],{encoding:'utf8'});
 if(info.status!==0)throw new Error('objdump is required to verify the Linux worker ABI');
 for(const match of info.stdout.matchAll(/GLIBC_(\d+)\.(\d+)/g)){
  if(Number(match[1])>2||Number(match[1])===2&&Number(match[2])>28)throw new Error(`Worker exceeds declared GLIBC 2.28 baseline: ${match[0]}`);
 }
 console.log('Verified worker ABI: GLIBC <= 2.28');
}

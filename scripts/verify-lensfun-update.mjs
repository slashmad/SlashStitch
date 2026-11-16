// Optional online validation: never activates or writes a user/system database.
import {mkdtempSync,rmSync,mkdirSync,writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {build} from 'esbuild';
import {createRequire} from 'node:module';
import {WorkerClient} from '../tests/helpers/worker-client.mjs';
const temporary=mkdtempSync(join(tmpdir(),'slashstitch-lensfun-validation-'));
const worker=new WorkerClient(temporary);
try{
 await build({entryPoints:['src/main/lensfun-update.ts'],outfile:join(temporary,'updater.cjs'),bundle:true,platform:'node',format:'cjs',logLevel:'silent'});
 const require=createRequire(import.meta.url);const{downloadLensfunSnapshot}=require(join(temporary,'updater.cjs'));
 let parsed;
 const snapshot=await downloadLensfunSnapshot(join(temporary,'snapshots'),async path=>{parsed=await worker.success('validate-lensfun',{path});});
 const result={sha:snapshot.sha,files:snapshot.files,parsed,activated:false};
 const output=resolve('build/validation');mkdirSync(output,{recursive:true});writeFileSync(join(output,'lensfun-online.json'),JSON.stringify(result,null,2)+'\n');
 console.log(JSON.stringify(result));
}finally{await worker.close();rmSync(temporary,{recursive:true,force:true});}

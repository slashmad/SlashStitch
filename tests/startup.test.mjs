import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,readFileSync,rmSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {createRequire} from 'node:module';
import {build} from 'esbuild';
const temporary=mkdtempSync(join(tmpdir(),'slashstitch-startup-'));
await build({entryPoints:['src/main/worker.ts'],outfile:join(temporary,'bridge.cjs'),bundle:true,platform:'node',format:'cjs',logLevel:'silent'});
const {WorkerBridge}=createRequire(import.meta.url)(join(temporary,'bridge.cjs'));
test.after(()=>rmSync(temporary,{recursive:true,force:true}));
let serial=0;
const fixture=(mode,extra={},timeoutMs=3000)=>{
 const trace=join(temporary,`${++serial}.jsonl`);
 const worker=new WorkerBridge(process.execPath,[resolve('tests/startup-worker-fixture.cjs')],{FIXTURE_MODE:mode,FIXTURE_TRACE:trace,...extra},{probeAction:'engine-info',timeoutMs,fallbackEnv:{FIXTURE_CPU:'1'},retryExitCodes:[73]});
 return {worker,trace:()=>readFileSync(trace,'utf8').trim().split('\n').map(JSON.parse)};
};
test('startup crash retries once with CPU before forwarding any user mutations',async()=>{
 const {worker,trace}=fixture('crash');const events=[];worker.on('event',(name,data)=>events.push({name,data}));
 try{
  const [mutation,info]=await Promise.all([worker.request('increment',{value:7}),worker.request('engine-info')]);
  assert.equal(mutation.ok,true);assert.equal(info.ok,true);assert.equal(JSON.parse(info.text).backend,'CPU');
  const history=trace();assert.equal(history.filter(x=>x.start).length,2);
  assert.equal(history.filter(x=>x.mutation).length,1);assert(history.filter(x=>x.action==='increment').every(x=>x.cpu));
  assert.equal(events.filter(x=>x.name==='worker-recovering').length,1);assert.equal(events.filter(x=>x.name==='worker-exit').length,0);
 }finally{await worker.stop();}
});
test('successful GPU startup is kept and its later jobs are never automatically replayed',async()=>{
 const {worker,trace}=fixture('normal');
 assert.equal(JSON.parse((await worker.request('engine-info')).text).backend,'GPU');
 const exited=new Promise(resolve=>worker.on('event',name=>{if(name==='worker-exit')resolve();}));
 assert.equal((await worker.request('crash')).ok,false);await exited;
 assert.equal((await worker.request('increment',1)).ok,false);assert.equal(trace().filter(x=>x.start).length,1);
 await worker.stop();
});
test('CPU startup failure reports both attempts and does not loop',async()=>{
 const {worker,trace}=fixture('crash',{FIXTURE_FAIL_CPU:'1'});
 const reply=await worker.request('increment',1);assert.equal(reply.ok,false);
 assert.match(reply.text,/simulated native GPU crash/);assert.match(reply.text,/CPU probe failed/);
 assert.equal(trace().filter(x=>x.start).length,2);assert(!trace().some(x=>x.mutation));
 assert.equal((await worker.request('engine-info')).ok,false);assert.equal(trace().filter(x=>x.start).length,2);
 await worker.stop();
});
test('a hung read-only startup probe is stopped before the CPU replacement starts',async()=>{
 const {worker,trace}=fixture('hang',{},1000);
 try{
  const reply=await worker.request('increment',3);assert.equal(reply.ok,true);
  const starts=trace().filter(x=>x.start);assert.equal(starts.length,2);assert.throws(()=>process.kill(starts[0].start,0));
  assert.equal(trace().filter(x=>x.mutation).length,1);
 }finally{await worker.stop();}
});
test('closing during startup does not start a CPU replacement or run a queued mutation',async()=>{
 const {worker,trace}=fixture('hang');const request=worker.request('increment',9);
 const end=Date.now()+3000;while(Date.now()<end){try{if(trace().some(x=>x.action==='engine-info'))break;}catch{}await new Promise(r=>setTimeout(r,10));}
 await worker.stop();assert.equal((await request).ok,false);assert.equal(trace().filter(x=>x.start).length,1);assert(!trace().some(x=>x.mutation));
});
test('a missing executable is reported without a misleading CPU retry',async()=>{
 const worker=new WorkerBridge(join(temporary,'missing'),[],{},{probeAction:'engine-info',timeoutMs:3000,fallbackEnv:{FIXTURE_CPU:'1'},retryExitCodes:[73]});
 let retries=0;worker.on('event',name=>{if(name==='worker-recovering')retries++;});
 const reply=await worker.request('engine-info');assert.equal(reply.ok,false);assert.match(reply.text,/could not start/);assert.equal(retries,0);await worker.stop();
});
test('a normal restart after a CPU restart starts with the configured (GPU) environment again',async()=>{
 const {worker,trace}=fixture('normal');
 try{
  assert.equal(JSON.parse((await worker.request('engine-info')).text).backend,'GPU');
  const exited=()=>new Promise(resolve=>worker.once('event',function on(name){if(name==='worker-exit')resolve();else worker.once('event',on);}));
  let gone=exited();assert.equal((await worker.request('crash')).ok,false);await gone;
  worker.restart({FIXTURE_CPU:'1'});assert.equal(JSON.parse((await worker.request('engine-info')).text).backend,'CPU');
  gone=exited();assert.equal((await worker.request('crash')).ok,false);await gone;
  worker.restart();assert.equal(JSON.parse((await worker.request('engine-info')).text).backend,'GPU');
  assert.equal(trace().filter(x=>x.start).length,3);
 }finally{await worker.stop();}
});

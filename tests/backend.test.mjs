import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {createRequire} from 'node:module';
import {once} from 'node:events';
import {build} from 'esbuild';
const temporary=mkdtempSync(join(tmpdir(),'slashapp-tests-'));
const require=createRequire(import.meta.url);
await build({entryPoints:['src/main/worker.ts','src/main/lifecycle.ts','src/main/demo-worker.ts'],outdir:temporary,bundle:true,platform:'node',format:'cjs',logLevel:'silent'});
const {WorkerBridge}=require(join(temporary,'worker.js'));
const {ShutdownGate}=require(join(temporary,'lifecycle.js'));
const fixture=()=>new WorkerBridge(process.execPath,[resolve('tests/worker-fixture.cjs')]);
test.after(()=>rmSync(temporary,{recursive:true,force:true}));

test('invalid messages do not crash; later valid requests still work',async()=>{
 const worker=fixture();let errors=0;
 worker.on('event',name=>{if(name==='protocol-error')errors++;});
 try {
  assert.deepEqual(await worker.request('malformed'),{ok:true,text:'"survived"'});
  assert.equal(errors,6);
  assert.equal(JSON.parse((await worker.request('echo',{value:42})).text).value,42);
 } finally {await worker.stop();}
});
test('timeouts settle requests without cancelling the worker',async()=>{
 const worker=fixture();
 try {
  assert.match((await worker.request('ignore',null,50)).text,/timed out/);
  assert.equal((await worker.request('echo','still alive')).ok,true);
 } finally {await worker.stop();}
});
test('closed stdin produces a settled failure instead of unhandled EPIPE',async()=>{
 const worker=fixture();
 const closed=once(worker,'event').then(()=>{});
 const exited=new Promise(resolve=>worker.on('event',name=>{if(name==='worker-exit')resolve();}));
 await worker.request('close-input');await closed;
 assert.equal((await worker.request('echo')).ok,false);
 await exited;
 await worker.stop();
});
test('missing executable reports a failure',async()=>{
 const worker=new WorkerBridge(join(temporary,'does-not-exist'),[]);
 assert.equal((await worker.request('echo')).ok,false);
 await new Promise(resolve=>setTimeout(resolve,30));
});
test('backend-owned descendants are reaped before shutdown is acknowledged',async()=>{
 const worker=fixture();
 const pid=JSON.parse((await worker.request('descendant')).text);
 assert.doesNotThrow(()=>process.kill(pid,0));
 await worker.stop();
 assert.throws(()=>process.kill(pid,0));
});
test('demo rejects shutdown while active and emits finalization before idle',async()=>{
 const worker=new WorkerBridge(process.execPath,[join(temporary,'demo-worker.js')]);
 const states=[];worker.on('event',(name,data)=>{if(name==='job-state')states.push(data.state);});
 try {
  assert.equal((await worker.request('demo-start')).ok,true);
  await assert.rejects(worker.stop(),/Stop the active job/);
  assert.equal((await worker.request('demo-stop')).ok,true);
  assert.deepEqual(states,['running','finalizing','idle']);
 } finally {await worker.stop();}
});
test('Close/Quit gate blocks running and finalizing states; duplicate quit waits once',async()=>{
 let state='running',calls=0,finish;
 const gate=new ShutdownGate(()=>state!=='idle',()=>{calls++;return new Promise(resolve=>{finish=resolve;});},()=>assert.fail('unexpected shutdown error'));
 assert.equal(await gate.request(),false);state='finalizing';assert.equal(await gate.request(),false);assert.equal(calls,0);
 state='idle';const first=gate.request();const second=gate.request();assert.equal(first,second);assert.equal(gate.stopping,true);
 await Promise.resolve();assert.equal(calls,1);finish();assert.equal(await first,true);assert.equal(gate.allowed,true);
});
test('failed graceful shutdown keeps the application open',async()=>{
 let reported='';const gate=new ShutdownGate(()=>false,async()=>{throw new Error('not finalized');},error=>{reported=error.message;});
 assert.equal(await gate.request(),false);assert.equal(gate.allowed,false);assert.equal(gate.stopping,false);assert.equal(reported,'not finalized');
});
